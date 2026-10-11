"""Generation utilities for FramerAI inference."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .configs import FramerConfig
from .framer import FramerModel
from .modules.kv_cache import truncate_cache
from .tokenizer import FramerTokenizer
from .utils.image_request import resolve_image_request

# Prompt tokens pushed through the KV cache per prefill step. Bounded so a very
# long prompt costs the same activation memory as a short one.
DEFAULT_PREFILL_CHUNK = 4096

# Tokens a draft model proposes per verification step.
DEFAULT_SPECULATE_K = 4


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Apply top-k then nucleus (top-p) filtering to a (1, V) logits row."""
    if top_k and top_k > 0:
        kth = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        idx_remove = remove.scatter(1, sorted_idx, remove)
        logits = logits.masked_fill(idx_remove, float("-inf"))
    return logits


def _apply_repetition_penalty(
    logits: torch.Tensor,
    seen_tokens: list[int] | set[int],
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits for tokens that have already appeared.

    Standard semantics:
    - If logit < 0: multiply by penalty (making it more negative)
    - If logit > 0: divide by penalty (making it less positive)
    Does not mutate the original tensor.
    """
    if penalty is None or penalty == 1.0 or not seen_tokens:
        return logits
    if penalty <= 0:
        raise ValueError(f"repetition_penalty must be positive, got {penalty}")

    vocab_size = logits.shape[-1]
    token_ids = [int(t) for t in set(seen_tokens) if 0 <= t < vocab_size]
    if not token_ids:
        return logits

    logits = logits.clone()
    idx = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    scores = logits[0, idx]

    pos_mask = scores > 0
    neg_mask = scores < 0
    penalized = scores.clone()
    penalized[pos_mask] = scores[pos_mask] / penalty
    penalized[neg_mask] = scores[neg_mask] * penalty

    logits[0, idx] = penalized
    return logits


def _make_seed_generator(device, seed: int | None) -> torch.Generator | None:
    """Create a seeded torch.Generator on device, falling back to default device if unsupported."""
    if seed is None:
        return None
    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except (RuntimeError, TypeError):
        return torch.Generator().manual_seed(int(seed))


class FramerGenerator:
    """High-level generation interface for FramerAI."""

    def __init__(self, model: FramerModel, tokenizer: FramerTokenizer, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        # What the last call had to give up to fit the context window. Reported
        # rather than hidden, so a short or truncated answer is explicable.
        self.last_prompt_tokens_dropped = 0
        self.last_max_new_tokens = 0
        self.last_finish_reason = None
        # The text the last generate_text call produced, without the prompt it
        # was given. generate_text itself returns prompt and continuation
        # together, which callers that parse prompts rely on; a reply sent to a
        # user must be this instead.
        self.last_completion = ""
        # What the last call consumed and produced, in tokens. Billing reads
        # these, so they count positions the backbone actually ran: the prompt
        # after any trimming plus any modality prefix, and every sampled token.
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        # Of the input tokens, how many came from a reused prompt cache.
        self.last_cached_input_tokens = 0
        # Optional serving accelerators, both off unless a caller attaches
        # them: a PrefixCache, and a smaller FramerGenerator sharing this
        # tokenizer that drafts tokens for speculative decoding.
        self.prefix_cache = None
        self.draft = None
        self.speculate_k = DEFAULT_SPECULATE_K
        self.last_draft_accepted = 0
        self.last_draft_proposed = 0

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, tokenizer_path: str, device: str = "auto"):
        """Load model from checkpoint."""
        from .utils import get_device
        device = get_device(device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", FramerConfig())
        if isinstance(config, dict):
            config = FramerConfig.from_dict(config)

        model = FramerModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        tokenizer = FramerTokenizer.load(tokenizer_path)
        return cls(model, tokenizer, str(device))

    def _text_context(self, prompt: str) -> torch.Tensor:
        """Encode a prompt to the LM's hidden states for conditioning generators."""
        tokens = self.tokenizer.encode(prompt, add_special=True)
        input_ids = torch.tensor([tokens], device=self.device)
        return self.model.forward_lm(input_ids)["hidden"]

    def _chunk_modality_embeds(self, piece, modality_embeds, consumed):
        """The slice of each modality's embeddings whose placeholders are here.

        Chunked prefill hands the backbone a window of the prompt at a time,
        and the scatter requires exactly as many embeddings as the window has
        placeholder tokens. So the embeddings are walked alongside the chunks
        rather than passed whole to every one of them.
        """
        chunk_embeds = {}
        for placeholder_id, embeds in modality_embeds.items():
            slots = int((piece == placeholder_id).sum().item())
            if not slots:
                continue
            flat = embeds.reshape(-1, embeds.shape[-1])
            taken = consumed[placeholder_id]
            chunk_embeds[placeholder_id] = flat[taken:taken + slots].unsqueeze(0)
            consumed[placeholder_id] = taken + slots
        return chunk_embeds

    def _prefill(self, input_ids, prefix_embeds=None, chunk_size=None, modality_embeds=None):
        """Seed the KV cache from a prompt, one chunk at a time.

        A single-shot prefill runs the whole prompt through the backbone in one
        forward, which materialises a logits row for every prompt position: at a
        million tokens that is a (1, 1048576, vocab_size) tensor built before the
        first token is ever sampled. Walking the prompt in chunks through the
        cache bounds activations by the chunk size and keeps only the last logits
        row, which is the only one generation reads. Attention is offset-aware
        against a cache, so the result is identical either way.

        Returns the populated cache and the logits for the final prompt position.
        """
        chunk = chunk_size or DEFAULT_PREFILL_CHUNK
        self.last_input_tokens = int(input_ids.shape[1]) + (
            int(prefix_embeds.shape[1]) if prefix_embeds is not None else 0
        )
        self.last_output_tokens = 0
        self.last_cached_input_tokens = 0

        # Only a text prompt is cacheable: modality positions are not tokens,
        # so equal ids do not mean equal context.
        prefix_cache = getattr(self, "prefix_cache", None)
        cacheable = (
            prefix_cache is not None
            and prefix_embeds is None
            and not modality_embeds
            and input_ids.shape[0] == 1
        )
        past, reused = None, 0
        if cacheable:
            tokens = input_ids[0].tolist()
            reused, past = prefix_cache.lookup(tokens)
            self.last_cached_input_tokens = reused

        remaining = input_ids[:, reused:]
        starts = list(range(0, remaining.shape[1], chunk)) or [0]
        logits = None
        consumed = {placeholder: 0 for placeholder in (modality_embeds or {})}
        for start in starts:
            piece = remaining[:, start:start + chunk]
            chunk_embeds = (
                self._chunk_modality_embeds(piece, modality_embeds, consumed)
                if modality_embeds
                else None
            )
            out = self.model.forward_lm(
                piece,
                prefix_embeds=prefix_embeds if start == 0 else None,
                modality_embeds=chunk_embeds,
                past_kvs=past,
                use_cache=True,
            )
            past = out["past_kvs"]
            logits = out["logits"][:, -1, :]

        if cacheable:
            prefix_cache.store(tokens, past)
        return past, logits

    def _fit_to_window(
        self,
        prompt: str,
        max_new_tokens: int,
        modality_embeds: list,
        allowed_special=None,
    ):
        """Fit a request inside the context window, and record what it cost.

        The window has to hold the prompt, whatever the modalities occupy, and
        everything about to be generated. Nothing used to reserve any of that,
        so an over-long request reached the backbone and produced positions the
        model was never trained for, returning degraded output with no
        diagnostic. Here the room is divided up front: a share is held back for
        the answer, the prompt is trimmed to the rest keeping its end, and the
        generation length is clamped to what is actually left.

        Both decisions land on ``last_prompt_tokens_dropped`` and
        ``last_max_new_tokens`` so a caller can say why an answer came out short
        rather than guessing.
        """
        budget = self._window_budget(max_new_tokens, modality_embeds)
        if budget is None:
            return prompt, max_new_tokens
        prompt_budget, room = budget

        ids = self.tokenizer.encode(prompt, add_special=False, allowed_special=allowed_special)
        ids, max_new_tokens = self._fit_ids(ids, max_new_tokens, prompt_budget, room)
        if self.last_prompt_tokens_dropped:
            prompt = self.tokenizer.decode(ids)
        return prompt, max_new_tokens

    def _window_budget(self, max_new_tokens: int, modality_embeds: list):
        """How many tokens the prompt may occupy, or None with no window set.

        Returns ``(prompt_budget, room)``: what the prompt may use, and the
        total left after the modalities and the start and end tokens.
        """
        self.last_prompt_tokens_dropped = 0
        self.last_max_new_tokens = max_new_tokens

        window = int(getattr(self.model.config, "max_seq_len", 0) or 0)
        if not window:
            return None

        modality_tokens = sum(e.shape[1] for e in modality_embeds)
        # Interleaved placement writes an opening and closing marker per
        # modality into the sequence; prefix placement does not.
        markers = 2 * len(modality_embeds) if self._interleaving() else 0
        room = window - modality_tokens - markers - 2  # start and end tokens

        if room < 2:
            raise ValueError(
                f"the attached modalities need {modality_tokens + markers} of the "
                f"{window}-token context window, leaving no room for a prompt or "
                f"an answer. Send fewer modalities, lower the tile count, or use "
                f"a preset with a larger window."
            )

        # Hold back a share of the room for the answer, so a large
        # max_new_tokens cannot squeeze the prompt down to nothing and a long
        # prompt cannot leave the model with no room to reply.
        reserved = min(max_new_tokens, max(1, room // 4))
        return max(1, room - reserved), room

    def _fit_ids(self, ids: list, max_new_tokens: int, prompt_budget: int, room: int):
        """Trim prompt ids to the budget, keeping the end, and clamp the reply.

        Trimming ids rather than characters is what keeps a control marker
        whole: it is one token here, so a cut lands between markers instead of
        halfway through the bytes that spell one.
        """
        if len(ids) > prompt_budget:
            self.last_prompt_tokens_dropped = len(ids) - prompt_budget
            ids = ids[-prompt_budget:]

        self.last_max_new_tokens = max(1, min(max_new_tokens, room - len(ids)))
        return ids, self.last_max_new_tokens

    def _interleaving(self) -> bool:
        """True when the model asks for modalities placed, not prepended."""
        return getattr(self.model.config, "mm_token_placement", "prefix") == "interleaved"

    def _interleaved_prompt(
        self, prompt: str, image_embeds=None, audio_embeds=None, allowed_special=None
    ):
        """Build a prompt with modality runs where the prompt mentions them.

        Prefix concatenation puts every image ahead of the whole prompt, so
        "compare the first with the second" loses which is which. Here the
        ``<img>`` and ``<audio>`` markers in the prompt say where a modality
        belongs, and its embeddings are scattered into the placeholder run left
        at that position. A prompt with no marker keeps the old order, with the
        modality first, so nothing has to change to keep working.
        """
        from .data import InterleavedSequenceBuilder

        builder = InterleavedSequenceBuilder(self.tokenizer)
        segments = []

        # Find all markers with their positions to process them in order
        markers_info = []
        for marker, embeds, kind in (
            ("<img>", image_embeds, "image"),
            ("<audio>", audio_embeds, "audio"),
        ):
            if embeds is not None and marker in prompt:
                pos = prompt.find(marker)
                count = embeds.shape[1]
                markers_info.append((pos, marker, kind, count))

        # Sort by position to preserve order
        markers_info.sort(key=lambda x: x[0])

        # Build segments by processing markers in order
        remaining = prompt
        for _, marker, kind, count in markers_info:
            before, remaining = remaining.split(marker, 1)
            if before:
                segments.append(("text", before))
            segments.append((kind, count))

        # Add any remaining modalities that weren't mentioned (old behavior)
        for marker, embeds, kind in (
            ("<img>", image_embeds, "image"),
            ("<audio>", audio_embeds, "audio"),
        ):
            if embeds is not None and marker not in prompt:
                count = embeds.shape[1]
                segments.insert(0, (kind, count))

        # Add remaining text
        if remaining:
            segments.append(("text", remaining))

        built = builder.build(segments, allowed_special=allowed_special)
        tokens = [self.tokenizer.sos_id] + built["input_ids"] + [self.tokenizer.eos_id]

        modality_embeds = {}
        if image_embeds is not None:
            modality_embeds[builder.image_patch] = image_embeds
        if audio_embeds is not None:
            modality_embeds[builder.audio_frame] = audio_embeds
        return tokens, modality_embeds

    def attach_draft(self, draft, k: int = DEFAULT_SPECULATE_K):
        """Use ``draft``, a smaller FramerAI generator, for speculative decoding.

        The draft has to speak the same tokens, or its proposals would be ids
        that mean something else to the target.
        """
        if draft.model.config.vocab_size != self.model.config.vocab_size:
            raise ValueError(
                f"draft vocabulary {draft.model.config.vocab_size} does not match "
                f"target vocabulary {self.model.config.vocab_size}"
            )
        same = (
            draft.tokenizer.merges == self.tokenizer.merges
            and draft.tokenizer.marker_tokens == self.tokenizer.marker_tokens
        )
        if not same:
            raise ValueError("draft and target tokenizers differ")
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        self.draft = draft
        self.speculate_k = int(k)
        return self

    def _distribution(self, logits, temperature, top_k, top_p):
        """The sampling distribution over a (1, V) logits row, after filtering."""
        step = _filter_logits(logits / temperature, top_k, top_p)
        return F.softmax(step, dim=-1)

    def _pick(self, logits, generated, temperature, top_k, top_p, repetition_penalty, generator):
        step = _apply_repetition_penalty(logits, generated, repetition_penalty)
        if temperature <= 0:
            return step.argmax(dim=-1).item()
        probs = self._distribution(step, temperature, top_k, top_p)
        return torch.multinomial(probs, num_samples=1, generator=generator).item()

    def _can_speculate(self, input_ids, repetition_penalty, text_only=True) -> bool:
        """Whether the attached draft may propose tokens for this request.

        The draft sees token ids only, so a prompt carrying an image or audio
        is decoded plainly. A repetition penalty depends on the tokens accepted
        so far, which the draft's proposals would read out of order.
        """
        return (
            getattr(self, "draft", None) is not None
            and text_only
            and self.speculate_k > 0
            and input_ids.shape[0] == 1
            and (repetition_penalty is None or repetition_penalty == 1.0)
        )

    def _decode_tokens(self, past, logits, input_ids, generated, max_new_tokens,
                       temperature, top_k, top_p, repetition_penalty, generator,
                       text_only=True):
        """Yield sampled token ids, running the backbone between them.

        The caller appends each token to ``generated`` and decides when to stop;
        closing the iterator early is how an end marker or a stop string ends
        generation. No forward runs after the last token is handed out.
        """
        self.last_draft_accepted = 0
        self.last_draft_proposed = 0
        if self._can_speculate(input_ids, repetition_penalty, text_only):
            yield from self._speculative_tokens(
                past, logits, input_ids, max_new_tokens, temperature, top_k, top_p, generator
            )
            return

        for i in range(max_new_tokens):
            next_token = self._pick(
                logits, generated, temperature, top_k, top_p, repetition_penalty, generator
            )
            yield next_token
            if i == max_new_tokens - 1:
                return
            nxt = torch.tensor([[next_token]], device=self.device)
            out = self.model.forward_lm(nxt, past_kvs=past, use_cache=True)
            past = out["past_kvs"]
            logits = out["logits"][:, -1, :]

    def _speculative_tokens(self, past, logits, input_ids, max_new_tokens,
                            temperature, top_k, top_p, generator):
        """Speculative decoding against the attached draft model.

        The draft proposes ``speculate_k`` tokens one at a time, which is cheap
        because it is small; the target then scores all of them in a single
        forward, which costs about what one decode step does because decode is
        bound by reading weights, not by arithmetic. Each proposal is accepted
        with probability ``min(1, p/q)`` and the first rejection is replaced by
        a draw from the normalised residual ``max(p - q, 0)``, so the output
        follows the target's distribution exactly. Under greedy decoding a
        proposal is accepted iff it is the target's argmax, so the output is
        token for token what plain greedy decoding gives.

        The draft is another FramerAI checkpoint sharing this tokenizer, so no
        outside model is involved in what is produced.
        """
        greedy = temperature <= 0
        draft = self.draft
        draft_past, draft_logits = draft._prefill(input_ids)
        # Text only, so both caches hold exactly the prompt's tokens.
        length = int(input_ids.shape[1])
        produced = 0

        def dist(row):
            if greedy:
                return None
            return self._distribution(row, temperature, top_k, top_p)

        while produced < max_new_tokens:
            k = min(self.speculate_k, max_new_tokens - produced)

            proposals, q_rows = [], []
            d_logits, d_past = draft_logits, draft_past
            for j in range(k):
                q = dist(d_logits)
                if greedy:
                    t = d_logits.argmax(dim=-1).item()
                else:
                    t = torch.multinomial(q, num_samples=1, generator=generator).item()
                proposals.append(t)
                q_rows.append(q)
                if j < k - 1:
                    out = draft.model.forward_lm(
                        torch.tensor([[t]], device=draft.device), past_kvs=d_past, use_cache=True
                    )
                    d_past, d_logits = out["past_kvs"], out["logits"][:, -1, :]

            out = self.model.forward_lm(
                torch.tensor([proposals], device=self.device), past_kvs=past, use_cache=True
            )
            past = out["past_kvs"]
            # Row j scores the position proposal j was drafted for.
            rows = [logits] + [out["logits"][:, j, :] for j in range(k)]

            accepted, extra = 0, None
            for j, t in enumerate(proposals):
                row = rows[j]
                if greedy:
                    target = row.argmax(dim=-1).item()
                    if target == t:
                        accepted += 1
                        continue
                    extra = target
                    break
                p = dist(row)
                ratio = (p[0, t] / q_rows[j][0, t].clamp(min=1e-20)).item()
                u = torch.rand((), generator=generator, device=p.device).item() if generator is not None                     else torch.rand(()).item()
                if u < min(1.0, ratio):
                    accepted += 1
                    continue
                residual = (p - q_rows[j]).clamp(min=0)
                total = residual.sum()
                residual = residual / total if total > 0 else p
                extra = torch.multinomial(residual, num_samples=1, generator=generator).item()
                break

            self.last_draft_proposed += k
            self.last_draft_accepted += accepted

            for t in proposals[:accepted]:
                yield t
                produced += 1
            if produced >= max_new_tokens:
                return

            if extra is None:
                # Every proposal stood: the target's next row is free.
                bonus = rows[k]
                extra = bonus.argmax(dim=-1).item() if greedy else torch.multinomial(
                    dist(bonus), num_samples=1, generator=generator
                ).item()

            # Both caches go back to the accepted run, then take the extra token.
            length += accepted
            past = truncate_cache(past, length)
            draft_cache_len = length
            if accepted == k:
                # The draft never ran its last proposal; run it so its cache
                # covers every accepted token.
                out = draft.model.forward_lm(
                    torch.tensor([[proposals[-1]]], device=draft.device),
                    past_kvs=d_past, use_cache=True,
                )
                d_past = out["past_kvs"]
            draft_past = truncate_cache(d_past, draft_cache_len)

            yield extra
            produced += 1
            if produced >= max_new_tokens:
                return

            step = torch.tensor([[extra]], device=self.device)
            out = self.model.forward_lm(step, past_kvs=past, use_cache=True)
            past, logits = out["past_kvs"], out["logits"][:, -1, :]
            out = draft.model.forward_lm(
                torch.tensor([[extra]], device=draft.device), past_kvs=draft_past, use_cache=True
            )
            draft_past, draft_logits = out["past_kvs"], out["logits"][:, -1, :]
            length += 1

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | str | None = None,
        seed: int | None = None,
        image: torch.Tensor = None,
        audio: torch.Tensor = None,
        prefill_chunk_size: int = None,
        return_reason: bool = False,
        allowed_special=None,
        vision_embeds: torch.Tensor = None,
    ) -> str | tuple[str, str]:
        """Generate text, optionally conditioned on an image or audio.

        ``vision_embeds`` are already-encoded visual tokens, used in place of
        encoding ``image``; it is how several video frames reach the model as
        one sequence.

        Uses an incremental KV cache: the prompt is encoded once, then each new
        token attends to the cache in O(1) rather than re-running the whole
        prefix every step. The prompt itself is prefilled in chunks of
        ``prefill_chunk_size`` tokens (default :data:`DEFAULT_PREFILL_CHUNK`),
        which is what makes a context of a million tokens reachable.

        ``prompt`` may also arrive as token ids. That is how a caller that
        mixes its own control markers with text it did not write - the chat
        template above all - keeps the two apart: it encodes each piece with
        its own rules and passes the result straight through.

        ``allowed_special`` applies only to a string prompt, and defaults to
        none, so text that merely contains ``<assistant>`` encodes as those
        characters. Naming markers here suits a prompt this codebase wrote
        end to end, such as the one :meth:`transcribe` sends; anything
        carrying user text should be encoded and passed as ids instead.
        """
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if top_p < 0.0 or top_p > 1.0:
            raise ValueError(f"top_p must be in [0.0, 1.0], got {top_p}")
        image_embeds = audio_embeds = None
        if vision_embeds is not None:
            image_embeds = vision_embeds.to(self.device)
        elif image is not None:
            image_embeds = self.model.forward_vision(image.unsqueeze(0).to(self.device))
        if audio is not None:
            audio_embeds = self.model.forward_audio(audio.unsqueeze(0).to(self.device))

        present = [e for e in (image_embeds, audio_embeds) if e is not None]
        prefix_embeds, modality_embeds = None, None

        if not isinstance(prompt, str):
            # Pre-encoded: the caller already decided which markers are control
            # and which are text, and there is no string left to second-guess
            # that on. Modalities go in front, as they do for any prompt that
            # names no placement.
            ids = list(prompt)
            budget = self._window_budget(max_new_tokens, present)
            if budget is not None:
                ids, max_new_tokens = self._fit_ids(ids, max_new_tokens, *budget)
            tokens = [self.tokenizer.sos_id] + ids + [self.tokenizer.eos_id]
            prefix_embeds = torch.cat(present, dim=1) if present else None
        else:
            prompt, max_new_tokens = self._fit_to_window(
                prompt, max_new_tokens, present, allowed_special=allowed_special
            )
            if self._interleaving() and present:
                tokens, modality_embeds = self._interleaved_prompt(
                    prompt, image_embeds, audio_embeds, allowed_special=allowed_special
                )
            else:
                tokens = self.tokenizer.encode(
                    prompt, add_special=True, allowed_special=allowed_special
                )
                prefix_embeds = torch.cat(present, dim=1) if present else None

        input_ids = torch.tensor([tokens], device=self.device)

        # Prefill the prompt (+ modality prefix) and seed the cache.
        past, logits = self._prefill(
            input_ids, prefix_embeds, prefill_chunk_size, modality_embeds=modality_embeds
        )

        stop_list = [stop] if isinstance(stop, str) else list(stop or [])
        stop_list = [s for s in stop_list if s]

        generator = _make_seed_generator(self.device, seed)

        generated = list(tokens)
        new_tokens = []
        finish_reason = "length"

        for next_token in self._decode_tokens(
            past, logits, input_ids, generated, max_new_tokens,
            temperature, top_k, top_p, repetition_penalty, generator,
            text_only=prefix_embeds is None and not modality_embeds,
        ):

            if next_token == self.tokenizer.eos_id:
                finish_reason = "eos"
                generated.append(next_token)
                break

            generated.append(next_token)
            new_tokens.append(next_token)

            if stop_list:
                gen_text = self.tokenizer.decode(new_tokens)
                earliest_stop_idx = -1
                for s in stop_list:
                    idx = gen_text.find(s)
                    if idx != -1:
                        if earliest_stop_idx == -1 or idx < earliest_stop_idx:
                            earliest_stop_idx = idx
                if earliest_stop_idx != -1:
                    finish_reason = "stop"
                    break

        self.last_finish_reason = finish_reason
        self.last_output_tokens = len(new_tokens)

        if finish_reason == "stop":
            gen_text = self.tokenizer.decode(new_tokens)[:earliest_stop_idx]
            prompt_text = self.tokenizer.decode(tokens)
            final_text = prompt_text + gen_text
            self.last_completion = gen_text
        else:
            final_text = self.tokenizer.decode(generated)
            self.last_completion = self.tokenizer.decode(new_tokens)

        if return_reason:
            return final_text, finish_reason
        return final_text

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | str | None = None,
        seed: int | None = None,
        image: torch.Tensor = None,
        audio: torch.Tensor = None,
        prefill_chunk_size: int = None,
        return_reason: bool = False,
        allowed_special=None,
    ):
        """Yield text deltas incrementally as generation progresses."""
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if top_p < 0.0 or top_p > 1.0:
            raise ValueError(f"top_p must be in [0.0, 1.0], got {top_p}")
        image_embeds = audio_embeds = None
        if image is not None:
            image_embeds = self.model.forward_vision(image.unsqueeze(0).to(self.device))
        if audio is not None:
            audio_embeds = self.model.forward_audio(audio.unsqueeze(0).to(self.device))

        present = [e for e in (image_embeds, audio_embeds) if e is not None]
        prefix_embeds, modality_embeds = None, None

        if not isinstance(prompt, str):
            ids = list(prompt)
            budget = self._window_budget(max_new_tokens, present)
            if budget is not None:
                ids, max_new_tokens = self._fit_ids(ids, max_new_tokens, *budget)
            tokens = [self.tokenizer.sos_id] + ids + [self.tokenizer.eos_id]
            prefix_embeds = torch.cat(present, dim=1) if present else None
        else:
            prompt, max_new_tokens = self._fit_to_window(
                prompt, max_new_tokens, present, allowed_special=allowed_special
            )
            if self._interleaving() and present:
                tokens, modality_embeds = self._interleaved_prompt(
                    prompt, image_embeds, audio_embeds, allowed_special=allowed_special
                )
            else:
                tokens = self.tokenizer.encode(
                    prompt, add_special=True, allowed_special=allowed_special
                )
                prefix_embeds = torch.cat(present, dim=1) if present else None

        input_ids = torch.tensor([tokens], device=self.device)

        # Prefill the prompt (+ modality prefix) and seed the cache.
        past, logits = self._prefill(
            input_ids, prefix_embeds, prefill_chunk_size, modality_embeds=modality_embeds
        )

        stop_list = [stop] if isinstance(stop, str) else list(stop or [])
        stop_list = [s for s in stop_list if s]

        generator = _make_seed_generator(self.device, seed)

        generated = list(tokens)
        new_tokens = []
        emitted_len = 0
        pending_buffer = ""
        finish_reason = "length"

        for next_token in self._decode_tokens(
            past, logits, input_ids, generated, max_new_tokens,
            temperature, top_k, top_p, repetition_penalty, generator,
            text_only=prefix_embeds is None and not modality_embeds,
        ):

            if next_token == self.tokenizer.eos_id:
                finish_reason = "eos"
                generated.append(next_token)
                break

            generated.append(next_token)
            new_tokens.append(next_token)

            decoded_full = self.tokenizer.decode(new_tokens)
            if not decoded_full.endswith("\ufffd"):
                unemitted = decoded_full[emitted_len:]
                if unemitted:
                    pending_buffer += unemitted
                    emitted_len = len(decoded_full)

                if stop_list:
                    earliest_idx = -1
                    for s in stop_list:
                        idx = pending_buffer.find(s)
                        if idx != -1:
                            if earliest_idx == -1 or idx < earliest_idx:
                                earliest_idx = idx

                    if earliest_idx != -1:
                        finish_reason = "stop"
                        to_emit = pending_buffer[:earliest_idx]
                        pending_buffer = ""
                        if return_reason:
                            yield (to_emit, finish_reason)
                        elif to_emit:
                            yield to_emit
                        break

                    holdback_len = 0
                    for s in stop_list:
                        for p_len in range(min(len(s) - 1, len(pending_buffer)), 0, -1):
                            if pending_buffer.endswith(s[:p_len]):
                                holdback_len = max(holdback_len, p_len)
                                break

                    safe_len = len(pending_buffer) - holdback_len
                    if safe_len > 0:
                        delta = pending_buffer[:safe_len]
                        pending_buffer = pending_buffer[safe_len:]
                        yield (delta, None) if return_reason else delta
                else:
                    if pending_buffer:
                        delta = pending_buffer
                        pending_buffer = ""
                        yield (delta, None) if return_reason else delta

        if finish_reason != "stop":
            decoded_full = self.tokenizer.decode(new_tokens)
            unemitted = decoded_full[emitted_len:]
            if unemitted:
                pending_buffer += unemitted
                emitted_len = len(decoded_full)

            if return_reason:
                delta = pending_buffer
                pending_buffer = ""
                yield (delta, finish_reason)
            elif pending_buffer:
                delta = pending_buffer
                pending_buffer = ""
                yield delta

        self.last_finish_reason = finish_reason
        self.last_output_tokens = len(new_tokens)

    @torch.no_grad()
    def generate_chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | str | None = None,
        seed: int | None = None,
        image: torch.Tensor = None,
        audio: torch.Tensor = None,
        prefill_chunk_size: int = None,
        return_reason: bool = False,
    ) -> str | tuple[str, str]:
        """Generate an assistant continuation for structured conversation messages using ChatTemplate."""
        from .tokenizer.chat_template import ChatTemplate

        template = ChatTemplate(version="v1")
        prompt = template.format_messages(messages, add_generation_prompt=True)
        raw_output = self.generate_text(
            template.encode_prompt(messages, self.tokenizer, add_generation_prompt=True),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop,
            seed=seed,
            image=image,
            audio=audio,
            prefill_chunk_size=prefill_chunk_size,
            return_reason=return_reason,
        )
        if return_reason:
            if isinstance(raw_output, tuple) and len(raw_output) == 2:
                raw_text, reason = raw_output
            else:
                raw_text, reason = raw_output, getattr(self, "last_finish_reason", "eos") or "eos"
            if raw_text.startswith(prompt):
                raw_text = raw_text[len(prompt):]
            return raw_text, reason
        if raw_output.startswith(prompt):
            return raw_output[len(prompt):]
        return raw_output

    @torch.no_grad()
    def generate_chat_stream(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | str | None = None,
        seed: int | None = None,
        image: torch.Tensor = None,
        audio: torch.Tensor = None,
        prefill_chunk_size: int = None,
        return_reason: bool = False,
    ):
        """Stream an assistant continuation for structured conversation messages using ChatTemplate."""
        from .tokenizer.chat_template import ChatTemplate

        template = ChatTemplate(version="v1")
        yield from self.generate_stream(
            template.encode_prompt(messages, self.tokenizer, add_generation_prompt=True),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop,
            seed=seed,
            image=image,
            audio=audio,
            prefill_chunk_size=prefill_chunk_size,
            return_reason=return_reason,
        )

    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        num_images: int = 1,
        width: int = None,
        height: int = None,
        aspect: str = None,
        tier: int = None,
        seed: int = None,
        resolution: int = None,
        return_request: bool = False,
        init_image=None,
        mask=None,
        strength: float = 0.6,
        shift: float = None,
        guidance_interval: tuple = None,
    ):
        """Generate images from a text prompt at a requested size.

        ``init_image`` (a PIL image or path) edits instead of drawing from
        nothing: redrawn to the degree ``strength`` says, or, with ``mask`` (white
        where to generate), repainted inside the mask only. The output takes the
        init image's size unless one is asked for. ``shift`` and
        ``guidance_interval`` tune the sampler; see ``ODESampler.sample``.

        Size comes from explicit ``width``/``height``, then an ``aspect`` at a
        size ``tier``, then whatever the prompt itself asks for ("make it 16:9",
        "1024x1024", "portrait"), then the configured default. ``resolution`` is
        the old square-only parameter, still accepted.

        Set ``return_request`` to also receive the resolved
        :class:`~model.utils.image_request.ImageRequest`, which records what was
        understood and where it came from.
        """
        if resolution is not None and width is None and height is None:
            width = height = int(resolution)
        source = _open_image(init_image) if init_image is not None else None
        if source is not None and width is None and height is None:
            width, height = _snap_size(source.size, self.model.config)

        request = resolve_image_request(
            prompt,
            width=width, height=height, aspect=aspect, tier=tier,
            num_images=num_images, seed=seed, config=self.model.config,
        )
        self._require_size_support(request)

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(request.seed))

        context = self._text_context(request.prompt).expand(request.num_images, -1, -1)
        edit = {}
        if source is not None:
            edit["init_images"] = _image_tensor(source, request.width, request.height).expand(
                request.num_images, -1, -1, -1
            ).to(self.device)
            edit["strength"] = float(strength)
            if mask is not None:
                edit["mask"] = _mask_tensor(_open_image(mask), request.width, request.height).expand(
                    request.num_images, -1, -1, -1
                ).to(self.device)
        if shift is not None:
            edit["shift"] = float(shift)
        if guidance_interval is not None:
            edit["cfg_interval"] = tuple(float(v) for v in guidance_interval)
        if edit and self.model.config.image_gen_arch != "latent_dit":
            raise ValueError(
                "editing, shift and guidance_interval need image_gen_arch='latent_dit'"
            )
        images = self._sample_images(request, context, generator, **edit)

        results = []
        for i in range(request.num_images):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            results.append(Image.fromarray(img))
        return (results, request) if return_request else results

    def _require_size_support(self, request):
        """The pixel U-Net only ever supported square output at small sizes."""
        if self.model.config.image_gen_arch == "latent_dit":
            return
        if request.width != request.height:
            raise ValueError(
                f"image_gen_arch='unet' generates square images only; "
                f"{request.width}x{request.height} was requested. "
                "Set image_gen_arch='latent_dit' for arbitrary aspect ratios."
            )

    def _sample_images(self, request, context, generator, **edit):
        """Sample, passing seed and guidance through when the decoder takes them."""
        sampler = self.model.diffusion.sample
        if edit:
            return sampler(request.shape, context, self.device, generator=generator, **edit)
        try:
            return sampler(request.shape, context, self.device, generator=generator)
        except TypeError:
            # The pixel U-Net's sampler has no seed parameter.
            return sampler(request.shape, context, self.device)

    @torch.no_grad()
    def describe_video(self, frames: torch.Tensor, prompt: str = "Describe this video:",
                       max_new_tokens: int = 256, **sampling) -> str:
        """Answer ``prompt`` about ``frames`` ``(T, 3, H, W)``, read in order.

        Each frame is encoded by the vision tower and the frames are placed one
        after another ahead of the prompt, so the model sees what changes as
        well as what is there. Returns what generate_text returns; the answer
        alone is ``last_completion``.
        """
        embeds = [self.model.forward_vision(f.unsqueeze(0).to(self.device)) for f in frames]
        return self.generate_text(
            prompt, max_new_tokens=max_new_tokens,
            vision_embeds=torch.cat(embeds, dim=1), **sampling,
        )

    @torch.no_grad()
    def generate_video(
        self,
        prompt: str,
        num_frames: int = None,
        width: int = None,
        height: int = None,
        aspect: str = None,
        tier: int = None,
        fps: int = None,
        seed: int = None,
        return_request: bool = False,
        first_frame=None,
    ):
        """Generate video frames from a text prompt at a requested size.

        ``first_frame`` (a PIL image or path) animates that image: the clip
        opens on it and continues from it. The clip takes its size unless one
        is asked for.

        Size resolves exactly as it does for images, reusing the same buckets and
        prompt parsing; ``num_frames`` and ``fps`` control duration separately.
        """
        config = self.model.config
        source = _open_image(first_frame) if first_frame is not None else None
        if source is not None and width is None and height is None:
            width, height = _snap_size(source.size, config)
        request = resolve_image_request(
            prompt, width=width, height=height, aspect=aspect, tier=tier, seed=seed, config=config
        )
        frames_requested = int(num_frames or config.video_frames)
        self._require_video_size_support(request, frames_requested)

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(request.seed))

        context = self._text_context(request.prompt)
        if source is not None:
            if config.video_gen_arch != "spacetime_dit" or frames_requested > config.video_frames:
                raise ValueError(
                    "animating an image needs video_gen_arch='spacetime_dit' and a clip that "
                    f"fits one window ({config.video_frames} frames)"
                )
            video = self.model.video_gen.sample(
                1, context, self.device, frames=frames_requested, height=request.height,
                width=request.width, fps=fps, generator=generator,
                first_frame=_image_tensor(source, request.width, request.height).to(self.device),
            )
        else:
            video = self._sample_video(request, frames_requested, fps, context, generator)

        frames = []
        video_np = video[0].cpu().permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        for i in range(video_np.shape[0]):
            frame = ((video_np[i] + 1) * 127.5).clip(0, 255).astype(np.uint8)
            frames.append(Image.fromarray(frame))
        return (frames, request) if return_request else frames

    def _require_video_size_support(self, request, frames):
        """The 3D U-Net was built for one fixed square resolution."""
        config = self.model.config
        if config.video_gen_arch == "spacetime_dit":
            return
        if request.width != request.height or request.width != config.video_resolution:
            raise ValueError(
                f"video_gen_arch='unet3d' generates {config.video_resolution}x"
                f"{config.video_resolution} only; {request.width}x{request.height} was "
                "requested. Set video_gen_arch='spacetime_dit' for arbitrary sizes."
            )
        if frames != config.video_frames:
            raise ValueError(
                f"video_gen_arch='unet3d' generates {config.video_frames} frames only. "
                "Set video_gen_arch='spacetime_dit' for variable duration."
            )

    def _sample_video(self, request, frames, fps, context, generator):
        """Sample, passing duration and size through when the decoder takes them.

        A request longer than the decoder's window goes through the overlapped
        path, so duration is bounded by memory over time rather than by what
        fits in one denoising window. Anything that fits keeps the single-window
        path exactly as it was.
        """
        generator_module = self.model.video_gen
        window = getattr(self.model.config, "video_frames", frames)

        if frames > window and hasattr(generator_module, "sample_long"):
            return generator_module.sample_long(
                1, context, self.device,
                frames=frames, height=request.height, width=request.width,
                fps=fps, generator=generator, window_frames=window,
            )

        try:
            return generator_module.sample(
                1, context, self.device,
                frames=frames, height=request.height, width=request.width,
                fps=fps, generator=generator,
            )
        except TypeError:
            # The 3D U-Net's sampler takes only (batch_size, context, device).
            return generator_module.sample(1, context, self.device)

    @torch.no_grad()
    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate code with lower temperature for more deterministic output."""
        code_prompt = f"<code>Write {language} code: {prompt}\n```{language}\n"
        return self.generate_text(
            code_prompt,
            allowed_special={"<code>"},
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.95,
        )

    @torch.no_grad()
    def generate_audio(
        self,
        prompt: str,
        speaker_audio: torch.Tensor = None,
        reference_text: str = None,
        seed: int = None,
        temperature: float = None,
        top_k: int = None,
        prompt_seconds: float = 3.0,
    ) -> tuple:
        """Generate an audio waveform from a text prompt.

        ``speaker_audio`` is a mono reference clip at the codec's sample rate;
        its speaker embedding conditions the voice. With ``reference_text``,
        what the clip says, the clip's own last ``prompt_seconds`` of acoustic
        tokens are continued as well, with the transcript placed ahead of the
        prompt so the text the tokens follow lines up with them.

        Returns a tuple of (waveform numpy array, sample_rate).
        """
        audio_gen = self.model.audio_gen
        clones = speaker_audio is not None
        if clones and not getattr(audio_gen, "can_clone", False):
            raise ValueError(
                "voice cloning needs audio_gen_arch='rvq_lm' with use_speaker_conditioning=True; "
                "this model has neither"
            )

        kwargs = {}
        if clones:
            ref = speaker_audio.reshape(1, -1).to(self.device)
            kwargs["speaker"] = audio_gen.speaker_embedding(ref)
            if reference_text:
                frames_per_second = audio_gen.sample_rate / self.model.config.codec_hop
                # At most half the positions, so the prompt never crowds out
                # the speech it is there to shape.
                frames = min(int(prompt_seconds * frames_per_second), audio_gen.max_positions // 2)
                kwargs["prompt_codes"] = audio_gen.voice_prompt(ref, max(1, frames))
                prompt = f"{reference_text.strip()} {prompt}"
        if hasattr(audio_gen, "can_clone"):
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_k is not None:
                kwargs["top_k"] = top_k
            if seed is not None:
                kwargs["generator"] = _make_seed_generator(self.device, seed)

        context = self._text_context(prompt)
        if hasattr(audio_gen, "can_clone"):
            waveform = audio_gen.sample(context, self.device, **kwargs)
            return waveform.reshape(-1).cpu().numpy(), audio_gen.sample_rate
        mel = audio_gen.sample(context, self.device)
        waveform = audio_gen.mel_to_waveform(mel)
        return waveform.cpu().numpy(), audio_gen.sample_rate

    @torch.no_grad()
    def transcribe(
        self,
        audio: torch.Tensor,
        prompt: str = "<audio><audio_end>Transcribe the audio:",
        max_new_tokens: int = 256,
    ) -> str:
        """Transcribe or describe audio (audio understanding -> text).

        A recording longer than the encoder's window is cut at pauses into
        windows that fit, each is transcribed, and the pieces are joined, so
        nothing past the first window is dropped. Like generate_text, the
        prompt leads the returned text and ``last_completion`` holds the
        transcript alone.
        """
        from .audio_segment import split_on_silence

        config = self.model.config
        limit = int(getattr(config, "audio_max_frames", 0) or 0) * int(
            getattr(config, "audio_hop_length", 0) or 0
        )
        pieces = split_on_silence(audio, limit) if limit and audio.dim() == 1 else [audio]

        texts = []
        for piece in pieces:
            self.generate_text(
                prompt,
                allowed_special={"<audio>", "<audio_end>"},
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_k=40,
                top_p=0.95,
                audio=piece,
            )
            texts.append(self.last_completion.strip())
        transcript = " ".join(t for t in texts if t)
        self.last_completion = transcript
        return prompt + transcript


def _open_image(source):
    """A PIL image from a PIL image or a path, in RGB or L as given."""
    if isinstance(source, Image.Image):
        return source
    return Image.open(source)


def _snap_size(size, config):
    """An image's own size, rounded to what the generators can produce."""
    multiple = int(getattr(config, "vae_downsample", 8) or 8) * int(getattr(config, "dit_patch_size", 2) or 2)
    width, height = size
    snap = lambda v: max(multiple, int(round(v / multiple)) * multiple)  # noqa: E731
    return snap(width), snap(height)


def _image_tensor(image, width, height) -> torch.Tensor:
    """``(1, 3, H, W)`` in ``[-1, 1]``."""
    rgb = image.convert("RGB").resize((width, height), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0)
    return arr.permute(2, 0, 1).unsqueeze(0)


def _mask_tensor(image, width, height) -> torch.Tensor:
    """``(1, 1, H, W)`` in ``[0, 1]``, 1 where content is to be generated."""
    gray = image.convert("L").resize((width, height), Image.NEAREST)
    return torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0)[None, None]
