"""The mind: one tick loop tying perception, memory, curiosity, affect, and sleep together.

Every experience runs the same cycle.

1. **Encode** it into the shared vector space.
2. **Appraise** it: how unfamiliar, how surprising, where is error falling.
3. **Recall** what it resembles - episodes and concepts - *before* storing it, so
   the cue retrieves the past rather than itself.
4. **Feel** it: the appraisal moves the affective homeostat.
5. **Act**, with decoder settings the current state bent.
6. **Record** it, including the mind's own output, which is how it accumulates a
   history of what it did and not only what happened to it.
7. **Sleep** when fatigue passes threshold: replay, consolidate, forget.

What this is not: a claim that FramerAI is conscious. There is no evidence any of
this produces experience, and this module makes no such assertion. What it does
provide are the functional analogues - persistent autobiographical memory,
intrinsic motivation, an affective state that changes behaviour, a self-model
that can be wrong, and offline consolidation - each of them observable in the
trace and testable in isolation.
"""

import threading
from dataclasses import asdict, dataclass, field

import torch

from .affect import AffectState
from .config import CognitionConfig
from .consolidation import Consolidator
from .curiosity import CuriosityEngine
from .encoder import ExperienceEncoder
from .language import (
    NON_SPACING_SCRIPTS,
    Language,
    detect_language,
    detect_script,
    is_function_word,
    language_name,
)
from .memory import EpisodicMemory, Experience, SemanticMemory, WorkingMemory
from .self_model import SelfModel

# Words that never make a good topic label.
_STOPWORDS = frozenset("""
about above after again against all also am an and any are aren't as at be because been
before being below between both but by can cannot could did do does doing don't down during
each few for from further had has have having he her here hers him his how i if in into is
it its itself just me more most my no nor not now of off on once only or other ought our out
over own same she should so some such than that the their them then there these they this
those through to too under until up very was we were what when where which while who whom
why will with would you your yours
""".split())


@dataclass
class MindTrace:
    """Everything the mind did on one tick, in a form you can print or assert on."""

    tick: int
    kind: str
    topic: str
    text: str
    modality: str = "text"
    language: dict = field(default_factory=dict)
    novelty: float = 0.0
    surprise: float = 0.0
    learning_progress: float = 0.0
    drive: float = 0.0
    affect: dict = field(default_factory=dict)
    feeling: str = ""
    recalled: list = field(default_factory=list)
    concepts: list = field(default_factory=list)
    sampling: dict = field(default_factory=dict)
    response: str | None = None
    # Which language the reply came out in, so a mismatch with `language` is
    # visible rather than something the user has to notice.
    answered_in: dict = field(default_factory=dict)
    slept: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Mind:
    """A persistent cognitive layer around a FramerAI model.

    Usable three ways: attached to a :class:`~model.generate.FramerGenerator`
    (full loop, real generation), attached to a bare model and tokenizer
    (grounded encoding, no generation), or standalone (hashing encoder), which is
    how the tests exercise it without a checkpoint.
    """

    def __init__(
        self,
        config: CognitionConfig | None = None,
        generator=None,
        model=None,
        tokenizer=None,
        device: str = "cpu",
    ):
        self.config = (config or CognitionConfig()).validate()
        self.generator = generator
        if generator is not None:
            model = model or getattr(generator, "model", None)
            tokenizer = tokenizer or getattr(generator, "tokenizer", None)
            device = getattr(generator, "device", device)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.encoder = ExperienceEncoder(
            self.config.d_embed, model=model, tokenizer=tokenizer,
            device=device, seed=self.config.seed,
        )
        self.episodic = EpisodicMemory(self.config)
        self.semantic = SemanticMemory(self.config)
        self.working = WorkingMemory(self.config)
        self.curiosity = CuriosityEngine(self.config)
        self.consolidator = Consolidator(self.config)
        self.affect = AffectState()
        self.self_model = SelfModel(narrative_capacity=self.config.narrative_capacity)

        self.tick = 0
        self.last_trace: MindTrace | None = None
        self._defer_sleep = False
        # A live session perceives on a background thread while the foreground
        # may be mid-conversation. Re-entrant because converse() calls perceive().
        self.lock = threading.RLock()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_generator(cls, generator, config: CognitionConfig | None = None) -> "Mind":
        return cls(config=config, generator=generator)

    # ------------------------------------------------------------------
    # perception
    # ------------------------------------------------------------------

    @staticmethod
    def infer_topic(text: str) -> str:
        """Cheap, deterministic topic label: the longest content word present.

        Crude on purpose. Topics only have to be *stable* for learning progress
        to be measured per area; they do not have to be a taxonomy. Two details
        keep it working outside English: function words from every known language
        profile are skipped, not just English ones, and scripts that do not put
        spaces between words (Han, kana, Thai, Khmer...) are labelled by a short
        character n-gram instead of by "the whole sentence, which is one word".
        """
        script, _, _ = detect_script(text)
        if script in NON_SPACING_SCRIPTS:
            characters = [c for c in text if c.isalpha()]
            return "".join(characters[:3]) or "general"

        words = [w.strip(".,!?;:'\"()[]«»¿¡").lower() for w in text.split()]
        candidates = [
            w for w in words
            if len(w) >= 4 and w.isalpha() and w not in _STOPWORDS and not is_function_word(w)
        ]
        if not candidates:
            return "general"
        return max(candidates, key=lambda w: (len(w), w))

    def perceive(
        self,
        text: str,
        kind: str = "perception",
        topic: str | None = None,
        reward: float = 0.0,
        embedding: torch.Tensor | None = None,
        modality: str = "text",
        language: Language | None = None,
    ) -> MindTrace:
        """Run one full cognitive tick over an experience."""
        with self.lock:
            return self._perceive(text, kind, topic, reward, embedding, modality, language)

    def _perceive(
        self,
        text: str,
        kind: str,
        topic: str | None,
        reward: float,
        embedding: torch.Tensor | None,
        modality: str,
        language: Language | None,
    ) -> MindTrace:
        cfg = self.config
        self.tick += 1
        topic = topic or self.infer_topic(text)
        if embedding is None:
            embedding = self.encoder.encode_text(text)
        language = language or detect_language(text)

        appraisal = self.curiosity.appraise(embedding, topic)
        appraisal.reward = reward
        appraisal.cost = cfg.fatigue_per_tick
        drive = self.curiosity.drive(topic, appraisal.novelty)

        # File the same event under language and sense as well as subject, so
        # "I am worse at Tamil than at English" and "I am worse at what I hear
        # than at what I read" are things the mind can notice about itself.
        self.self_model.note_language(language.code, identified=language.identified)
        self.self_model.note_modality(modality)
        if language.identified:
            self.curiosity.progress.record(f"lang:{language.code}", self.curiosity.last_error)
        self.curiosity.progress.record(f"sense:{modality}", self.curiosity.last_error)

        # Recall before storing, so the cue retrieves the past and not itself.
        hits = self.episodic.retrieve(embedding, self.tick)
        concepts = self.semantic.recall(embedding)

        self.affect.update(appraisal, gain=cfg.affect_gain, decay=cfg.affect_decay)

        experience = Experience(
            tick=self.tick,
            text=text,
            kind=kind,
            topic=topic,
            modality=modality,
            language=language.code,
            novelty=round(appraisal.novelty, 4),
            surprise=round(appraisal.surprise, 4),
            reward=reward,
            affect=self.affect.to_dict(),
        )
        self.episodic.write(embedding, experience)
        self.working.push(experience)
        self.working.focus = topic
        self.self_model.observe(
            topic, self.curiosity.progress.competence(topic), drive
        )

        trace = MindTrace(
            tick=self.tick,
            kind=kind,
            topic=topic,
            text=text,
            modality=modality,
            language=language.to_dict(),
            novelty=experience.novelty,
            surprise=experience.surprise,
            learning_progress=round(appraisal.learning_progress, 4),
            drive=round(drive, 4),
            affect=self.affect.to_dict(),
            feeling=self.affect.describe(),
            recalled=[
                {"tick": h.experience.tick, "text": h.experience.text,
                 "score": round(h.score, 4), "topic": h.experience.topic}
                for h in hits
            ],
            concepts=[
                {"label": c.label, "visits": c.visits, "similarity": round(s, 4)}
                for c, s in concepts
            ],
        )

        if not self._defer_sleep and self.consolidator.should_sleep(self.affect):
            trace.slept = self.rest()

        self.last_trace = trace
        return trace

    def perceive_image(
        self,
        image: torch.Tensor,
        caption: str = "",
        describe: bool = False,
        **kwargs,
    ) -> MindTrace:
        """See an image: encode it, optionally describe it, remember both.

        With ``describe`` set and a generator attached, the model captions the
        image first and the caption becomes the episode's text - so what the mind
        remembers of a scene is what it understood of it, in language, and that
        description is what later recall matches against.
        """
        embedding = self.encoder.encode_image(image)
        text = caption
        if describe and self.generator is not None:
            text = self.describe_image(image) or caption
        return self.perceive(
            text or "<image>", embedding=embedding, modality="vision", **kwargs
        )

    def perceive_audio(
        self,
        audio: torch.Tensor,
        caption: str = "",
        transcribe: bool = False,
        **kwargs,
    ) -> MindTrace:
        """Hear audio: encode it, optionally transcribe it, remember both.

        Transcription runs through the same audio tower the encoder uses, so a
        spoken sentence lands in memory as its words - in whatever language it
        was spoken - rather than as an opaque vector.
        """
        embedding = self.encoder.encode_audio(audio)
        text = caption
        if transcribe and self.generator is not None:
            text = self.transcribe_audio(audio) or caption
        return self.perceive(
            text or "<audio>", embedding=embedding, modality="hearing", **kwargs
        )

    def perceive_video(
        self,
        frames,
        caption: str = "",
        describe: bool = False,
        keyframes: int = 4,
        **kwargs,
    ) -> MindTrace:
        """Watch a clip: sample keyframes, pool them into one episode.

        A clip is remembered as an event, not as N unrelated stills: the
        keyframe embeddings are averaged so the episode's vector is the whole
        clip, which is what makes "the time the room went dark" recallable as one
        thing.
        """
        stack = frames if isinstance(frames, torch.Tensor) else torch.stack(list(frames))
        if stack.dim() != 4:
            raise ValueError(f"expected (T, C, H, W) frames, got shape {tuple(stack.shape)}")

        step = max(1, stack.shape[0] // max(1, keyframes))
        sampled = stack[::step][:keyframes]
        embedding = torch.stack([self.encoder.encode_image(f) for f in sampled]).mean(dim=0)
        embedding = embedding / embedding.norm().clamp_min(1e-8)

        text = caption
        if describe and self.generator is not None:
            text = self.describe_image(sampled[len(sampled) // 2]) or caption
        return self.perceive(
            text or f"<video {stack.shape[0]} frames>",
            embedding=embedding, modality="vision", **kwargs
        )

    # ------------------------------------------------------------------
    # understanding what it just sensed
    # ------------------------------------------------------------------

    def describe_image(self, image: torch.Tensor, prompt: str = "Describe this:") -> str | None:
        """Caption an image with the model, or None with no generator attached."""
        if self.generator is None:
            return None
        decoded = self.generator.generate_text(
            prompt, max_new_tokens=64, image=image if image.dim() == 3 else image[0],
            **self.affect.modulate(0.6, top_p=0.9, top_k=40)
        )
        return self._continuation(prompt, decoded)

    def transcribe_audio(self, audio: torch.Tensor) -> str | None:
        """Transcribe or describe audio with the model, in whatever language it is in."""
        if self.generator is None:
            return None
        prompt = "<audio><audio_end>Transcribe the audio:"
        decoded = self.generator.transcribe(
            audio if audio.dim() == 1 else audio[0], prompt=prompt
        )
        return self._continuation(prompt, decoded).strip()

    # ------------------------------------------------------------------
    # recall and context
    # ------------------------------------------------------------------

    def recall(self, query: str, k: int | None = None, topic: str | None = None) -> list:
        """Retrieve episodes for a query without living through it as an event."""
        embedding = self.encoder.encode_text(query)
        return self.episodic.retrieve(embedding, self.tick, k=k, topic=topic)

    def context(self, trace: MindTrace | None = None, budget: int | None = None) -> str:
        """Build the preamble injected ahead of a prompt.

        Identity, what it recalls, what it has generalised, and how it feels -
        bounded, because context that grows without limit eats the window it is
        supposed to be helping.
        """
        trace = trace or self.last_trace
        budget = budget or self.config.context_char_budget
        blocks = [self.self_model.summary(budget=budget // 3)]

        if trace and trace.recalled:
            lines = [f"- (t{r['tick']}) {r['text'].strip()}" for r in trace.recalled]
            blocks.append("I remember:\n" + "\n".join(lines))
        if trace and trace.concepts:
            named = ", ".join(f"{c['label']} (x{c['visits']})" for c in trace.concepts)
            blocks.append(f"This resembles: {named}.")
        working = self.working.render(budget=budget // 4)
        if working:
            blocks.append("Just now:\n" + working)

        # Answer in the language you were addressed in. Stated only when the
        # identification is confident, because instructing the model to reply in
        # a language that was guessed wrong is worse than not saying anything.
        if trace and trace.language.get("confidence", 0.0) >= 0.5:
            code = trace.language.get("code", "und")
            if not code.startswith("und"):
                name = language_name(code)
                blocks.append(f"This was said in {name}. Answer in {name}.")

        blocks.append(f"I feel {self.affect.describe()}.")

        text = "\n\n".join(b for b in blocks if b)
        if len(text) > budget:
            text = text[:budget].rsplit("\n", 1)[0]
        return text

    # ------------------------------------------------------------------
    # acting
    # ------------------------------------------------------------------

    def converse(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        reward: float = 0.0,
        use_context: bool = True,
    ) -> tuple[str | None, MindTrace]:
        """Perceive a prompt, answer it in the current state, and remember both.

        Returns ``(reply, trace)`` where ``reply`` is the continuation alone -
        not the prompt echo the generator returns - or None when no generator is
        attached, which is the mode where only the cognitive loop runs.

        The mind's own reply is fed back through perception as an ``action``
        episode. That is what makes the history autobiographical rather than a
        log of inputs, and it is what lets novelty fall for things it has already
        said.
        """
        with self.lock:
            return self._converse(
                prompt, max_new_tokens, temperature, top_k, top_p, reward, use_context
            )

    def _converse(
        self, prompt, max_new_tokens, temperature, top_k, top_p, reward, use_context
    ) -> tuple[str | None, MindTrace]:
        self._defer_sleep = True
        try:
            trace = self.perceive(prompt, kind="perception", reward=reward)
            trace.sampling = self.affect.modulate(
                temperature, top_p=top_p, top_k=top_k,
                temperature_span=self.config.temperature_span,
                top_p_span=self.config.top_p_span,
            )

            response = None
            if self.generator is not None:
                preamble = self.context(trace) if use_context else ""
                full_prompt = f"{preamble}\n\n{prompt}" if preamble else prompt
                decoded = self.generator.generate_text(
                    full_prompt, max_new_tokens=max_new_tokens, **trace.sampling
                )
                response = self._continuation(full_prompt, decoded)
                trace.response = response
                if response:
                    self.perceive(response, kind="action", topic=trace.topic)
                    trace.answered_in = detect_language(response).to_dict()

            self.self_model.exchanges += 1
        finally:
            self._defer_sleep = False

        if self.consolidator.should_sleep(self.affect):
            trace.slept = self.rest()

        self.last_trace = trace
        return response, trace

    @staticmethod
    def _continuation(prompt: str, decoded: str) -> str:
        """The reply alone, with the echoed prompt removed.

        ``FramerGenerator.generate_text`` returns prompt plus continuation. Left
        as it is, the mind would remember its own preamble as something it said,
        and recall would fill up with copies of the context it had just built.
        """
        if decoded.startswith(prompt):
            return decoded[len(prompt):].lstrip()
        anchor = prompt[-48:]
        if anchor:
            cut = decoded.rfind(anchor)
            if cut >= 0:
                return decoded[cut + len(anchor):].lstrip()
        return decoded

    def wonder(self, topics: list[str] | None = None) -> str:
        """Ask itself something, unprompted, from wherever curiosity is pulling.

        The question is recorded as an episode and raised as a self-authored
        goal, so an idle mind still accumulates history and intent.
        """
        topic = self.curiosity.frontier(topics)
        question = self.curiosity.question(topic)
        self.perceive(question, kind="question", topic=topic or "general")
        self.self_model.add_goal(
            f"understand {topic}" if topic else "find something worth learning",
            tick=self.tick,
            priority=round(self.affect.curiosity, 4),
            origin="self",
        )
        return question

    def reward(self, value: float, note: str = "") -> MindTrace:
        """External feedback about the last exchange. Positive or negative."""
        text = note or f"feedback: {value:+.2f}"
        topic = self.working.focus or "general"
        return self.perceive(text, kind="feedback", topic=topic, reward=value)

    # ------------------------------------------------------------------
    # rest and introspection
    # ------------------------------------------------------------------

    def rest(self, train_step=None) -> dict:
        """Force a consolidation pass. ``train_step`` may update the backbone."""
        return self.consolidator.sleep(
            tick=self.tick,
            episodic=self.episodic,
            semantic=self.semantic,
            curiosity=self.curiosity,
            self_model=self.self_model,
            affect=self.affect,
            train_step=train_step,
        )

    def competence_by(self, axis: str) -> dict:
        """Competence along a bookkeeping axis: ``"lang"`` or ``"sense"``.

        This is the mind's estimate of itself, not a benchmark - it is derived
        from its own prediction error, so it says where it is struggling, not how
        good it actually is.
        """
        prefix = f"{axis}:"
        return {
            topic[len(prefix):]: round(self.curiosity.progress.competence(topic), 4)
            for topic in self.curiosity.progress.topics()
            if topic.startswith(prefix)
        }

    def introspect(self) -> dict:
        """A readable snapshot of the whole cognitive state."""
        return {
            "tick": self.tick,
            "grounded": self.encoder.grounded,
            "affect": self.affect.to_dict(),
            "feeling": self.affect.describe(),
            "memory": self.episodic.stats(self.tick),
            "concepts": len(self.semantic),
            "working_memory": [e.text for e in self.working.items()],
            "focus": self.working.focus,
            "interests": self.self_model.strongest_interests(5),
            "weakest": self.self_model.weakest_areas(3),
            "languages": self.self_model.known_languages(8),
            "unidentified_languages": self.self_model.unidentified_languages,
            "language_competence": self.competence_by("lang"),
            "senses": dict(self.self_model.modalities),
            "sense_competence": self.competence_by("sense"),
            "goals": [g.to_dict() for g in self.self_model.top_goals(5)],
            "frontier": self.curiosity.frontier(),
            "exchanges": self.self_model.exchanges,
            "sleeps": self.self_model.sleeps,
            "narrative": [n["text"] for n in self.self_model.narrative[-3:]],
        }

    # ------------------------------------------------------------------
    # persistence - continuity across restarts
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "tick": self.tick,
            "episodic": self.episodic.state_dict(),
            "semantic": self.semantic.state_dict(),
            "working": self.working.state_dict(),
            "curiosity": self.curiosity.state_dict(),
            "consolidator": self.consolidator.state_dict(),
            "affect": self.affect.to_dict(),
            "self_model": self.self_model.to_dict(),
        }

    def load_state_dict(self, state: dict) -> "Mind":
        self.tick = int(state.get("tick", 0))
        self.episodic.load_state_dict(state["episodic"])
        self.semantic.load_state_dict(state["semantic"])
        self.working.load_state_dict(state["working"])
        self.curiosity.load_state_dict(state["curiosity"])
        self.consolidator.load_state_dict(state.get("consolidator", {}))
        self.affect = AffectState.from_dict(state["affect"])
        self.self_model = SelfModel.from_dict(state["self_model"])
        return self

    def save(self, path: str) -> str:
        """Persist the whole mind. Without this, every restart is a new mind."""
        torch.save(self.state_dict(), path)
        return path

    @classmethod
    def load(cls, path: str, generator=None, model=None, tokenizer=None, device: str = "cpu"):
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = CognitionConfig.from_dict(state["config"])
        mind = cls(config=config, generator=generator, model=model,
                   tokenizer=tokenizer, device=device)
        return mind.load_state_dict(state)
