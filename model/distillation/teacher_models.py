"""
Local open-source teacher model loader.

Downloads and loads open-source LLM weights from HuggingFace Hub for
knowledge distillation. No API keys, no cloud calls — everything runs locally.

Supported model families:
- Meta Llama 3 / 3.1 / 3.2
- LLaVA (multimodal)
- DeepSeek R1 / V3 / Coder
- Mistral / Mixtral
- Qwen 2 / 2.5
- Phi-3 / Phi-4
- Gemma 2
- Any HuggingFace causal LM
"""

import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("framerai.distillation")

# ---------------------------------------------------------------------------
# Well-known open-source model registry
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    # Meta Llama
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",

    # LLaVA (multimodal)
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
    "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
    "llava-next-8b": "llava-hf/llava-v1.6-mistral-7b-hf",

    # DeepSeek
    "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-r1-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-coder-7b": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "deepseek-coder-33b": "deepseek-ai/deepseek-coder-33b-instruct",

    # Mistral / Mixtral
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",

    # Qwen
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",

    # Phi
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct",
    "phi-3-medium": "microsoft/Phi-3-medium-4k-instruct",
    "phi-4-mini": "microsoft/Phi-4-mini-instruct",

    # Gemma
    "gemma2-2b": "google/gemma-2-2b-it",
    "gemma2-9b": "google/gemma-2-9b-it",
    "gemma2-27b": "google/gemma-2-27b-it",
}

# Recommended subset that fits on a single consumer GPU (4-bit quantized)
# These are loaded one at a time, so only one model's VRAM is needed at once
RECOMMENDED_ALL_MODELS = [
    "llama3-8b",
    "deepseek-r1-7b",
    "deepseek-r1-8b",
    "deepseek-coder-7b",
    "mistral-7b",
    "qwen2.5-7b",
    "qwen2.5-coder-7b",
    "phi-4-mini",
    "gemma2-9b",
]


def resolve_model_list(teacher_model_arg: str) -> list:
    """
    Parse --teacher-model argument into a list of model names.

    Supports:
      - "all"                     -> all recommended models
      - "all-small"               -> models <=3B (fit on any GPU)
      - "all-medium"              -> models <=14B
      - "all-large"               -> models >14B
      - "llama3-8b,mistral-7b"    -> explicit comma-separated list
      - "llama3-8b"               -> single model
    """
    if teacher_model_arg == "all":
        return list(RECOMMENDED_ALL_MODELS)
    elif teacher_model_arg == "all-small":
        small = ["llama3.2-1b", "llama3.2-3b", "deepseek-r1-1.5b", "phi-4-mini", "gemma2-2b"]
        return [m for m in small if m in SUPPORTED_MODELS]
    elif teacher_model_arg == "all-medium":
        medium = [
            "llama3-8b", "llama3.1-8b", "deepseek-r1-7b", "deepseek-r1-8b",
            "deepseek-r1-14b", "deepseek-coder-7b", "mistral-7b",
            "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-coder-7b",
            "phi-3-mini", "phi-3-small", "phi-3-medium", "phi-4-mini",
            "gemma2-2b", "gemma2-9b",
        ]
        return [m for m in medium if m in SUPPORTED_MODELS]
    elif teacher_model_arg == "all-large":
        large = [
            "llama3-70b", "llama3.1-70b", "deepseek-r1-32b", "deepseek-r1-70b",
            "deepseek-coder-33b", "mixtral-8x7b", "qwen2.5-32b", "qwen2.5-72b",
            "gemma2-27b",
        ]
        return [m for m in large if m in SUPPORTED_MODELS]
    elif "," in teacher_model_arg:
        return [m.strip() for m in teacher_model_arg.split(",") if m.strip()]
    else:
        return [teacher_model_arg]


@dataclass
class TeacherConfig:
    """Configuration for loading a teacher model."""
    model_name_or_path: str
    torch_dtype: str = "auto"           # auto, float16, bfloat16, float32
    load_in_4bit: bool = False          # QLoRA-style 4-bit quantization
    load_in_8bit: bool = False          # 8-bit quantization
    device_map: str = "auto"            # auto, cpu, cuda:0, etc.
    trust_remote_code: bool = True
    attn_implementation: str = None     # flash_attention_2, sdpa, eager
    max_memory: dict = None             # {0: "24GiB", "cpu": "64GiB"}
    cache_dir: str = None               # HuggingFace cache directory
    hf_token: str = None                # HF token for gated models (Llama, etc.)


class TeacherModel:
    """
    Wraps a HuggingFace causal LM as a teacher for distillation.

    Handles loading, tokenization, and extracting logits/hidden states
    that FramerAI's student model trains against.
    """

    def __init__(self, model, tokenizer, config: TeacherConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def vocab_size(self):
        return self.model.config.vocab_size

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def name(self):
        return self.config.model_name_or_path

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Get teacher logits for a batch of token IDs."""
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
            output_hidden_states=False,
        )
        return outputs.logits.cpu()

    @torch.no_grad()
    def get_logits_and_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Get teacher logits AND last hidden state for feature distillation."""
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
            output_hidden_states=True,
        )
        return outputs.logits.cpu(), outputs.hidden_states[-1].cpu()

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate text from a prompt (for data generation)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        batch_size: int = 4,
    ) -> list:
        """Generate text for multiple prompts."""
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            for j, output in enumerate(outputs):
                prompt_len = inputs["input_ids"].shape[1]
                new_tokens = output[prompt_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                results.append(text)
        return results

    def tokenize(self, text: str, max_length: int = 2048) -> dict:
        """Tokenize text using the teacher's tokenizer."""
        return self.tokenizer(
            text, return_tensors="pt", max_length=max_length,
            truncation=True, padding="max_length",
        )

    def unload(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    if dtype_str == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_str, "auto")


def load_teacher(config: TeacherConfig) -> TeacherModel:
    """
    Download (if needed) and load an open-source teacher model from HuggingFace.

    Supports quantization (4-bit, 8-bit), flash attention, multi-GPU sharding.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = config.model_name_or_path

    # Resolve shorthand names
    if model_path in SUPPORTED_MODELS:
        model_path = SUPPORTED_MODELS[model_path]
        logger.info(f"Resolved model name to: {model_path}")

    logger.info(f"Loading teacher model: {model_path}")

    # Build kwargs
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "device_map": config.device_map,
        "torch_dtype": _resolve_dtype(config.torch_dtype),
    }

    if config.cache_dir:
        model_kwargs["cache_dir"] = config.cache_dir

    if config.hf_token:
        model_kwargs["token"] = config.hf_token

    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    if config.max_memory:
        model_kwargs["max_memory"] = config.max_memory

    # Quantization
    if config.load_in_4bit or config.load_in_8bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.load_in_4bit else None,
            bnb_4bit_use_double_quant=config.load_in_4bit,
            bnb_4bit_quant_type="nf4" if config.load_in_4bit else None,
        )
        model_kwargs["quantization_config"] = quant_config
        logger.info(f"Quantization: {'4-bit' if config.load_in_4bit else '8-bit'}")

    # Load tokenizer
    tok_kwargs = {"trust_remote_code": config.trust_remote_code}
    if config.cache_dir:
        tok_kwargs["cache_dir"] = config.cache_dir
    if config.hf_token:
        tok_kwargs["token"] = config.hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Teacher loaded: {param_count:,} parameters")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory used: {mem:.1f} GB")

    return TeacherModel(model, tokenizer, config)


def load_teacher_simple(
    model_name: str,
    quantize: str = None,
    device_map: str = "auto",
    dtype: str = "auto",
    hf_token: str = None,
    cache_dir: str = None,
) -> TeacherModel:
    """
    Convenience function to load a teacher with minimal args.

    Args:
        model_name: Model name (shorthand like "llama3-8b" or full HF path)
        quantize: None, "4bit", or "8bit"
        device_map: "auto", "cpu", "cuda:0"
        dtype: "auto", "float16", "bfloat16"
        hf_token: HuggingFace token for gated models
        cache_dir: Where to cache downloaded models
    """
    config = TeacherConfig(
        model_name_or_path=model_name,
        torch_dtype=dtype,
        load_in_4bit=(quantize == "4bit"),
        load_in_8bit=(quantize == "8bit"),
        device_map=device_map,
        hf_token=hf_token or os.environ.get("HF_TOKEN"),
        cache_dir=cache_dir,
    )
    return load_teacher(config)
