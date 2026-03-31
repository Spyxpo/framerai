"""Knowledge distillation pipeline for FramerAI - train using local open-source LLM weights."""

from .teacher_models import TeacherModel, load_teacher, SUPPORTED_MODELS, resolve_model_list
from .data_generator import DistillationDataGenerator
from .distill_trainer import DistillationTrainer, DistillationConfig

__all__ = [
    "TeacherModel",
    "load_teacher",
    "SUPPORTED_MODELS",
    "DistillationDataGenerator",
    "DistillationTrainer",
    "DistillationConfig",
]
