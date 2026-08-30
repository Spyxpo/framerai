"""Supervised Fine-Tuning (SFT) training loop."""

import logging
from typing import Any

from .trainer import train_language_model

logger = logging.getLogger("framerai")


def train_sft(
    config: Any,
    model: Any,
    dataloader: Any,
    device: Any,
    output_dir: str,
    start_step: int = 0,
    log_interval: int = 10,
    save_interval: int = 500,
    logger_obj: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
) -> int:
    """Train model using Supervised Fine-Tuning (SFT) with masked prompt labels.

    Reuses the core LM training loop while enforcing SFT logging and loss masking.
    """
    log = logger_obj if logger_obj else logger
    log.info("Starting Supervised Fine-Tuning (SFT) training pass...")
    return train_language_model(
        config=config,
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        start_step=start_step,
        log_interval=log_interval,
        save_interval=save_interval,
        logger=log,
        optimizer=optimizer,
        scheduler=scheduler,
    )
