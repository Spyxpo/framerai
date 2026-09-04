"""
Benchmark distilled image generation: wall-clock timing and quality metrics.

Compares a multi-step teacher model against a few-step distilled student on:
- Inference latency (wall-clock time per image)
- Theoretical sampling cost (denoiser forward passes)
- Text-image alignment (contrastive score)
- FID against real images (when --real-images-dir is provided)

Usage:
    # Basic benchmark with built-in captions
    python benchmarks/distillation_image_gen.py \\
        --teacher checkpoints/teacher.pt \\
        --student checkpoints/student.pt \\
        --tokenizer tokenizer/ \\
        --device cuda:0

    # With real images for FID calculation
    python benchmarks/distillation_image_gen.py \\
        --teacher checkpoints/teacher.pt \\
        --student checkpoints/student.pt \\
        --tokenizer tokenizer/ \\
        --real-images-dir data/validation_images/ \\
        --output results.json

    # CPU testing with custom captions
    python benchmarks/distillation_image_gen.py \\
        --teacher checkpoints/teacher.pt \\
        --student checkpoints/student.pt \\
        --tokenizer tokenizer/ \\
        --device cpu \\
        --resolution 64 \\
        --num-images 16 \\
        --captions captions.txt
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from model.configs import FramerConfig
from model.eval.image import alignment_score, fid
from model.generate import FramerGenerator
from model.modules.flow import sampling_cost

# Built-in captions for convenience when --captions is not provided
DEFAULT_CAPTIONS = [
    "a red apple on a wooden table",
    "a sunset over mountains",
    "two dogs playing in a park",
    "a white cat sleeping on a couch",
    "a vintage car on a city street",
    "a lighthouse by the ocean at night",
    "a bowl of fresh fruit",
    "a snowy forest in winter",
    "a colorful hot air balloon in the sky",
    "a modern skyscraper at dusk",
    "a field of sunflowers",
    "a steaming cup of coffee",
    "a wooden bridge over a river",
    "a red rose with water droplets",
    "a sandy beach with palm trees",
    "a small cottage in the countryside",
]


def load_captions(caption_file: str = None, num_images: int = 50) -> list[str]:
    """Load captions, cycling if needed to reach num_images."""
    if caption_file is not None:
        if not os.path.exists(caption_file):
            raise FileNotFoundError(f"Caption file not found: {caption_file}")
        with open(caption_file, encoding="utf-8") as f:
            captions = [line.strip() for line in f if line.strip()]
        if not captions:
            raise ValueError(f"Caption file is empty: {caption_file}")
    else:
        captions = DEFAULT_CAPTIONS

    # Cycle captions to match num_images deterministically
    result = []
    for i in range(num_images):
        result.append(captions[i % len(captions)])
    return result


def load_real_images(real_images_dir: str, resolution: int, device: str) -> torch.Tensor:
    """Load real images from directory as tensors."""
    if not os.path.isdir(real_images_dir):
        raise ValueError(f"Real images directory not found: {real_images_dir}")

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        image_paths.extend(Path(real_images_dir).glob(ext))

    if not image_paths:
        raise ValueError(f"No images found in {real_images_dir}")

    images = []
    for path in sorted(image_paths):  # Deterministic order
        img = Image.open(path).convert("RGB").resize((resolution, resolution))
        tensor = torch.from_numpy(
            __import__("numpy").asarray(img, dtype="float32")
        ).permute(2, 0, 1) / 127.5 - 1.0
        images.append(tensor)

    return torch.stack(images).to(device)


def validate_checkpoint_config(checkpoint_path: str, expected_distilled: bool, name: str):
    """Validate that checkpoint has expected configuration for teacher/student."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{name} checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise ValueError(f"{name} checkpoint missing config")

    config_dict = ckpt["config"]
    if isinstance(config_dict, FramerConfig):
        config = config_dict
    else:
        config = FramerConfig.from_dict(config_dict)

    # Validate architecture
    if config.image_gen_arch != "latent_dit":
        raise ValueError(
            f"{name} must use image_gen_arch='latent_dit', got '{config.image_gen_arch}'"
        )

    # Validate distillation flag
    flow_distilled = getattr(config, "flow_distilled", False)
    if flow_distilled != expected_distilled:
        raise ValueError(
            f"{name} must have flow_distilled={expected_distilled}, got {flow_distilled}"
        )

    return config


def benchmark_inference_time(
    generator: FramerGenerator,
    captions: list[str],
    resolution: int,
    seed: int,
    device: torch.device,
    n_warmup: int = 3,
) -> dict:
    """Benchmark wall-clock inference time."""
    # Warmup
    for i in range(n_warmup):
        _ = generator.generate_image(
            captions[i % len(captions)],
            width=resolution,
            height=resolution,
            seed=seed + i,
        )

    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Timed generation
    start = time.perf_counter()

    images = []
    for i, caption in enumerate(captions):
        imgs = generator.generate_image(
            caption,
            width=resolution,
            height=resolution,
            seed=seed + i,
        )
        images.extend(imgs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    num_images = len(captions)
    latency_ms = (elapsed / num_images) * 1000
    images_per_sec = num_images / elapsed

    return {
        "total_time_sec": elapsed,
        "latency_ms_per_image": latency_ms,
        "images_per_sec": images_per_sec,
        "num_images": num_images,
    }


def benchmark_alignment(
    generator: FramerGenerator,
    captions: list[str],
    resolution: int,
    seed: int,
    device: torch.device,
) -> float:
    """Benchmark text-image alignment score."""
    # Generate images and extract features
    image_tensors = []
    caption_ids = []

    for i, caption in enumerate(captions):
        imgs = generator.generate_image(
            caption,
            width=resolution,
            height=resolution,
            seed=seed + i,
        )
        # Convert PIL to tensor
        import numpy as np
        img_array = np.asarray(imgs[0], dtype="float32")
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1) / 127.5 - 1.0
        image_tensors.append(img_tensor)

        # Encode caption
        tokens = generator.tokenizer.encode(caption, add_special=True)
        caption_ids.append(torch.tensor(tokens, dtype=torch.long))

    # Stack tensors
    images = torch.stack(image_tensors)
    captions_tensor = torch.nn.utils.rnn.pad_sequence(
        caption_ids, batch_first=True, padding_value=0
    )

    # Compute alignment
    return alignment_score(generator.model, images, captions_tensor, device=str(device))


def compute_fid(
    generator: FramerGenerator,
    captions: list[str],
    real_images: torch.Tensor,
    resolution: int,
    seed: int,
    device: torch.device,
) -> float:
    """Compute FID between generated and real images."""
    # Generate images
    fake_tensors = []
    for i, caption in enumerate(captions):
        imgs = generator.generate_image(
            caption,
            width=resolution,
            height=resolution,
            seed=seed + i,
        )
        import numpy as np
        img_array = np.asarray(imgs[0], dtype="float32")
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1) / 127.5 - 1.0
        fake_tensors.append(img_tensor)

    fake_images = torch.stack(fake_tensors)

    # Use only the number of real images we have
    num_samples = min(len(fake_images), len(real_images))
    return fid(
        generator.model,
        real_images[:num_samples],
        fake_images[:num_samples],
        device=str(device),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark distilled image generation: timing and quality"
    )
    parser.add_argument("--teacher", required=True, help="Teacher checkpoint path")
    parser.add_argument("--student", required=True, help="Student checkpoint path")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer directory path")
    parser.add_argument(
        "--device", default="cuda:0", help="Device: cuda:0, cpu, etc. (default: cuda:0)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Image resolution (square) (default: 256)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of images to generate (default: 50)",
    )
    parser.add_argument(
        "--captions",
        default=None,
        help="Text file with one caption per line (optional)",
    )
    parser.add_argument(
        "--real-images-dir",
        default=None,
        help="Directory with real images for FID calculation (optional)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output", default=None, help="JSON output file path (optional)"
    )
    parser.add_argument(
        "--n-warmup", type=int, default=3, help="Warmup iterations (default: 3)"
    )

    args = parser.parse_args()

    # Validate checkpoints
    print("Validating checkpoints...")
    teacher_config = validate_checkpoint_config(args.teacher, False, "Teacher")
    student_config = validate_checkpoint_config(args.student, True, "Student")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load captions
    print(f"Loading {args.num_images} captions...")
    captions = load_captions(args.captions, args.num_images)

    # Load real images if provided
    real_images = None
    if args.real_images_dir:
        print(f"Loading real images from {args.real_images_dir}...")
        real_images = load_real_images(args.real_images_dir, args.resolution, device)
        print(f"Loaded {len(real_images)} real images")

    # Load models
    print("Loading teacher model...")
    teacher_gen = FramerGenerator.from_checkpoint(args.teacher, args.tokenizer, str(device))

    print("Loading student model...")
    student_gen = FramerGenerator.from_checkpoint(args.student, args.tokenizer, str(device))

    # Calculate theoretical sampling costs
    teacher_cost = sampling_cost(
        steps=teacher_config.sampler_steps,
        method=teacher_config.sampler_method,
        guidance=teacher_config.cfg_scale > 1.0,
        guidance_distilled=False,
    )
    student_cost = sampling_cost(
        steps=student_config.flow_distilled_steps,
        method=student_config.sampler_method,
        guidance=False,  # Student has guidance baked in
        guidance_distilled=True,
    )

    print("\n" + "=" * 70)
    print("Distillation Benchmark: Image Generation")
    print("=" * 70)
    print(f"Teacher:      {args.teacher}")
    print(f"Student:      {args.student}")
    print(f"Device:       {device}")
    print(f"Resolution:   {args.resolution}×{args.resolution}")
    print(f"Images:       {args.num_images}")
    print(f"Seed:         {args.seed}")
    print(f"Warmup:       {args.n_warmup} iterations")
    print("=" * 70)

    # Benchmark teacher timing
    print("\n[1/5] Benchmarking teacher inference time...")
    teacher_timing = benchmark_inference_time(
        teacher_gen, captions, args.resolution, args.seed, device, args.n_warmup
    )
    print(f"      Total time:  {teacher_timing['total_time_sec']:.2f} sec")
    print(f"      Latency:     {teacher_timing['latency_ms_per_image']:.1f} ms/image")
    print(f"      Throughput:  {teacher_timing['images_per_sec']:.2f} images/sec")

    # Benchmark student timing
    print("\n[2/5] Benchmarking student inference time...")
    student_timing = benchmark_inference_time(
        student_gen, captions, args.resolution, args.seed, device, args.n_warmup
    )
    print(f"      Total time:  {student_timing['total_time_sec']:.2f} sec")
    print(f"      Latency:     {student_timing['latency_ms_per_image']:.1f} ms/image")
    print(f"      Throughput:  {student_timing['images_per_sec']:.2f} images/sec")

    # Benchmark teacher alignment
    print("\n[3/5] Benchmarking teacher text-image alignment...")
    teacher_alignment = benchmark_alignment(
        teacher_gen, captions, args.resolution, args.seed, device
    )
    print(f"      Alignment:   {teacher_alignment:.4f}")

    # Benchmark student alignment
    print("\n[4/5] Benchmarking student text-image alignment...")
    student_alignment = benchmark_alignment(
        student_gen, captions, args.resolution, args.seed, device
    )
    print(f"      Alignment:   {student_alignment:.4f}")

    # Benchmark FID if real images provided
    fid_teacher = None
    fid_student = None
    if real_images is not None:
        print("\n[5/5] Computing FID against real images...")
        print("      Teacher FID...")
        fid_teacher = compute_fid(
            teacher_gen, captions, real_images, args.resolution, args.seed, device
        )
        print(f"      Teacher FID: {fid_teacher:.2f}")

        print("      Student FID...")
        fid_student = compute_fid(
            student_gen, captions, real_images, args.resolution, args.seed, device
        )
        print(f"      Student FID: {fid_student:.2f}")
    else:
        print("\n[5/5] Skipping FID (no --real-images-dir provided)")

    # Calculate speedup
    speedup = teacher_timing["latency_ms_per_image"] / student_timing["latency_ms_per_image"]
    theoretical_speedup = teacher_cost / student_cost

    # Print results table
    print("\n" + "=" * 70)
    print(f"{'Metric':<35} {'Teacher':<15} {'Student':<15}")
    print("=" * 70)
    print(
        f"{'Sampler steps':<35} {teacher_config.sampler_steps:<15} "
        f"{student_config.flow_distilled_steps:<15}"
    )
    print(
        f"{'Sampler method':<35} {teacher_config.sampler_method:<15} "
        f"{student_config.sampler_method:<15}"
    )
    print(
        f"{'Guidance distilled':<35} {'False':<15} {'True':<15}"
    )
    print(
        f"{'Theoretical cost (forwards)':<35} {teacher_cost:<15} {student_cost:<15}"
    )
    print(
        f"{'Latency (ms/image)':<35} {teacher_timing['latency_ms_per_image']:<15.1f} "
        f"{student_timing['latency_ms_per_image']:<15.1f}"
    )
    print(
        f"{'Throughput (images/sec)':<35} {teacher_timing['images_per_sec']:<15.2f} "
        f"{student_timing['images_per_sec']:<15.2f}"
    )
    print(
        f"{'Text-image alignment':<35} {teacher_alignment:<15.4f} "
        f"{student_alignment:<15.4f}"
    )
    if fid_teacher is not None:
        print(f"{'FID':<35} {fid_teacher:<15.2f} {fid_student:<15.2f}")
    else:
        print(f"{'FID':<35} {'N/A':<15} {'N/A':<15}")
    print("=" * 70)
    print(f"Measured speedup:     {speedup:.2f}x")
    print(f"Theoretical speedup:  {theoretical_speedup:.2f}x")
    print("=" * 70)

    # Save JSON output if requested
    if args.output:
        result = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "teacher_checkpoint": args.teacher,
            "student_checkpoint": args.student,
            "device": str(device),
            "resolution": args.resolution,
            "num_images": args.num_images,
            "seed": args.seed,
            "teacher": {
                "sampler_steps": teacher_config.sampler_steps,
                "sampler_method": teacher_config.sampler_method,
                "cfg_scale": teacher_config.cfg_scale,
                "flow_distilled": False,
                "theoretical_cost": teacher_cost,
                "timing": teacher_timing,
                "alignment": teacher_alignment,
                "fid": fid_teacher,
            },
            "student": {
                "sampler_steps": student_config.flow_distilled_steps,
                "sampler_method": student_config.sampler_method,
                "flow_distilled": True,
                "theoretical_cost": student_cost,
                "timing": student_timing,
                "alignment": student_alignment,
                "fid": fid_student,
            },
            "speedup": {
                "measured": speedup,
                "theoretical": theoretical_speedup,
            },
        }

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
