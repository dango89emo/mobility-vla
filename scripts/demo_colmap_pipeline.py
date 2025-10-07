"""Demonstration script that runs MobilityVLA on COLMAP outputs with Qwen3-VL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is available when invoking via `python scripts/demo_colmap_pipeline.py`.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from mobility_vla import (
    Instruction,
    MobilityVLA,
    Observation,
    Qwen3VLGoalSelector,
    load_tour_from_colmap,
)
from mobility_vla.pipeline import MobilityVLAConfig


def _resolve_dtype(name: str) -> torch.dtype | None:
    value = name.strip().lower()
    if value in {"", "auto", "none"}:
        return None
    if not hasattr(torch, value):
        raise ValueError(
            f"Unsupported torch dtype '{name}'. Known dtypes include float32, float16, bfloat16."
        )
    dtype = getattr(torch, value)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Attribute torch.{name} is not a dtype.")
    return dtype


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poses-path",
        type=Path,
        default=Path("data/sceaux_colmap/poses.txt"),
        help="Path to the COLMAP pose export (poses.txt).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/ImageDataset_SceauxCastle-master/images"),
        help="Directory containing the input images referenced by poses.txt.",
    )
    parser.add_argument(
        "--instruction",
        default="Find the viewpoint closest to the final frame of the tour.",
        help="Instruction to feed into the MobilityVLA pipeline.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="How many localization steps to simulate (frames are sampled sequentially).",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Hugging Face model id for Qwen3-VL (downloaded automatically if missing).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map hint passed to transformers.from_pretrained (e.g., 'auto', 'cpu').",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for model weights (e.g., float16, bfloat16, float32, auto).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit quantization via bitsandbytes (requires compatible GPU setup).",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Enable 8-bit quantization via bitsandbytes.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tour = load_tour_from_colmap(args.poses_path, args.images_dir)

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Specify only one of --load-in-4bit or --load-in-8bit.")

    dtype = _resolve_dtype(args.torch_dtype)
    if (args.load_in_4bit or args.load_in_8bit) and dtype is not None:
        print("[WARN] Overriding --torch-dtype because quantization was enabled.", file=sys.stderr)
        dtype = None

    model_kwargs: dict[str, object] = {}
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    selector = Qwen3VLGoalSelector.from_pretrained(
        args.qwen_model,
        device_map=args.device_map,
        dtype=dtype,
        **model_kwargs,
    )

    mobility_vla = MobilityVLA(
        tour,
        config=MobilityVLAConfig(high_level_selector=selector),
    )

    instruction = Instruction(text=args.instruction)
    goal = mobility_vla.set_instruction(instruction)
    print(f"Selected goal frame: {goal}")

    frames = tour.frames[: max(1, min(args.steps, len(tour.frames)))]
    for idx, frame in enumerate(frames, start=1):
        hint = frame.metadata.get("keywords") if frame.metadata else frame.frame_id
        observation = Observation(hint=hint)
        result = mobility_vla.step(observation)
        action = result.action
        print(
            f"[Step {idx}] observation_hint='{hint}' "
            f"path={result.path} "
            f"action=(dx={action.dx:.2f}, dy={action.dy:.2f}, dtheta={action.dtheta:.2f}) "
            f"localized={result.localization.frame_id}"
        )


if __name__ == "__main__":
    main()
