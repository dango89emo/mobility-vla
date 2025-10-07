"""End-to-end automation for processing an iPhone tour video with PyCOLMAP.

The script orchestrates:
1. Directory setup and frame extraction via FFmpeg.
2. A first SfM pass that estimates intrinsics (focal length & distortion).
3. A second SfM pass that re-runs COLMAP with intrinsics locked.

Example:
    python scripts/run_iphone_colmap_pipeline.py \\
        --video-path data/sceaux_tour.mp4 \\
        --workspace-dir data/iphone_colmap \\
        --fps 2 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from run_pycolmap_sfm import run_pycolmap_pipeline

try:  # Pillow is optional; used for EXIF if available.
    from PIL import Image, ExifTags
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]
    ExifTags = None  # type: ignore[assignment]


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    params: List[float]
    source: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Path to the recorded iPhone tour video (MOV/MP4).",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("data/iphone_colmap"),
        help="Root directory for generated artefacts (frames, COLMAP outputs).",
    )
    parser.add_argument(
        "--frames-subdir",
        default="frames",
        help="Relative name for the frame extraction directory.",
    )
    parser.add_argument(
        "--calibration-subdir",
        default="reconstruction_pass1",
        help="Relative name for the first-pass (intrinsics estimation) output.",
    )
    parser.add_argument(
        "--reconstruction-subdir",
        default="reconstruction_final",
        help="Relative name for the final reconstruction with locked intrinsics.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frame extraction rate for FFmpeg (frames per second).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="FFmpeg executable to use for frame extraction.",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="ffprobe executable for querying video metadata.",
    )
    parser.add_argument(
        "--default-fov-deg",
        type=float,
        default=60.0,
        help="Fallback horizontal field-of-view (degrees) if EXIF is unavailable.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Maximum longer-side size for SIFT extraction in PyCOLMAP.",
    )
    parser.add_argument(
        "--camera-model",
        default="OPENCV",
        help="COLMAP camera model to assume when importing frames.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration in PyCOLMAP where available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow directories to be cleared if they already exist.",
    )
    return parser.parse_args()


def _ensure_command_available(executable: str) -> None:
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"Executable '{executable}' was not found in PATH. Provide a valid path via CLI arguments."
        )


def _prepare_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite and any(path.iterdir()):
            raise FileExistsError(
                f"Directory '{path}' already exists. Re-run with --overwrite to replace its contents."
            )
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _extract_frames(
    ffmpeg_bin: str,
    video_path: Path,
    frames_dir: Path,
    fps: float,
) -> None:
    if fps <= 0.0:
        raise ValueError("--fps must be positive.")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video '{video_path}' does not exist.")

    output_template = frames_dir / "frame_%04d.jpg"
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(output_template),
    ]
    subprocess.run(command, check=True)


def _probe_video_dimensions(ffprobe_bin: str, video_path: Path) -> Tuple[int, int]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe did not return stream metadata for '{video_path}'.")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    return width, height


def _collect_frames(frames_dir: Path) -> List[Path]:
    frames = sorted(
        [*frames_dir.glob("*.jpg"), *frames_dir.glob("*.jpeg"), *frames_dir.glob("*.png")]
    )
    return frames


def _rational_to_float(value: object) -> Optional[float]:
    if isinstance(value, tuple) and len(value) == 2:
        num, denom = value
        denom_f = float(denom)
        if denom_f == 0.0:
            return None
        return float(num) / denom_f
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _estimate_focal_from_exif(frame_path: Path, width: int) -> Tuple[Optional[float], str]:
    if Image is None:  # Pillow missing
        return None, "no_pillow"

    try:
        with Image.open(frame_path) as img:
            exif_raw = img.getexif()
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"exif_read_error:{exc}"

    if not exif_raw:
        return None, "no_exif"

    exif: Dict[str, object] = {}
    for tag, value in exif_raw.items():
        tag_name = ExifTags.TAGS.get(tag, tag) if ExifTags else tag
        exif[tag_name] = value

    focal_35mm = _rational_to_float(exif.get("FocalLengthIn35mmFilm"))
    if focal_35mm:
        focal_px = (focal_35mm / 36.0) * width
        return focal_px, "exif:FocalLengthIn35mmFilm"

    focal_mm = _rational_to_float(exif.get("FocalLength"))
    fpx_res = _rational_to_float(exif.get("FocalPlaneXResolution"))
    unit = exif.get("FocalPlaneResolutionUnit")

    if focal_mm and fpx_res and isinstance(unit, int):
        # Convert resolution to per-millimeter.
        unit_scale = {
            2: 25.4,  # inches → mm
            3: 10.0,  # centimeters → mm
            4: 1.0,  # millimeters
            5: 0.001,  # micrometers
        }.get(unit)
        if unit_scale:
            pixels_per_mm = fpx_res / unit_scale
            if pixels_per_mm > 0.0:
                sensor_width_mm = width / pixels_per_mm
                focal_px = (focal_mm / sensor_width_mm) * width
                return focal_px, "exif:FocalPlaneResolution"

    return None, "unsupported_exif"


def _initial_params_for_camera_model(
    camera_model: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> List[float]:
    model = camera_model.upper()
    if model == "OPENCV":
        return [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0]
    if model == "FULL_OPENCV":
        return [fx, fy, cx, cy] + [0.0] * 8
    if model == "OPENCV_FISHEYE":
        return [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0]
    if model == "PINHOLE":
        return [fx, fy, cx, cy]
    if model == "SIMPLE_PINHOLE":
        return [fx, cx, cy]
    if model == "SIMPLE_RADIAL":
        return [fx, cx, cy, 0.0]
    if model == "RADIAL":
        return [fx, cx, cy, 0.0, 0.0]
    raise ValueError(
        f"Automatic intrinsics initialisation is not implemented for camera model '{camera_model}'."
    )


def _estimate_intrinsics(
    frames_dir: Path,
    video_path: Path,
    ffprobe_bin: str,
    default_fov_deg: float,
    camera_model: str,
) -> CameraIntrinsics:
    frames = _collect_frames(frames_dir)
    if not frames:
        raise RuntimeError(
            f"No frames were found in '{frames_dir}'. Increase --fps or verify FFmpeg output."
        )
    first_frame = frames[0]

    width: Optional[int] = None
    height: Optional[int] = None
    if Image is not None:
        try:
            with Image.open(first_frame) as probe:
                width, height = probe.size
        except Exception:
            width = height = None

    if width is None or height is None:
        width, height = _probe_video_dimensions(ffprobe_bin, video_path)

    focal_px, source = _estimate_focal_from_exif(first_frame, width)

    if focal_px is None:
        # Default fallback: assume horizontal FOV specified.
        half_fov_rad = math.radians(default_fov_deg) / 2.0
        focal_px = (width / 2.0) / math.tan(half_fov_rad)
        source = f"default_fov:{default_fov_deg:.1f}"

    fx = focal_px
    fy = focal_px  # square pixels assumption
    cx = width / 2.0
    cy = height / 2.0
    try:
        params = _initial_params_for_camera_model(camera_model, fx, fy, cx, cy)
    except ValueError as exc:
        raise ValueError(
            f"{exc} Supported models: OPENCV, FULL_OPENCV, OPENCV_FISHEYE, "
            "PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL."
        ) from exc
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        params=params,
        source=source,
    )


def _write_intrinsics_summary(
    path: Path,
    *,
    camera_model: str,
    initial: CameraIntrinsics,
    refined_params: Sequence[float],
    pass1_summary: Dict[str, object],
    pass2_summary: Dict[str, object],
) -> None:
    payload = {
        "camera_model": camera_model,
        "image_size": {"width": initial.width, "height": initial.height},
        "initial_intrinsics": {
            "fx": initial.fx,
            "fy": initial.fy,
            "cx": initial.cx,
            "cy": initial.cy,
            "params": initial.params,
            "source": initial.source,
        },
        "refined_params": list(refined_params),
        "pass1_summary": pass1_summary,
        "pass2_summary": pass2_summary,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = _parse_args()
    _ensure_command_available(args.ffmpeg_bin)
    _ensure_command_available(args.ffprobe_bin)

    workspace_dir = args.workspace_dir
    frames_dir = workspace_dir / args.frames_subdir
    pass1_dir = workspace_dir / args.calibration_subdir
    pass2_dir = workspace_dir / args.reconstruction_subdir

    workspace_dir.mkdir(parents=True, exist_ok=True)
    _prepare_directory(frames_dir, overwrite=args.overwrite)

    _extract_frames(
        ffmpeg_bin=args.ffmpeg_bin,
        video_path=args.video_path,
        frames_dir=frames_dir,
        fps=args.fps,
    )

    intrinsics = _estimate_intrinsics(
        frames_dir=frames_dir,
        video_path=args.video_path,
        ffprobe_bin=args.ffprobe_bin,
        default_fov_deg=args.default_fov_deg,
        camera_model=args.camera_model,
    )

    print(
        f"[INFO] Initial intrinsics source: {intrinsics.source}, fx ≈ {intrinsics.fx:.2f}px",
        file=sys.stderr,
    )

    pass1_result = run_pycolmap_pipeline(
        image_dir=frames_dir,
        output_dir=pass1_dir,
        camera_model=args.camera_model,
        max_image_size=args.max_image_size,
        use_gpu=args.use_gpu,
        overwrite=args.overwrite,
        initial_camera_params=intrinsics.params,
        refine_intrinsics=True,
        return_reconstruction=True,
    )

    if not isinstance(pass1_result, tuple) or pass1_result[1] is None:
        raise RuntimeError("First COLMAP pass did not return a reconstruction object.")

    pass1_summary, reconstruction = pass1_result
    cameras = reconstruction.cameras
    if not cameras:
        raise RuntimeError("No cameras were reconstructed during the first pass.")
    # Assume a single camera model; pick the most common.
    best_camera = next(iter(cameras.values()))
    refined_params = list(best_camera.params)

    if len(refined_params) >= 4:
        print(
            "[INFO] Refinement completed. "
            f"fx={refined_params[0]:.2f}, fy={refined_params[1]:.2f}, "
            f"cx={refined_params[2]:.2f}, cy={refined_params[3]:.2f}",
            file=sys.stderr,
        )
    else:  # pragma: no cover - defensive
        print(
            "[INFO] Refinement completed. Camera parameters:"
            f" {', '.join(f'{v:.4f}' for v in refined_params)}",
            file=sys.stderr,
        )

    pass2_summary = run_pycolmap_pipeline(
        image_dir=frames_dir,
        output_dir=pass2_dir,
        camera_model=args.camera_model,
        max_image_size=args.max_image_size,
        use_gpu=args.use_gpu,
        overwrite=args.overwrite,
        initial_camera_params=refined_params,
        refine_intrinsics=False,
        return_reconstruction=False,
    )

    intrinsics_summary_path = workspace_dir / "intrinsics_summary.json"
    _write_intrinsics_summary(
        intrinsics_summary_path,
        camera_model=args.camera_model,
        initial=intrinsics,
        refined_params=refined_params,
        pass1_summary=pass1_summary,
        pass2_summary=pass2_summary,
    )

    result = {
        "video": str(args.video_path),
        "frames_dir": str(frames_dir),
        "calibration_reconstruction_dir": str(pass1_dir),
        "final_reconstruction_dir": str(pass2_dir),
        "intrinsics_source": intrinsics.source,
        "intrinsics_summary_path": str(intrinsics_summary_path),
        "pass1_summary": pass1_summary,
        "pass2_summary": pass2_summary,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
