#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int = 1,
    image_format: str = "jpg",
    max_frames: int | None = None,
) -> None:
    """
    Extract frames from a video file.

    Parameters
    ----------
    video_path:
        Path to the input video.
    output_dir:
        Directory where extracted frames will be saved.
    every_n_frames:
        Save one frame every N frames. Use 1 to save all frames.
    image_format:
        Output image format, usually jpg or png.
    max_frames:
        Optional maximum number of frames to save.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_index = 0
    saved_count = 0

    video_stem = video_path.stem

    while True:
        success, frame = capture.read()

        if not success:
            break

        if frame_index % every_n_frames == 0:
            output_path = output_dir / f"{video_stem}_frame_{frame_index:06d}.{image_format}"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

            if max_frames is not None and saved_count >= max_frames:
                break

        frame_index += 1

    capture.release()

    print(f"Video: {video_path}")
    print(f"Total frames read: {frame_index}")
    print(f"Frames saved: {saved_count}")
    print(f"Output directory: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from a UAV/drone video for pHash duplicate analysis."
    )

    parser.add_argument(
        "--video-path",
        required=True,
        type=Path,
        help="Path to the input video file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where extracted frames will be saved.",
    )

    parser.add_argument(
        "--every-n-frames",
        default=1,
        type=int,
        help="Save one frame every N frames. Default: 1.",
    )

    parser.add_argument(
        "--image-format",
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format. Default: jpg.",
    )

    parser.add_argument(
        "--max-frames",
        default=None,
        type=int,
        help="Optional maximum number of frames to save.",
    )

    args = parser.parse_args()

    extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        every_n_frames=args.every_n_frames,
        image_format=args.image_format,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
