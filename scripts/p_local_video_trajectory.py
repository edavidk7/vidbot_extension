import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_intrinsics(path):
    with open(path, "r") as f:
        data = json.load(f)

    if "camera_matrix" in data:
        matrix = np.array(data["camera_matrix"], dtype=np.float32)
        width, height = data.get("image_size_px", [0, 0])
    elif "intrinsic_matrix" in data:
        matrix = np.array(data["intrinsic_matrix"], dtype=np.float32).reshape(3, 3).T
        width = data.get("width", 0)
        height = data.get("height", 0)
    else:
        raise ValueError(
            f"Unsupported intrinsics format in {path}. Expected camera_matrix or intrinsic_matrix."
        )

    return {
        "matrix": matrix,
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
        "width": int(width),
        "height": int(height),
    }


def _write_camera_intrinsic(path, intrinsics, width, height):
    matrix = intrinsics["matrix"]
    payload = {
        "intrinsic_matrix": [
            float(matrix[0, 0]),
            0.0,
            0.0,
            0.0,
            float(matrix[1, 1]),
            0.0,
            float(matrix[0, 2]),
            float(matrix[1, 2]),
            1.0,
        ],
        "width": int(width),
        "height": int(height),
        "note": "Generated from calibration video via scripts/estimate_instrinsics.py",
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _extract_frames(video_path, color_dir):
    color_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_paths = []
    frame_idx = 0
    width = 0
    height = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        frame_path = color_dir / f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        frame_idx += 1

    cap.release()

    if not frame_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return frame_paths, width, height, fps


def _largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _candidate_from_mask(mask):
    contour = _largest_contour(mask)
    if contour is None:
        return None

    area = cv2.contourArea(contour)
    if area <= 1000:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    x, y, w, h = cv2.boundingRect(contour)
    return {
        "center": np.array([cx, cy], dtype=np.float32),
        "bbox": [int(x), int(y), int(w), int(h)],
        "area": float(area),
    }


def _score_candidate(candidate, previous_center, width):
    score = candidate["area"]
    center = candidate["center"]

    if previous_center is not None:
        score -= 75.0 * np.linalg.norm(center - previous_center)
    else:
        # Prefer the entering hand on the right half in this local demo.
        score += 0.5 * center[0]

    if center[0] < width * 0.5:
        score -= 1000.0

    return score


def _select_candidate(motion_skin_mask, motion_mask, previous_center, width):
    candidates = []

    primary = _candidate_from_mask(motion_skin_mask)
    if primary is not None:
        candidates.append(("motion_skin", primary))

    fallback = _candidate_from_mask(motion_mask)
    if fallback is not None:
        candidates.append(("motion", fallback))

    if not candidates:
        return None, None

    best_name = None
    best_candidate = None
    best_score = None
    for name, candidate in candidates:
        score = _score_candidate(candidate, previous_center, width)
        if best_score is None or score > best_score:
            best_name = name
            best_candidate = candidate
            best_score = score

    return best_name, best_candidate


def _smooth_points(points, window=7):
    if not points:
        return []

    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)

    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad = window // 2
    xs_pad = np.pad(xs, (pad, pad), mode="edge")
    ys_pad = np.pad(ys, (pad, pad), mode="edge")
    xs_smooth = np.convolve(xs_pad, kernel, mode="valid")
    ys_smooth = np.convolve(ys_pad, kernel, mode="valid")

    return [(float(x), float(y)) for x, y in zip(xs_smooth, ys_smooth)]


def _filter_interaction_band(trajectory, width, height):
    filtered = []
    for item in trajectory:
        x, y = item["point_smoothed"]
        if x < width * 0.55:
            continue
        if x > width * 0.95:
            continue
        if y < height * 0.30:
            continue
        if y > height * 0.85:
            continue
        filtered.append(item)

    if len(filtered) >= max(20, len(trajectory) // 4):
        return filtered
    return trajectory


def _track_motion(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read first frame from {video_path}")

    baseline = cv2.GaussianBlur(first_frame, (5, 5), 0)
    baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    previous_frame = baseline
    previous_gray = baseline_gray

    trajectory = []
    debug_frames = []
    previous_center = None
    frame_idx = 0
    height, width = first_frame.shape[:2]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)

        diff_baseline = cv2.absdiff(gray, baseline_gray)
        diff_previous = cv2.absdiff(gray, previous_gray)

        _, mask_baseline = cv2.threshold(diff_baseline, 18, 255, cv2.THRESH_BINARY)
        _, mask_previous = cv2.threshold(diff_previous, 10, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.bitwise_or(mask_baseline, mask_previous)

        motion_mask[: height // 8, :] = 0
        motion_mask[:, : width // 2] = 0
        kernel_small = np.ones((5, 5), dtype=np.uint8)
        kernel_large = np.ones((9, 9), dtype=np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel_small)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kernel_large)

        ycrcb = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 135))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel_small)

        motion_skin_mask = cv2.bitwise_and(motion_mask, skin_mask)
        candidate_kind, candidate = _select_candidate(
            motion_skin_mask, motion_mask, previous_center, width
        )

        if candidate is not None:
            center = candidate["center"]
            trajectory.append(
                {
                    "frame_idx": frame_idx + 1,
                    "point": [float(center[0]), float(center[1])],
                    "bbox": candidate["bbox"],
                    "source": candidate_kind,
                }
            )
            previous_center = center

            if len(debug_frames) < 10:
                debug_frame = frame.copy()
                x, y, w, h = candidate["bbox"]
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(
                    debug_frame,
                    (int(center[0]), int(center[1])),
                    6,
                    (0, 0, 255),
                    -1,
                )
                debug_frames.append((frame_idx + 1, debug_frame))

        previous_frame = frame_blur
        previous_gray = gray
        frame_idx += 1

    cap.release()

    if not trajectory:
        raise RuntimeError(
            "No motion trajectory detected. The local fallback tracker could not find a stable moving hand region."
        )

    smoothed = _smooth_points([item["point"] for item in trajectory])
    for item, point in zip(trajectory, smoothed):
        item["point_smoothed"] = [float(point[0]), float(point[1])]

    filtered_trajectory = _filter_interaction_band(trajectory, width, height)

    return first_frame, filtered_trajectory, debug_frames


def _color_for_index(index, total):
    progress = 0.0 if total <= 1 else index / float(total - 1)
    red = int(255 * (1.0 - progress) + 40 * progress)
    blue = int(40 * (1.0 - progress) + 255 * progress)
    return blue, 90, red


def _draw_overlay(first_frame, trajectory, output_path):
    canvas = first_frame.copy()
    points = [item["point_smoothed"] for item in trajectory]

    for idx, point in enumerate(points):
        color = _color_for_index(idx, len(points))
        x, y = int(point[0]), int(point[1])
        cv2.circle(canvas, (x, y), 5, color, -1)
        if idx > 0:
            prev = points[idx - 1]
            cv2.line(
                canvas,
                (int(prev[0]), int(prev[1])),
                (x, y),
                color,
                3,
            )

    start = tuple(int(v) for v in points[0])
    end = tuple(int(v) for v in points[-1])
    cv2.circle(canvas, start, 10, (0, 255, 0), 2)
    cv2.circle(canvas, end, 10, (255, 0, 255), 2)
    cv2.putText(
        canvas,
        "start",
        (start[0] + 12, start[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        canvas,
        "end",
        (end[0] + 12, end[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
    )
    cv2.imwrite(str(output_path), canvas)


def _draw_timeseries(trajectory, output_path):
    frame_ids = [item["frame_idx"] for item in trajectory]
    xs = [item["point_smoothed"][0] for item in trajectory]
    ys = [item["point_smoothed"][1] for item in trajectory]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frame_ids, xs, color="tab:blue", linewidth=2)
    axes[0].set_ylabel("x (px)")
    axes[0].set_title("Tracked Motion Trajectory")
    axes[0].grid(alpha=0.25)

    axes[1].plot(frame_ids, ys, color="tab:orange", linewidth=2)
    axes[1].set_ylabel("y (px)")
    axes[1].set_xlabel("frame")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_debug_frames(debug_frames, debug_dir):
    debug_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame in debug_frames:
        cv2.imwrite(str(debug_dir / f"debug_frame_{frame_idx:06d}.jpg"), frame)


def main():
    parser = argparse.ArgumentParser(
        description="Local trajectory extraction for a plain MP4 using 6sense-style outputs."
    )
    parser.add_argument("--video", required=True, help="Path to the source MP4 video")
    parser.add_argument(
        "--intrinsics",
        required=True,
        help="Calibration JSON from scripts/estimate_instrinsics.py or camera_intrinsic.json",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where dataset-style frames and trajectory outputs will be written",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    color_dir = output_dir / "color"
    trajectory_dir = output_dir / "trajectory"
    debug_dir = trajectory_dir / "debug"

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = _load_intrinsics(args.intrinsics)
    frame_paths, width, height, fps = _extract_frames(video_path, color_dir)
    _write_camera_intrinsic(
        output_dir / "camera_intrinsic.json", intrinsics, width=width, height=height
    )

    first_frame, trajectory, debug_frames = _track_motion(video_path)
    _write_debug_frames(debug_frames, debug_dir)

    overlay_path = trajectory_dir / "trajectory_visualization.jpg"
    _draw_overlay(first_frame, trajectory, overlay_path)

    timeseries_path = trajectory_dir / "trajectory_xy.png"
    _draw_timeseries(trajectory, timeseries_path)

    payload = {
        "video": str(video_path),
        "frame_count": len(frame_paths),
        "fps": float(fps),
        "intrinsics": {
            "fx": intrinsics["fx"],
            "fy": intrinsics["fy"],
            "cx": intrinsics["cx"],
            "cy": intrinsics["cy"],
            "width": width,
            "height": height,
        },
        "trajectory": trajectory,
        "artifacts": {
            "overlay": str(overlay_path),
            "timeseries": str(timeseries_path),
            "camera_intrinsic": str(output_dir / "camera_intrinsic.json"),
        },
    }

    metadata_path = trajectory_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote dataset-style frames to {color_dir}")
    print(f"Wrote camera intrinsics to {output_dir / 'camera_intrinsic.json'}")
    print(f"Wrote trajectory overlay to {overlay_path}")
    print(f"Wrote trajectory timeseries to {timeseries_path}")
    print(f"Wrote trajectory metadata to {metadata_path}")


if __name__ == "__main__":
    main()
