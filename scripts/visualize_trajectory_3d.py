"""Visualize a tracked 2D trajectory in camera-space 3D.

This script combines:
  - trajectory/metadata.json from scripts/p_local_video_trajectory.py
  - camera_intrinsic.json
  - per-frame depth maps from depth_m3d, depth_dav3, or depth

It can render either:
  - a trajectory-only 3D figure
  - or a richer figure with representative RGB/depth context and a sparse
    scene point cloud

Note:
  Without per-frame camera poses this stays in camera coordinates, not a
  world-stabilized trajectory. If the camera moves, the 3D path includes that
  ego-motion.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def resolve_dataset_path(dataset, dataset_dir):
    dataset_path = Path(dataset)
    if dataset_path.is_dir():
        return dataset_path
    return Path(dataset_dir) / dataset


def resolve_depth_dir(dataset_path, depth_dir):
    if depth_dir is not None:
        candidate = Path(depth_dir)
        if candidate.is_dir():
            return candidate
        candidate = dataset_path / depth_dir
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    for name in ("depth_m3d", "depth_dav3", "depth"):
        candidate = dataset_path / name
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"No depth directory found under {dataset_path}. "
        "Expected one of depth_m3d, depth_dav3, or depth."
    )


def frame_image_path(frame_dir, frame_idx):
    stem = f"{int(frame_idx):06d}"
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = frame_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for frame {frame_idx} in {frame_dir}")


def load_color_frame(dataset_path, frame_idx):
    color_path = frame_image_path(dataset_path / "color", frame_idx)
    with Image.open(color_path) as img:
        return np.array(img.convert("RGB"))


def load_depth_frame(depth_dir, frame_idx):
    depth_path = depth_dir / f"{int(frame_idx):06d}.png"
    if not depth_path.exists():
        raise FileNotFoundError(depth_path)

    with Image.open(depth_path) as img:
        depth = np.array(img)

    if np.issubdtype(depth.dtype, np.integer):
        depth_m = depth.astype(np.float32) / 1000.0
    else:
        depth_m = depth.astype(np.float32)
    return depth_m


def load_intrinsic(dataset_path):
    intr_path = dataset_path / "camera_intrinsic.json"
    with open(intr_path, "r") as f:
        info = json.load(f)
    return np.array(info["intrinsic_matrix"], dtype=np.float32).reshape(3, 3).T


def load_trajectory(metadata_path):
    with open(metadata_path, "r") as f:
        payload = json.load(f)
    if "trajectory" not in payload:
        raise ValueError(f"No trajectory field found in {metadata_path}")
    return payload["trajectory"]


def sample_local_depth(depth_m, u, v, window, min_depth, max_depth):
    height, width = depth_m.shape[:2]
    radius = max(int(window) // 2, 0)
    u0 = max(int(round(u)) - radius, 0)
    u1 = min(int(round(u)) + radius + 1, width)
    v0 = max(int(round(v)) - radius, 0)
    v1 = min(int(round(v)) + radius + 1, height)

    patch = depth_m[v0:v1, u0:u1]
    valid = np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)
    if valid.any():
        return float(np.median(patch[valid]))
    return None


def sample_bbox_depth(depth_m, bbox, min_depth, max_depth):
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None

    height, width = depth_m.shape[:2]
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, width)
    y1 = min(y + h, height)
    if x0 >= x1 or y0 >= y1:
        return None

    patch = depth_m[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)
    if valid.any():
        return float(np.median(patch[valid]))
    return None


def sample_depth_for_item(depth_m, item, window, min_depth, max_depth):
    point = item.get("point_smoothed", item.get("point"))
    if point is None:
        return None

    depth = sample_local_depth(
        depth_m,
        u=point[0],
        v=point[1],
        window=window,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    if depth is not None:
        return depth

    bbox = item.get("bbox")
    if bbox is not None:
        return sample_bbox_depth(depth_m, bbox, min_depth=min_depth, max_depth=max_depth)

    return None


def backproject_pixel(u, v, z, intr):
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    x = (float(u) - cx) * float(z) / fx
    y = (float(v) - cy) * float(z) / fy
    return np.array([x, y, float(z)], dtype=np.float32)


def build_scene_cloud(depth_m, intr, color_rgb, stride, min_depth, max_depth):
    height, width = depth_m.shape[:2]
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    zs = depth_m[::stride, ::stride]

    valid = np.isfinite(zs) & (zs >= min_depth) & (zs <= max_depth)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid].astype(np.float32)

    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])

    points = np.stack(
        [
            (xs - cx) * zs / fx,
            (ys - cy) * zs / fy,
            zs,
        ],
        axis=-1,
    )

    colors = color_rgb[::stride, ::stride][valid].astype(np.float32) / 255.0
    return points, colors


def to_display_coords(points):
    points = np.asarray(points, dtype=np.float32)
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=-1)


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius < 1e-6:
        radius = 0.1

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def lift_trajectory(dataset_path, trajectory, depth_dir, intr, window, min_depth, max_depth):
    lifted = []
    skipped_missing_depth = 0
    skipped_invalid_depth = 0

    for item in trajectory:
        frame_idx = int(item["frame_idx"])
        point = item.get("point_smoothed", item.get("point"))
        if point is None:
            skipped_invalid_depth += 1
            continue

        try:
            depth_m = load_depth_frame(depth_dir, frame_idx)
        except FileNotFoundError:
            skipped_missing_depth += 1
            continue

        depth_value = sample_depth_for_item(
            depth_m,
            item,
            window=window,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if depth_value is None:
            skipped_invalid_depth += 1
            continue

        xyz = backproject_pixel(point[0], point[1], depth_value, intr)
        lifted.append(
            {
                "frame_idx": frame_idx,
                "u": float(point[0]),
                "v": float(point[1]),
                "depth_m": float(depth_value),
                "xyz": xyz,
            }
        )

    return lifted, skipped_missing_depth, skipped_invalid_depth


def draw_trajectory_only_panel(ax, lifted, title):
    traj_points = np.array([item["xyz"] for item in lifted], dtype=np.float32)
    traj_display = to_display_coords(traj_points)
    traj_colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(traj_display)))

    for idx in range(1, len(traj_display)):
        ax.plot(
            traj_display[idx - 1 : idx + 1, 0],
            traj_display[idx - 1 : idx + 1, 1],
            traj_display[idx - 1 : idx + 1, 2],
            color=traj_colors[idx],
            linewidth=3.0,
        )

    ax.scatter(
        traj_display[:, 0],
        traj_display[:, 1],
        traj_display[:, 2],
        c=traj_colors,
        s=42,
        depthshade=False,
    )
    ax.scatter(
        traj_display[0, 0],
        traj_display[0, 1],
        traj_display[0, 2],
        c="lime",
        s=150,
        marker="o",
        edgecolors="black",
        depthshade=False,
    )
    ax.scatter(
        traj_display[-1, 0],
        traj_display[-1, 1],
        traj_display[-1, 2],
        c="magenta",
        s=170,
        marker="X",
        edgecolors="black",
        depthshade=False,
    )

    axis_len = max(0.1, float(np.median(traj_points[:, 2]) * 0.1))
    ax.quiver(0.0, 0.0, 0.0, axis_len, 0.0, 0.0, color="red", linewidth=2)
    ax.quiver(0.0, 0.0, 0.0, 0.0, axis_len, 0.0, color="blue", linewidth=2)
    ax.quiver(0.0, 0.0, 0.0, 0.0, 0.0, axis_len, color="green", linewidth=2)
    ax.text(axis_len, 0.0, 0.0, "X", color="red")
    ax.text(0.0, axis_len, 0.0, "Z", color="blue")
    ax.text(0.0, 0.0, axis_len, "-Y", color="green")

    set_axes_equal(ax, traj_display)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("-Y (m)")
    ax.set_title(title)
    ax.view_init(elev=24, azim=-62)


def draw_rgb_panel(ax, color_rgb, lifted, rep_frame_idx):
    ax.imshow(color_rgb)
    if not lifted:
        ax.set_title(f"Representative RGB Frame {rep_frame_idx}")
        ax.axis("off")
        return

    coords = np.array([[item["u"], item["v"]] for item in lifted], dtype=np.float32)
    colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(lifted)))

    for idx in range(1, len(coords)):
        ax.plot(
            coords[idx - 1 : idx + 1, 0],
            coords[idx - 1 : idx + 1, 1],
            color=colors[idx],
            linewidth=2.0,
        )

    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=18, linewidths=0)
    ax.scatter(coords[0, 0], coords[0, 1], c="lime", s=90, marker="o", edgecolors="black")
    ax.scatter(coords[-1, 0], coords[-1, 1], c="magenta", s=90, marker="X", edgecolors="black")
    ax.set_title(f"2D Trajectory Overlay (Frame {rep_frame_idx})")
    ax.axis("off")


def draw_depth_panel(ax, depth_m, lifted, rep_frame_idx, max_depth):
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_depth)
    if valid.any():
        vmin = float(np.percentile(depth_m[valid], 2))
        vmax = float(np.percentile(depth_m[valid], 98))
    else:
        vmin, vmax = 0.0, max_depth

    depth_vis = np.where(valid, depth_m, np.nan)
    im = ax.imshow(depth_vis, cmap="turbo", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Depth (m)")

    if lifted:
        coords = np.array([[item["u"], item["v"]] for item in lifted], dtype=np.float32)
        colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(lifted)))
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=18, linewidths=0)

    ax.set_title(f"Representative Depth Frame {rep_frame_idx}")
    ax.axis("off")


def draw_3d_panel(ax, scene_points, scene_colors, lifted):
    scene_display = to_display_coords(scene_points)
    traj_points = np.array([item["xyz"] for item in lifted], dtype=np.float32)
    traj_display = to_display_coords(traj_points)
    traj_colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(traj_display)))

    if len(scene_display) > 0:
        ax.scatter(
            scene_display[:, 0],
            scene_display[:, 1],
            scene_display[:, 2],
            c=scene_colors,
            s=0.6,
            alpha=0.35,
            linewidths=0,
        )

    for idx in range(1, len(traj_display)):
        ax.plot(
            traj_display[idx - 1 : idx + 1, 0],
            traj_display[idx - 1 : idx + 1, 1],
            traj_display[idx - 1 : idx + 1, 2],
            color=traj_colors[idx],
            linewidth=2.5,
        )

    ax.scatter(
        traj_display[:, 0],
        traj_display[:, 1],
        traj_display[:, 2],
        c=traj_colors,
        s=30,
        depthshade=False,
    )
    ax.scatter(
        traj_display[0, 0],
        traj_display[0, 1],
        traj_display[0, 2],
        c="lime",
        s=120,
        marker="o",
        edgecolors="black",
        depthshade=False,
    )
    ax.scatter(
        traj_display[-1, 0],
        traj_display[-1, 1],
        traj_display[-1, 2],
        c="magenta",
        s=140,
        marker="X",
        edgecolors="black",
        depthshade=False,
    )

    axis_len = max(0.1, float(np.median(traj_points[:, 2]) * 0.1))
    ax.quiver(0.0, 0.0, 0.0, axis_len, 0.0, 0.0, color="red", linewidth=2)
    ax.quiver(0.0, 0.0, 0.0, 0.0, axis_len, 0.0, color="blue", linewidth=2)
    ax.quiver(0.0, 0.0, 0.0, 0.0, 0.0, axis_len, color="green", linewidth=2)
    ax.text(axis_len, 0.0, 0.0, "X", color="red")
    ax.text(0.0, axis_len, 0.0, "Z", color="blue")
    ax.text(0.0, 0.0, axis_len, "-Y", color="green")

    set_axes_equal(ax, np.concatenate([scene_display, traj_display], axis=0))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("-Y (m)")
    ax.set_title("Camera-Space 3D Trajectory")
    ax.view_init(elev=24, azim=-62)


def default_output_path(dataset_path):
    traj_dir = dataset_path / "trajectory"
    if traj_dir.is_dir():
        return traj_dir / "trajectory_3d_camera_space.png"
    return dataset_path / "trajectory_3d_camera_space.png"


def visualize(
    dataset_path,
    metadata_path,
    depth_dir,
    output_path,
    scene_frame,
    depth_window,
    scene_stride,
    min_depth,
    max_depth,
    trajectory_only,
):
    intr = load_intrinsic(dataset_path)
    trajectory = load_trajectory(metadata_path)
    lifted, skipped_missing_depth, skipped_invalid_depth = lift_trajectory(
        dataset_path=dataset_path,
        trajectory=trajectory,
        depth_dir=depth_dir,
        intr=intr,
        window=depth_window,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    if not lifted:
        raise RuntimeError(
            "No 3D trajectory points could be lifted. "
            "Check that per-frame depth exists and overlaps the tracked pixels."
        )

    if trajectory_only:
        fig = plt.figure(figsize=(8, 8), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        draw_trajectory_only_panel(
            ax,
            lifted=lifted,
            title=f"{dataset_path.name} Trajectory In Camera Space",
        )
        fig.text(
            0.5,
            0.02,
            "Camera-space trajectory only. Per-frame depth is still required to lift 2D points into 3D.",
            ha="center",
            fontsize=10,
            color="dimgray",
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"Saved 3D trajectory visualization to {output_path}")
        print(f"Used {len(lifted)} / {len(trajectory)} trajectory points")
        print(f"Skipped {skipped_missing_depth} points with missing depth frames")
        print(f"Skipped {skipped_invalid_depth} points with invalid depth samples")
        print(f"Depth source: {depth_dir}")
        return

    if scene_frame is None:
        scene_frame = int(lifted[len(lifted) // 2]["frame_idx"])

    color_rgb = load_color_frame(dataset_path, scene_frame)
    depth_m = load_depth_frame(depth_dir, scene_frame)
    scene_points, scene_colors = build_scene_cloud(
        depth_m=depth_m,
        intr=intr,
        color_rgb=color_rgb,
        stride=scene_stride,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    fig = plt.figure(figsize=(22, 7), facecolor="white")
    fig.suptitle(
        "Tracked Trajectory Lifted Into Camera-Space 3D\n"
        f"{dataset_path.name} | {len(lifted)}/{len(trajectory)} points with depth | "
        f"scene frame {scene_frame}",
        fontsize=15,
        y=0.98,
    )

    ax1 = fig.add_subplot(1, 3, 1)
    draw_rgb_panel(ax1, color_rgb=color_rgb, lifted=lifted, rep_frame_idx=scene_frame)

    ax2 = fig.add_subplot(1, 3, 2)
    draw_depth_panel(ax2, depth_m=depth_m, lifted=lifted, rep_frame_idx=scene_frame, max_depth=max_depth)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    draw_3d_panel(ax3, scene_points=scene_points, scene_colors=scene_colors, lifted=lifted)

    fig.text(
        0.5,
        0.015,
        "This is camera-space, not world-space. Without per-frame poses, camera motion is not removed.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved 3D trajectory visualization to {output_path}")
    print(f"Used {len(lifted)} / {len(trajectory)} trajectory points")
    print(f"Skipped {skipped_missing_depth} points with missing depth frames")
    print(f"Skipped {skipped_invalid_depth} points with invalid depth samples")
    print(f"Depth source: {depth_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Lift trajectory/metadata.json into camera-space 3D using per-frame depth."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Dataset folder name inside datasets/ or a direct dataset path",
    )
    parser.add_argument(
        "--dataset_dir",
        default="./datasets",
        help="Dataset root used when --dataset is not an explicit path",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to trajectory metadata JSON (default: <dataset>/trajectory/metadata.json)",
    )
    parser.add_argument(
        "--depth_dir",
        default=None,
        help="Depth directory name or path (default: auto-detect depth_m3d/depth_dav3/depth)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: <dataset>/trajectory/trajectory_3d_camera_space.png)",
    )
    parser.add_argument(
        "--scene_frame",
        type=int,
        default=None,
        help="Representative frame used for the RGB/depth scene context",
    )
    parser.add_argument(
        "--depth_window",
        type=int,
        default=9,
        help="Odd pixel window size used for local depth sampling around each trajectory point",
    )
    parser.add_argument(
        "--scene_stride",
        type=int,
        default=10,
        help="Stride used when subsampling the representative depth map into a point cloud",
    )
    parser.add_argument(
        "--trajectory_only",
        action="store_true",
        help="Render only the lifted 3D trajectory, without RGB/depth or scene context",
    )
    parser.add_argument("--min_depth", type=float, default=0.05)
    parser.add_argument("--max_depth", type=float, default=5.0)
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.dataset, args.dataset_dir)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    metadata_path = Path(args.metadata) if args.metadata is not None else dataset_path / "trajectory" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Trajectory metadata not found: {metadata_path}")

    depth_dir = resolve_depth_dir(dataset_path, args.depth_dir)
    output_path = Path(args.output) if args.output is not None else default_output_path(dataset_path)

    visualize(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        depth_dir=depth_dir,
        output_path=output_path,
        scene_frame=args.scene_frame,
        depth_window=args.depth_window,
        scene_stride=max(1, args.scene_stride),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        trajectory_only=args.trajectory_only,
    )


if __name__ == "__main__":
    main()
