"""Visualize VidBot inference results from saved NPZ files.

Shows: detection results, contact/goal heatmaps, predicted trajectories,
grasp poses with coordinate frames.
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import proj3d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


def draw_coord_frame(ax, origin, rotation=None, scale=0.03, labels=True):
    """Draw a 3D coordinate frame at origin."""
    if rotation is None:
        rotation = np.eye(3)
    colors = ["red", "green", "blue"]
    axis_labels = ["X", "Y", "Z"]
    for i in range(3):
        direction = rotation[:, i] * scale
        arrow = Arrow3D(
            [origin[0], origin[0] + direction[0]],
            [origin[1], origin[1] + direction[1]],
            [origin[2], origin[2] + direction[2]],
            mutation_scale=10,
            lw=2,
            arrowstyle="-|>",
            color=colors[i],
        )
        ax.add_artist(arrow)
        if labels:
            ax.text(origin[0] + direction[0] * 1.3, origin[1] + direction[1] * 1.3, origin[2] + direction[2] * 1.3, axis_labels[i], color=colors[i], fontsize=8)


def load_intrinsic(dataset_path):
    with open(os.path.join(dataset_path, "camera_intrinsic.json")) as f:
        info = json.load(f)
    return np.array(info["intrinsic_matrix"]).reshape(3, 3).T.astype(np.float32)


def visualize_results(dataset_path, frame_id, out_dir=None):
    """Create comprehensive visualization of pipeline results."""
    if out_dir is None:
        out_dir = os.path.join(dataset_path, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    # Load original image
    color_path = os.path.join(dataset_path, "color", f"{frame_id:06d}.png")
    if not os.path.exists(color_path):
        color_path = color_path.replace(".png", ".jpg")
    color_bgr = cv2.imread(color_path)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # Load scene meta (detection results)
    meta_files = sorted([f for f in os.listdir(os.path.join(dataset_path, "scene_meta")) if f.startswith(f"{frame_id:06d}")])
    pred_files = sorted([f for f in os.listdir(os.path.join(dataset_path, "prediction")) if f.startswith(f"{frame_id:06d}")])

    if not meta_files:
        print(f"No results found for frame {frame_id}")
        return

    meta = np.load(os.path.join(dataset_path, "scene_meta", meta_files[0]), allow_pickle=True)

    # ---- Figure 1: Detection Overview ----
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor="white")
    fig.suptitle(f"VidBot Detection Results — Frame {frame_id}", fontsize=16, fontweight="bold")

    # Panel 1: Original image with bounding boxes (full-res coords)
    ax = axes[0]
    ax.imshow(color_rgb)
    if "bbox_raw_all" in meta:
        bboxes = meta["bbox_raw_all"]
        if bboxes.ndim == 3:
            bboxes = bboxes[0]  # unbatch
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=plt.cm.tab10(i), facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"Object {i}", color=plt.cm.tab10(i), fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    ax.set_title("Detected Objects", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Panel 2: Object masks (full-resolution masks)
    ax = axes[1]
    ax.imshow(color_rgb, alpha=0.5)
    if "object_bbox_mask_all" in meta:
        masks = meta["object_bbox_mask_all"]
        if masks.ndim == 4:
            masks = masks[0]  # unbatch -> (N_obj, H, W)
        for i, mask in enumerate(masks):
            color_overlay = np.zeros((*color_rgb.shape[:2], 4))
            c = plt.cm.tab10(i)[:3]
            color_overlay[mask > 0.5] = [*c, 0.5]
            ax.imshow(color_overlay)
    ax.set_title("Object Segmentation Masks", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Panel 3: Depth with detections
    depth_path = os.path.join(dataset_path, "depth_m3d", f"{frame_id:06d}.png")
    if not os.path.exists(depth_path):
        depth_path = os.path.join(dataset_path, "depth", f"{frame_id:06d}.png")
    ax = axes[2]
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        depth[depth > 5] = np.nan
        ax.imshow(depth, cmap="turbo")
        if "bbox_raw_all" in meta:
            bboxes = meta["bbox_raw_all"]
            if bboxes.ndim == 3:
                bboxes = bboxes[0]
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="white", facecolor="none", linestyle="--")
                ax.add_patch(rect)
    ax.set_title("Depth + Detections", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    path1 = os.path.join(out_dir, f"detection_{frame_id:06d}.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path1}")

    # ---- Figure 2: Per-object predictions ----
    n_objects = len(pred_files)
    if n_objects == 0:
        return

    n_show = min(n_objects, 5)
    fig = plt.figure(figsize=(24, 6 * n_show), facecolor="white")
    fig.suptitle(f"VidBot Affordance Predictions — Frame {frame_id}", fontsize=16, fontweight="bold", y=1.01)

    best_global_loss = float("inf")
    best_global_file = pred_files[0]
    for obj_idx, pred_file in enumerate(pred_files[:n_show]):
        pred = np.load(os.path.join(dataset_path, "prediction", pred_file), allow_pickle=True)

        row = obj_idx
        n_cols = 3

        # Panel A: Contact heatmap (mapped from patch space back to full image)
        ax = fig.add_subplot(n_show, n_cols, row * n_cols + 1)
        ax.imshow(color_rgb, alpha=0.5)
        if "contact_scores" in pred and "bbox" in pred:
            hmap = pred["contact_scores"]
            if hmap.ndim == 3:
                hmap = hmap[0]  # unbatch -> (256, 256)
            bbox = pred["bbox"]
            if bbox.ndim == 2:
                bbox = bbox[0]  # -> (4,)
            # bbox is in cropped 256x448 space; use bbox_raw_all for full-res
            bbox_raw = pred.get("bbox_raw_all", meta.get("bbox_raw_all", None))
            if bbox_raw is not None:
                if bbox_raw.ndim == 3:
                    bbox_raw = bbox_raw[0]
                obj_bbox = bbox_raw[obj_idx]  # full-res coords
                x1, y1, x2, y2 = [int(v) for v in obj_bbox]
                full_hmap = np.zeros(color_rgb.shape[:2], dtype=np.float32)
                patch_resized = cv2.resize(hmap, (max(x2 - x1, 1), max(y2 - y1, 1)))
                # Clip to image bounds
                y1c, y2c = max(y1, 0), min(y2, color_rgb.shape[0])
                x1c, x2c = max(x1, 0), min(x2, color_rgb.shape[1])
                py1, py2 = y1c - y1, y2c - y1
                px1, px2 = x1c - x1, x2c - x1
                full_hmap[y1c:y2c, x1c:x2c] = patch_resized[py1:py2, px1:px2]
                ax.imshow(full_hmap, cmap="hot", alpha=0.6, vmin=0)
        # Mark contact point
        if "contact_pix" in pred:
            cpix = pred["contact_pix"].flatten()  # (2,) in 256x448 space
            # Scale to full-res
            img_h, img_w = color_rgb.shape[:2]
            cu = cpix[0] * img_w / 448.0
            cv = cpix[1] * img_h / 256.0
            ax.plot(cu, cv, "x", color="lime", markersize=12, markeredgewidth=3)
        ax.set_title(f"Obj {obj_idx}: Contact Heatmap", fontsize=12, fontweight="bold")
        ax.axis("off")

        # Panel B: Goal heatmap (in 256x448 space, resize to full image)
        ax = fig.add_subplot(n_show, n_cols, row * n_cols + 2)
        ax.imshow(color_rgb, alpha=0.5)
        if "goal_heatmap" in pred:
            gmap = pred["goal_heatmap"]
            if gmap.ndim == 3:
                gmap = gmap[0]  # unbatch -> (256, 448)
            gmap_resized = cv2.resize(gmap, (color_rgb.shape[1], color_rgb.shape[0]))
            ax.imshow(gmap_resized, cmap="cool", alpha=0.6, vmin=0)
        # Mark goal point
        if "goal_pix" in pred:
            gpix = pred["goal_pix"].flatten()  # (2,) in 256x448 space
            img_h, img_w = color_rgb.shape[:2]
            gu = gpix[0] * img_w / 448.0
            gv = gpix[1] * img_h / 256.0
            ax.plot(gu, gv, "*", color="yellow", markersize=15, markeredgewidth=2, markeredgecolor="black")
        ax.set_title(f"Obj {obj_idx}: Goal Heatmap", fontsize=12, fontweight="bold")
        ax.axis("off")

        # Panel C: 3D Trajectories
        ax = fig.add_subplot(n_show, n_cols, row * n_cols + 3, projection="3d")
        if "pred_trajectories" in pred:
            trajs = pred["pred_trajectories"]  # [B, N, H, 3] or [N, H, 3]
            if trajs.ndim == 4:
                trajs = trajs[0]
            # Color by guidance loss (lower loss = more blue/cool)
            losses = None
            if "guide_losses-total_loss" in pred:
                losses = pred["guide_losses-total_loss"]
                if losses.ndim == 2:
                    losses = losses[0]  # unbatch -> (N_traj,)

            # Track global best across all objects
            if losses is not None and losses.min() < best_global_loss:
                best_global_loss = losses.min()
                best_global_file = pred_file

            cmap = plt.cm.turbo
            n_trajs = min(len(trajs), 20)
            # Sort by loss so best trajectories draw on top
            if losses is not None:
                order = np.argsort(losses)[::-1]  # worst first, best on top
            else:
                order = np.arange(n_trajs)
            for rank, ti in enumerate(order[:n_trajs]):
                traj = trajs[ti]  # [H, 3]
                if losses is not None:
                    # Normalize loss for coloring
                    lo, hi = losses.min(), losses.max()
                    norm = (losses[ti] - lo) / (hi - lo + 1e-8)
                    color = cmap(norm)  # low loss = blue, high = red
                    alpha = 0.3 + 0.5 * (1 - norm)
                else:
                    color = cmap(rank / n_trajs)
                    alpha = 0.5
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=alpha, linewidth=0.8)
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c="green", s=15, marker="o", zorder=5)

            # Draw camera coordinate frame at origin
            draw_coord_frame(ax, [0, 0, 0], scale=0.05)

            # Draw grasp pose coordinate frame if available
            if "grasp_pose" in pred:
                grasp = pred["grasp_pose"]
                if grasp.ndim == 3:
                    grasp = grasp[0]  # unbatch
                if grasp.shape == (4, 4):
                    grasp_pos = grasp[:3, 3]
                    grasp_rot = grasp[:3, :3]
                    draw_coord_frame(ax, grasp_pos, grasp_rot, scale=0.04, labels=True)
                    ax.scatter(*grasp_pos, c="magenta", s=80, marker="*", zorder=10, label="Grasp pose")

            ax.set_xlabel("X (m)", fontsize=8)
            ax.set_ylabel("Y (m)", fontsize=8)
            ax.set_zlabel("Z (m)", fontsize=8)
            ax.view_init(elev=-70, azim=-90)

        ax.set_title(f"Obj {obj_idx}: Predicted Trajectories ({len(trajs)} samples)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path2 = os.path.join(out_dir, f"predictions_{frame_id:06d}.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path2}")

    # ---- Figure 3: Per-object trajectory detail ----
    intr = load_intrinsic(dataset_path)
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

    for obj_idx, pred_file in enumerate(pred_files):
        pred = np.load(os.path.join(dataset_path, "prediction", pred_file), allow_pickle=True)
        if "pred_trajectories" not in pred:
            continue

        trajs = pred["pred_trajectories"]
        if trajs.ndim == 4:
            trajs = trajs[0]

        # Find best trajectory
        best_idx = 0
        best_loss = None
        if "guide_losses-total_loss" in pred:
            total_loss = pred["guide_losses-total_loss"]
            if total_loss.ndim == 2:
                total_loss = total_loss[0]
            best_idx = int(np.argmin(total_loss))
            best_loss = total_loss[best_idx]
        best_traj = trajs[best_idx]

        loss_str = f", loss={best_loss:.4f}" if best_loss is not None else ""
        print(f"  Object {obj_idx}: best traj #{best_idx}{loss_str}")

        fig = plt.figure(figsize=(16, 12), facecolor="white")
        fig.suptitle(f"Trajectory Detail — Frame {frame_id}, Object {obj_idx} (traj #{best_idx}{loss_str})", fontsize=16, fontweight="bold")

        # 3D view
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        for ti in range(min(len(trajs), 40)):
            ax.plot(trajs[ti, :, 0], trajs[ti, :, 1], trajs[ti, :, 2], color="gray", alpha=0.1, linewidth=0.5)
        colors = plt.cm.plasma(np.linspace(0, 1, len(best_traj)))
        for hi in range(len(best_traj) - 1):
            ax.plot(best_traj[hi : hi + 2, 0], best_traj[hi : hi + 2, 1], best_traj[hi : hi + 2, 2], color=colors[hi], linewidth=3)
        ax.scatter(*best_traj[0], c="lime", s=100, marker="o", zorder=10, label="Start")
        ax.scatter(*best_traj[-1], c="red", s=100, marker="X", zorder=10, label="End")

        draw_coord_frame(ax, [0, 0, 0], scale=0.05)
        ax.text(0.06, 0, 0, "Camera", fontsize=8, color="black")

        if "grasp_pose" in pred:
            grasp = pred["grasp_pose"]
            if grasp.ndim == 3:
                grasp = grasp[0]
            if grasp.shape == (4, 4):
                draw_coord_frame(ax, grasp[:3, 3], grasp[:3, :3], scale=0.04)
                ax.text(grasp[0, 3] + 0.02, grasp[1, 3], grasp[2, 3], "Grasp", fontsize=8, color="magenta")
                for hi in range(0, len(best_traj), max(1, len(best_traj) // 5)):
                    draw_coord_frame(ax, best_traj[hi], grasp[:3, :3], scale=0.015, labels=False)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(fontsize=10)
        ax.view_init(elev=-60, azim=-90)
        ax.set_title("3D View — Best Trajectory + Grasp Frames", fontsize=12, fontweight="bold")

        # 2D projection
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(color_rgb)

        for ti in range(min(len(trajs), 20)):
            t = trajs[ti]
            z = t[:, 2]
            valid = z > 0.01
            u = t[valid, 0] * fx / z[valid] + cx
            v = t[valid, 1] * fy / z[valid] + cy
            ax2.plot(u, v, color="cyan", alpha=0.15, linewidth=0.5)

        z = best_traj[:, 2]
        valid = z > 0.01
        u = best_traj[valid, 0] * fx / z[valid] + cx
        v = best_traj[valid, 1] * fy / z[valid] + cy
        colors_2d = plt.cm.plasma(np.linspace(0, 1, len(u)))
        ax2.scatter(u, v, c=colors_2d, s=3, zorder=5)
        if len(u) > 0:
            ax2.scatter(u[0], v[0], c="lime", s=80, marker="o", zorder=10, label="Start")
            ax2.scatter(u[-1], v[-1], c="red", s=80, marker="X", zorder=10, label="End")

        ax2.set_xlim(0, color_rgb.shape[1])
        ax2.set_ylim(color_rgb.shape[0], 0)
        ax2.legend(fontsize=10)
        ax2.set_title("2D Projection onto Image", fontsize=12, fontweight="bold")
        ax2.axis("off")

        plt.tight_layout()
        path3 = os.path.join(out_dir, f"trajectory_{frame_id:06d}_obj{obj_idx}.png")
        plt.savefig(path3, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path3}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-f", "--frame", type=int, default=0)
    parser.add_argument("--dataset_dir", default="./datasets")
    args = parser.parse_args()
    dataset_path = os.path.join(args.dataset_dir, args.dataset)
    visualize_results(dataset_path, args.frame)


if __name__ == "__main__":
    main()
