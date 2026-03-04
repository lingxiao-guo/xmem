#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file.")
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]

            name_bytes = bytearray()
            while True:
                ch = fid.read(1)
                if not ch:
                    raise EOFError("Unexpected EOF while reading image name.")
                if ch == b"\x00":
                    break
                name_bytes.extend(ch)
            image_name = name_bytes.decode("utf-8", errors="replace")

            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2d, os.SEEK_CUR)  # x, y, point3D_id per feature

            images[image_id] = ColmapImage(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
            )
    return images


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def compute_camera_center_and_forward(qvec, tvec):
    # COLMAP: x_cam = R * x_world + t, so camera center is -R^T * t.
    R = qvec2rotmat(qvec)
    center = -R.T @ tvec
    forward = R.T @ np.array([0.0, 0.0, 1.0])
    norm = np.linalg.norm(forward)
    if norm > 0:
        forward = forward / norm
    return center, forward


def farthest_point_sampling(features, k):
    n = features.shape[0]
    if k >= n:
        return list(range(n))
    centroid = features.mean(axis=0)
    min_dists = np.linalg.norm(features - centroid, axis=1)
    first = int(np.argmax(min_dists))
    selected = [first]
    min_dists = np.linalg.norm(features - features[first], axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(min_dists))
        selected.append(idx)
        new_dists = np.linalg.norm(features - features[idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
    return selected


def build_features(centers, forwards, dir_weight):
    centers = np.asarray(centers, dtype=np.float64)
    forwards = np.asarray(forwards, dtype=np.float64)
    pos_scale = np.mean(np.std(centers, axis=0))
    if not np.isfinite(pos_scale) or pos_scale < 1e-9:
        pos_scale = 1.0
    centers_scaled = centers / pos_scale
    forwards_scaled = forwards * float(dir_weight)
    return np.concatenate([centers_scaled, forwards_scaled], axis=1)


def resolve_source_path(images_dir, image_name):
    full_rel = Path(image_name)
    full_path = images_dir / full_rel
    if full_path.is_file():
        return full_path, full_rel

    basename_rel = Path(full_rel.name)
    basename_path = images_dir / basename_rel
    if basename_path.is_file():
        return basename_path, basename_rel

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Select diverse keyframes from a COLMAP images.bin and copy them."
    )
    parser.add_argument(
        "--images-bin",
        required=True,
        help="Path to COLMAP images.bin (camera poses).",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing source images (e.g., images_raw).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to copy selected images into.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        required=True,
        help="Number of keyframes to select.",
    )
    parser.add_argument(
        "--dir-weight",
        type=float,
        default=0.5,
        help="Weight for view direction diversity relative to position.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selected images without copying.",
    )
    args = parser.parse_args()

    images_bin = Path(args.images_bin)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not images_bin.is_file():
        raise FileNotFoundError(f"images.bin not found: {images_bin}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if args.num_views <= 0:
        raise ValueError("--num-views must be > 0")

    images = read_images_binary(str(images_bin))
    image_items = sorted(images.values(), key=lambda img: img.name)

    candidates = []
    centers = []
    forwards = []
    missing = []
    for img in image_items:
        src_path, rel_name = resolve_source_path(images_dir, img.name)
        if src_path is None:
            missing.append(img.name)
            continue
        center, forward = compute_camera_center_and_forward(img.qvec, img.tvec)
        candidates.append((str(rel_name), src_path))
        centers.append(center)
        forwards.append(forward)

    if missing:
        print(
            f"Warning: {len(missing)} image files were missing in {images_dir}.",
            file=sys.stderr,
        )

    if not candidates:
        raise RuntimeError("No images found that match images.bin entries.")

    if args.num_views > len(candidates):
        raise ValueError(
            f"--num-views ({args.num_views}) exceeds available images ({len(candidates)})."
        )

    features = build_features(centers, forwards, args.dir_weight)
    selected_idx = farthest_point_sampling(features, args.num_views)
    selected = [candidates[i] for i in selected_idx]

    if args.dry_run:
        for name, _ in selected:
            print(name)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, src_path in selected:
        dst_path = output_dir / name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    print(f"Selected {len(selected)} images -> {output_dir}")


if __name__ == "__main__":
    main()
