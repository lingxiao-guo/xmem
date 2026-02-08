#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reorganize images/masks/depth by object id.")
    parser.add_argument(
        "--task_name",
        required=True,
        help="Task directory name under the example folder (e.g., picking_up_trash).",
    )
    parser.add_argument(
        "--base_root",
        default="/home/lingxiao/data/projects/lingxiao/XMem/example",
        help="Root directory containing task folders.",
    )
    parser.add_argument(
        "--camera_name",
        default="head",
        choices=["head", "left_wrist", "right_wrist"],
        help="Camera subfolder name under the task directory.",
    )
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Directory containing RGB images.",
    )
    parser.add_argument(
        "--masks_dir",
        default=None,
        help="Directory containing mask images.",
    )
    parser.add_argument(
        "--depth_dir",
        default=None,
        help="Directory containing depth images.",
    )
    parser.add_argument(
        "--object_json",
        default=None,
        help="Path to object.json defining start/end frame ids.",
    )
    parser.add_argument(
        "--scene_json",
        default=None,
        help="Path to scene.json defining start/end frame ids.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Output root directory.",
    )
    return parser.parse_args()


def _parse_frame_id(value) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def load_objects(json_path: str) -> Dict[int, List[Tuple[int, int]]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("objects"), list):
        entries = payload["objects"]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(f"Unexpected object.json format: {json_path}")

    objects: Dict[int, List[Tuple[int, int]]] = {}
    for entry in entries:
        try:
            obj_id = int(entry.get("object id"))
        except Exception:
            continue
        segments = []
        segments_data = entry.get("segments")
        if isinstance(segments_data, list) and segments_data:
            for segment in segments_data:
                start_idx = _parse_frame_id(segment.get("start id"))
                end_idx = _parse_frame_id(segment.get("end id"))
                if start_idx is None or end_idx is None:
                    continue
                segments.append((start_idx, end_idx))
        else:
            start_idx = _parse_frame_id(entry.get("start id"))
            end_idx = _parse_frame_id(entry.get("end id"))
            if start_idx is not None and end_idx is not None:
                segments.append((start_idx, end_idx))

        if not segments:
            continue
        objects.setdefault(obj_id, []).extend(segments)
    return objects


def load_scene_segments(json_path: str) -> List[Tuple[int, int]]:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        segments_data = payload.get("segments")
    elif isinstance(payload, list):
        segments_data = payload
    else:
        raise ValueError(f"Unexpected scene.json format: {json_path}")

    segments: List[Tuple[int, int]] = []
    if isinstance(segments_data, list):
        for segment in segments_data:
            start_idx = _parse_frame_id(segment.get("start id"))
            end_idx = _parse_frame_id(segment.get("end id"))
            if start_idx is None or end_idx is None:
                continue
            segments.append((start_idx, end_idx))
    return segments


def build_index_map(dir_path: str) -> Dict[int, str]:
    if not os.path.isdir(dir_path):
        return {}
    allowed_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    index_map: Dict[int, str] = {}
    for name in os.listdir(dir_path):
        ext = os.path.splitext(name)[1].lower()
        if ext not in allowed_exts:
            continue
        stem = os.path.splitext(name)[0]
        if not stem.isdigit():
            continue
        index_map[int(stem)] = name
    return index_map

 
def ensure_dirs(base_dir: str) -> Dict[str, str]:
    images_out = os.path.join(base_dir, "rgb")
    masks_out = os.path.join(base_dir, "masks")
    depth_out = os.path.join(base_dir, "depth")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)
    return {"images": images_out, "masks": masks_out, "depth": depth_out}


def copy_if_exists(src_dir: str, filename: str, dst_dir: str) -> bool:
    src_path = os.path.join(src_dir, filename)
    if not os.path.exists(src_path):
        return False
    shutil.copy2(src_path, os.path.join(dst_dir, filename))
    return True


def save_object_mask(
    src_path: str,
    dst_path: str,
    object_id: int,
) -> None:
    image = Image.open(src_path)
    mask = image.convert("P")
    mask_array = np.array(mask, dtype=np.uint8)
    mask_array = np.where(mask_array == object_id, 255, 0).astype(np.uint8)
    mask = Image.fromarray(mask_array, mode="L")
    mask.save(dst_path)


def main() -> None:
    args = parse_args()
    base_dir = os.path.join(args.base_root, args.task_name)
    images_dir = args.images_dir or os.path.join(base_dir, args.camera_name, "rgb")
    masks_dir = args.masks_dir or os.path.join(base_dir, args.camera_name, "masks")
    depth_dir = args.depth_dir or os.path.join(base_dir, args.camera_name, "depth")
    object_json = args.object_json or os.path.join(base_dir, args.camera_name, "object.json")
    scene_json = args.scene_json or os.path.join(base_dir, args.camera_name, "scene.json")
    output_root = args.output_root or os.path.join(
        base_dir, "outputs", args.task_name, args.camera_name
    )

    objects = load_objects(object_json)
    if not objects:
        raise RuntimeError(f"No valid objects found in {object_json}")

    images_map = build_index_map(images_dir)
    masks_map = build_index_map(masks_dir)
    depth_map = build_index_map(depth_dir)

    if not images_map:
        raise RuntimeError(f"No images found in {images_dir}")

    os.makedirs(output_root, exist_ok=True)

    for obj_id, segments in sorted(objects.items()):
        obj_dir = os.path.join(output_root, f"object_{obj_id}")
        out_dirs = ensure_dirs(obj_dir)

        index_set = set()
        for start_idx, end_idx in segments:
            start = min(start_idx, end_idx)
            end = max(start_idx, end_idx)
            if end < 0:
                continue
            start = max(start, 0)
            index_set.update(range(start, end + 1))

        indices = sorted(i for i in images_map if i in index_set)
        copied_images = copied_masks = copied_depth = 0
        missing_masks = missing_depth = 0
        for idx in indices:
            img_name = images_map[idx]
            if copy_if_exists(images_dir, img_name, out_dirs["images"]):
                copied_images += 1

            mask_name = masks_map.get(idx)
            if mask_name:
                src_mask_path = os.path.join(masks_dir, mask_name)
                dst_mask_path = os.path.join(out_dirs["masks"], mask_name)
                try:
                    save_object_mask(src_mask_path, dst_mask_path, obj_id)
                    copied_masks += 1
                except Exception:
                    missing_masks += 1
            else:
                missing_masks += 1

            depth_name = depth_map.get(idx)
            if depth_name and copy_if_exists(depth_dir, depth_name, out_dirs["depth"]):
                copied_depth += 1
            else:
                missing_depth += 1

        segment_summary = ", ".join(f"{min(s,e)}-{max(s,e)}" for s, e in segments)
        print(
            f"object_{obj_id}: segments [{segment_summary}], "
            f"images={copied_images}, masks={copied_masks} (missing {missing_masks}), "
            f"depth={copied_depth} (missing {missing_depth})"
        )

    scene_segments = load_scene_segments(scene_json)
    if scene_segments:
        navigation_dir = os.path.join(output_root, "navigation")
        os.makedirs(navigation_dir, exist_ok=True)

        scene_indices = set()
        for start_idx, end_idx in scene_segments:
            start = min(start_idx, end_idx)
            end = max(start_idx, end_idx)
            if end < 0:
                continue
            start = max(start, 0)
            scene_indices.update(range(start, end + 1))

        scene_indices = sorted(i for i in images_map if i in scene_indices)
        missing_scene_masks = 0
        for idx in scene_indices:
            rgb_name = images_map[idx]
            rgb_path = os.path.join(images_dir, rgb_name)
            rgb = Image.open(rgb_path).convert("RGB")
            rgb_array = np.array(rgb, dtype=np.uint8)

            mask_name = masks_map.get(idx)
            if mask_name:
                mask_path = os.path.join(masks_dir, mask_name)
                mask = Image.open(mask_path).convert("L")
                mask_array = np.array(mask, dtype=np.uint8)
                rgb_array[mask_array > 0] = np.array([0, 255, 0], dtype=np.uint8)
            else:
                missing_scene_masks += 1

            out_path = os.path.join(navigation_dir, rgb_name)
            Image.fromarray(rgb_array).save(out_path)

        print(
            f"navigation: segments {len(scene_segments)}, frames={len(scene_indices)}, "
            f"missing masks={missing_scene_masks}"
        )
    else:
        print(f"navigation: no scene segments found in {scene_json}")


if __name__ == "__main__":
    main()
