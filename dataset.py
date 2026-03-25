#!/usr/bin/env python3
"""Convert Label Studio layout annotations to PaddleDetection-style COCO JSON.

This script supports three workflows:
1) Convert Label Studio export to one COCO JSON file.
2) Convert and split into train/val JSON files.
3) Build a dataset folder with:
	- annotations/train.json
	- annotations/val.json
	- imgs/ with copied and split-prefixed image names.

Input expectations:
- Label Studio tasks where geometry is in ``rectanglelabels`` results.
- Optional transcription in ``textarea`` results sharing the same result ``id``.

When source image folders contain more files than Label Studio annotations,
``--include-unannotated-images`` can include missing images as zero-annotation
entries before split/export.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass
class ConversionConfig:
	include_background_category: bool
	include_transcription: bool
	image_prefix: str


def _as_float(value: Any, default: float = 0.0) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return default


def _extract_file_name(task_data: dict[str, Any], image_id: int) -> str:
	"""Build a stable image file name from Label Studio task data."""
	ocr_url = task_data.get("ocr")
	page = task_data.get("page")

	if isinstance(ocr_url, str) and ocr_url:
		parsed = urlparse(ocr_url)
		parts = [part for part in parsed.path.split("/") if part]
		# Gallica URLs typically contain a page token like "f22" before "full".
		page_token = next((part for part in parts if part.startswith("f") and part[1:].isdigit()), None)
		if page_token:
			return f"{page_token}.jpg"

	if isinstance(page, int):
		return f"page_{page:04d}.jpg"

	return f"image_{image_id:06d}.jpg"


def _collect_transcriptions(results: list[dict[str, Any]]) -> dict[str, str]:
	transcriptions: dict[str, str] = {}
	for item in results:
		if item.get("type") != "textarea":
			continue

		result_id = item.get("id")
		value = item.get("value") or {}
		text = value.get("text")

		if not isinstance(result_id, str):
			continue
		if not isinstance(text, list) or not text:
			continue

		joined = "\n".join(str(t) for t in text if isinstance(t, str))
		if joined:
			transcriptions[result_id] = joined
	return transcriptions


def convert_label_studio_to_coco(
	label_studio_data: list[dict[str, Any]],
	config: ConversionConfig,
) -> dict[str, Any]:
	"""Convert Label Studio tasks to COCO/PaddleDetection annotations."""
	categories_by_name: OrderedDict[str, int] = OrderedDict()
	images: list[dict[str, Any]] = []
	annotations: list[dict[str, Any]] = []

	annotation_id = 0

	for image_id, task in enumerate(label_studio_data):
		if not isinstance(task, dict):
			continue

		data = task.get("data")
		predictions = task.get("predictions")
		if not isinstance(data, dict) or not isinstance(predictions, list) or not predictions:
			continue

		prediction = predictions[0] if isinstance(predictions[0], dict) else None
		if not prediction:
			continue

		results = prediction.get("result")
		if not isinstance(results, list):
			continue

		transcriptions = _collect_transcriptions(results)

		file_name = _extract_file_name(data, image_id)
		if config.image_prefix:
			file_name = f"{config.image_prefix}/{file_name}"

		# Width and height should be consistent for all regions in one page.
		first_geometry = next(
			(
				r
				for r in results
				if isinstance(r, dict)
				and r.get("type") == "rectanglelabels"
				and isinstance(r.get("value"), dict)
				and isinstance(r.get("value", {}).get("labels"), list)
				and r.get("value", {}).get("labels")
			),
			None,
		)

		if first_geometry is None:
			continue

		original_width = int(_as_float(first_geometry.get("original_width"), 0.0))
		original_height = int(_as_float(first_geometry.get("original_height"), 0.0))
		if original_width <= 0 or original_height <= 0:
			continue

		images.append(
			{
				"id": image_id,
				"file_name": file_name,
				"width": original_width,
				"height": original_height,
			}
		)

		for result in results:
			if not isinstance(result, dict) or result.get("type") != "rectanglelabels":
				continue

			value = result.get("value")
			if not isinstance(value, dict):
				continue

			labels = value.get("labels")
			if not isinstance(labels, list) or not labels:
				continue

			label_name = str(labels[0])
			if label_name not in categories_by_name:
				categories_by_name[label_name] = len(categories_by_name) + 1

			x_pct = _as_float(value.get("x"))
			y_pct = _as_float(value.get("y"))
			w_pct = _as_float(value.get("width"))
			h_pct = _as_float(value.get("height"))

			x = (x_pct / 100.0) * original_width
			y = (y_pct / 100.0) * original_height
			w = (w_pct / 100.0) * original_width
			h = (h_pct / 100.0) * original_height

			if w <= 0.0 or h <= 0.0:
				continue

			x2 = x + w
			y2 = y + h
			annotation: dict[str, Any] = {
				"id": annotation_id,
				"image_id": image_id,
				"category_id": categories_by_name[label_name],
				"bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
				"area": round(w * h, 3),
				"segmentation": [
					[
						round(x, 3),
						round(y, 3),
						round(x, 3),
						round(y2, 3),
						round(x2, 3),
						round(y2, 3),
						round(x2, 3),
						round(y, 3),
					]
				],
				"iscrowd": 0,
			}

			if config.include_transcription:
				result_id = result.get("id")
				if isinstance(result_id, str) and result_id in transcriptions:
					annotation["transcription"] = transcriptions[result_id]

			annotations.append(annotation)
			annotation_id += 1

	categories: list[dict[str, Any]] = []
	if config.include_background_category:
		categories.append({"id": 0, "name": "_background_", "supercategory": None})

	categories.extend(
		{
			"id": category_id,
			"name": category_name,
			"supercategory": None,
		}
		for category_name, category_id in categories_by_name.items()
	)

	return {
		"info": {
			"description": "Converted from Label Studio",
			"version": "1.0",
			"year": datetime.now().year,
			"date_created": datetime.now().isoformat(timespec="seconds"),
		},
		"licenses": [{"id": 0, "name": None, "url": None}],
		"images": images,
		"annotations": annotations,
		"categories": categories,
		"type": "instances",
	}


def split_coco_by_images(
	coco_data: dict[str, Any],
	val_ratio: float,
	seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
	"""Split a COCO dictionary into train and val sets using image ids."""
	images = coco_data.get("images", [])
	annotations = coco_data.get("annotations", [])

	if not isinstance(images, list) or not isinstance(annotations, list):
		raise ValueError("Invalid COCO data: 'images' and 'annotations' must be lists.")

	if not 0.0 < val_ratio < 1.0:
		raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

	image_ids = [img["id"] for img in images if isinstance(img, dict) and "id" in img]
	if len(image_ids) < 2:
		raise ValueError("Need at least 2 images to create train/val split.")

	rng = random.Random(seed)
	rng.shuffle(image_ids)

	val_count = int(round(len(image_ids) * val_ratio))
	val_count = max(1, min(val_count, len(image_ids) - 1))

	val_ids = set(image_ids[:val_count])
	train_ids = set(image_ids[val_count:])

	train_images = [img for img in images if isinstance(img, dict) and img.get("id") in train_ids]
	val_images = [img for img in images if isinstance(img, dict) and img.get("id") in val_ids]

	train_annotations = [
		ann for ann in annotations if isinstance(ann, dict) and ann.get("image_id") in train_ids
	]
	val_annotations = [
		ann for ann in annotations if isinstance(ann, dict) and ann.get("image_id") in val_ids
	]

	base_info = dict(coco_data.get("info", {}))
	base_licenses = coco_data.get("licenses", [])
	base_categories = coco_data.get("categories", [])
	base_type = coco_data.get("type", "instances")

	train_info = dict(base_info)
	train_info["split"] = "train"
	val_info = dict(base_info)
	val_info["split"] = "val"

	train_coco = {
		"info": train_info,
		"licenses": base_licenses,
		"images": train_images,
		"annotations": train_annotations,
		"categories": base_categories,
		"type": base_type,
	}

	val_coco = {
		"info": val_info,
		"licenses": base_licenses,
		"images": val_images,
		"annotations": val_annotations,
		"categories": base_categories,
		"type": base_type,
	}

	return train_coco, val_coco


def filter_coco_by_images_dir(
	coco_data: dict[str, Any],
	images_dir: Path,
) -> dict[str, Any]:
	"""Keep only images present in a local folder and their matching annotations."""
	images = coco_data.get("images", [])
	annotations = coco_data.get("annotations", [])

	if not isinstance(images, list) or not isinstance(annotations, list):
		raise ValueError("Invalid COCO data: 'images' and 'annotations' must be lists.")

	if not images_dir.exists() or not images_dir.is_dir():
		raise ValueError(f"Images directory does not exist or is not a directory: {images_dir}")

	available_names = {
		path.name
		for path in images_dir.iterdir()
		if path.is_file()
	}

	filtered_images = [
		img
		for img in images
		if isinstance(img, dict)
		and "id" in img
		and isinstance(img.get("file_name"), str)
		and Path(img["file_name"]).name in available_names
	]
	filtered_image_ids = {img["id"] for img in filtered_images}

	filtered_annotations = [
		ann
		for ann in annotations
		if isinstance(ann, dict) and ann.get("image_id") in filtered_image_ids
	]

	return {
		"info": dict(coco_data.get("info", {})),
		"licenses": coco_data.get("licenses", []),
		"images": filtered_images,
		"annotations": filtered_annotations,
		"categories": coco_data.get("categories", []),
		"type": coco_data.get("type", "instances"),
	}


def _read_image_size(image_path: Path) -> tuple[int, int]:
	"""Read image dimensions using Pillow only when needed."""
	try:
		from PIL import Image
	except ImportError as exc:
		raise ValueError(
			"Pillow is required to include unannotated images. Install it with: pip install Pillow"
		) from exc

	with Image.open(image_path) as image:
		width, height = image.size

	if width <= 0 or height <= 0:
		raise ValueError(f"Invalid image size for {image_path}: {width}x{height}")

	return int(width), int(height)


def include_unannotated_images_from_dir(
	coco_data: dict[str, Any],
	images_dir: Path,
) -> tuple[dict[str, Any], int]:
	"""Add images present on disk but missing from Label Studio as empty-annotation images."""
	images = coco_data.get("images", [])
	annotations = coco_data.get("annotations", [])

	if not isinstance(images, list) or not isinstance(annotations, list):
		raise ValueError("Invalid COCO data: 'images' and 'annotations' must be lists.")

	if not images_dir.exists() or not images_dir.is_dir():
		raise ValueError(f"Images directory does not exist or is not a directory: {images_dir}")

	existing_names = {
		Path(img["file_name"]).name
		for img in images
		if isinstance(img, dict) and isinstance(img.get("file_name"), str)
	}
	next_image_id = max(
		(
			int(img["id"])
			for img in images
			if isinstance(img, dict) and isinstance(img.get("id"), int)
		),
		default=-1,
	) + 1

	updated_images = list(images)
	added_count = 0

	for source_image in sorted(path for path in images_dir.iterdir() if path.is_file()):
		if source_image.name in existing_names:
			continue

		width, height = _read_image_size(source_image)
		updated_images.append(
			{
				"id": next_image_id,
				"file_name": source_image.name,
				"width": width,
				"height": height,
			}
		)
		next_image_id += 1
		added_count += 1

	updated_coco = {
		"info": dict(coco_data.get("info", {})),
		"licenses": coco_data.get("licenses", []),
		"images": updated_images,
		"annotations": annotations,
		"categories": coco_data.get("categories", []),
		"type": coco_data.get("type", "instances"),
	}

	return updated_coco, added_count


def copy_images_for_split_and_rewrite(
	coco_data: dict[str, Any],
	split_name: str,
	source_images_dir: Path,
	target_images_dir: Path,
) -> tuple[dict[str, Any], int]:
	"""Copy split images to target folder and rename file names with split prefix."""
	images = coco_data.get("images", [])
	annotations = coco_data.get("annotations", [])

	if not isinstance(images, list) or not isinstance(annotations, list):
		raise ValueError("Invalid COCO data: 'images' and 'annotations' must be lists.")

	if not source_images_dir.exists() or not source_images_dir.is_dir():
		raise ValueError(f"Images directory does not exist or is not a directory: {source_images_dir}")

	target_images_dir.mkdir(parents=True, exist_ok=True)

	updated_images: list[dict[str, Any]] = []
	missing_names: list[str] = []
	copied_index = 1

	for image in images:
		if not isinstance(image, dict):
			continue

		file_name = image.get("file_name")
		image_id = image.get("id")
		if not isinstance(file_name, str) or image_id is None:
			continue

		source_name = Path(file_name).name
		source_path = source_images_dir / source_name
		if not source_path.exists() or not source_path.is_file():
			missing_names.append(source_name)
			continue

		extension = source_path.suffix or ".jpg"
		new_name = f"{split_name}_{copied_index:06d}{extension.lower()}"
		copied_index += 1

		shutil.copy2(source_path, target_images_dir / new_name)

		updated_image = dict(image)
		updated_image["file_name"] = new_name
		updated_images.append(updated_image)

	if missing_names:
		preview = ", ".join(missing_names[:5])
		suffix = "" if len(missing_names) <= 5 else f" (+{len(missing_names) - 5} more)"
		raise ValueError(
			f"{len(missing_names)} images from split '{split_name}' were not found in {source_images_dir}: "
			f"{preview}{suffix}"
		)

	kept_ids = {
		img["id"]
		for img in updated_images
		if isinstance(img, dict) and "id" in img
	}
	updated_annotations = [
		ann
		for ann in annotations
		if isinstance(ann, dict) and ann.get("image_id") in kept_ids
	]

	updated_coco = {
		"info": dict(coco_data.get("info", {})),
		"licenses": coco_data.get("licenses", []),
		"images": updated_images,
		"annotations": updated_annotations,
		"categories": coco_data.get("categories", []),
		"type": coco_data.get("type", "instances"),
	}

	return updated_coco, len(updated_images)


def write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as fp:
		json.dump(payload, fp, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert Label Studio layout export to PaddleDetection COCO annotations.",
		epilog=(
			"Examples:\n"
			"  Convert: --input labels.json --output full.json\n"
			"  Split: --input labels.json --output full.json --split-train-val\n"
			"  Dataset: --input labels.json --images-dir imgs --dataset-dir dataset_out"
		),
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("--input", required=True, help="Path to Label Studio JSON export.")
	parser.add_argument(
		"--output",
		help="Path to output COCO JSON file (required unless using --split-train-val with explicit split outputs, or --dataset-dir).",
	)
	parser.add_argument(
		"--image-prefix",
		default="",
		help="Optional prefix added to image file names (example: JPEGImages).",
	)
	parser.add_argument(
		"--no-background",
		action="store_true",
		help="Do not include COCO category id=0 named _background_.",
	)
	parser.add_argument(
		"--include-transcription",
		action="store_true",
		help="Attach textarea text to annotations under the key 'transcription'.",
	)
	parser.add_argument(
		"--split-train-val",
		action="store_true",
		help="Create train/val COCO files by splitting images.",
	)
	parser.add_argument(
		"--val-ratio",
		type=float,
		default=0.2,
		help="Validation split ratio used with --split-train-val (default: 0.2).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for train/val split (default: 42).",
	)
	parser.add_argument(
		"--train-output",
		help="Path for train split COCO JSON. Optional when --output is provided.",
	)
	parser.add_argument(
		"--val-output",
		help="Path for val split COCO JSON. Optional when --output is provided.",
	)
	parser.add_argument(
		"--images-dir",
		help=(
			"Optional path to an images folder. Keep only annotations for images present in this folder "
			"(matching by file name). Required by --dataset-dir and --include-unannotated-images."
		),
	)
	parser.add_argument(
		"--include-unannotated-images",
		action="store_true",
		help=(
			"Include source-folder images that are not present in Label Studio as empty-annotation images. "
			"Requires --images-dir. Uses Pillow to read image size."
		),
	)
	parser.add_argument(
		"--dataset-dir",
		help=(
			"Create dataset outputs in this folder: annotations/train.json, annotations/val.json, "
			"and copied imgs/ with split-prefixed file names."
		),
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_path = Path(args.input)

	with input_path.open("r", encoding="utf-8") as fp:
		payload = json.load(fp)

	if not isinstance(payload, list):
		raise ValueError("Expected Label Studio export to be a JSON list of tasks.")

	config = ConversionConfig(
		include_background_category=not args.no_background,
		include_transcription=args.include_transcription,
		image_prefix=args.image_prefix.strip("/"),
	)
	coco = convert_label_studio_to_coco(payload, config)

	if args.images_dir:
		coco = filter_coco_by_images_dir(coco, Path(args.images_dir))

	added_unannotated = 0
	if args.include_unannotated_images:
		if not args.images_dir:
			raise ValueError("--include-unannotated-images requires --images-dir.")
		coco, added_unannotated = include_unannotated_images_from_dir(coco, Path(args.images_dir))

	if args.dataset_dir:
		if not args.images_dir:
			raise ValueError("--dataset-dir requires --images-dir to copy source images.")

		train_coco, val_coco = split_coco_by_images(coco, val_ratio=args.val_ratio, seed=args.seed)

		dataset_dir = Path(args.dataset_dir)
		annotations_dir = dataset_dir / "annotations"
		target_imgs_dir = dataset_dir / "imgs"

		train_coco, train_copied = copy_images_for_split_and_rewrite(
			train_coco,
			split_name="train",
			source_images_dir=Path(args.images_dir),
			target_images_dir=target_imgs_dir,
		)
		val_coco, val_copied = copy_images_for_split_and_rewrite(
			val_coco,
			split_name="val",
			source_images_dir=Path(args.images_dir),
			target_images_dir=target_imgs_dir,
		)

		train_output_path = annotations_dir / "train.json"
		val_output_path = annotations_dir / "val.json"
		write_json(train_output_path, train_coco)
		write_json(val_output_path, val_coco)

		if args.output:
			write_json(Path(args.output), coco)

		message = (
			f"Built dataset in {dataset_dir}. "
			f"Train: {train_copied} images, {len(train_coco['annotations'])} annotations -> {train_output_path}. "
			f"Val: {val_copied} images, {len(val_coco['annotations'])} annotations -> {val_output_path}."
		)
		if added_unannotated > 0:
			message += f" Included {added_unannotated} unannotated images from {args.images_dir}."
		if args.output:
			message += f" Wrote full annotations to {args.output}."

		print(message)
		return

	if args.split_train_val:
		train_coco, val_coco = split_coco_by_images(coco, val_ratio=args.val_ratio, seed=args.seed)

		default_base = Path(args.output) if args.output else None
		full_output_path = default_base
		train_output_path = Path(args.train_output) if args.train_output else None
		val_output_path = Path(args.val_output) if args.val_output else None

		if default_base and not train_output_path:
			train_output_path = default_base.with_name(f"{default_base.stem}_train{default_base.suffix or '.json'}")
		if default_base and not val_output_path:
			val_output_path = default_base.with_name(f"{default_base.stem}_val{default_base.suffix or '.json'}")

		if train_output_path is None or val_output_path is None:
			raise ValueError(
				"For --split-train-val, provide --output or both --train-output and --val-output."
			)

		if full_output_path is not None:
			write_json(full_output_path, coco)

		write_json(train_output_path, train_coco)

		write_json(val_output_path, val_coco)

		if full_output_path is not None:
			print(
				f"Converted {len(coco['images'])} images and {len(coco['annotations'])} annotations. "
				f"Wrote full annotations to {full_output_path}, "
				f"train split ({len(train_coco['images'])} images) to {train_output_path}, and "
				f"val split ({len(val_coco['images'])} images) to {val_output_path}."
			)
		else:
			print(
				f"Converted {len(coco['images'])} images and {len(coco['annotations'])} annotations. "
				f"Wrote train split ({len(train_coco['images'])} images) to {train_output_path} and "
				f"val split ({len(val_coco['images'])} images) to {val_output_path}."
			)
		return

	if not args.output:
		raise ValueError("--output is required when --split-train-val is not used.")

	output_path = Path(args.output)

	write_json(output_path, coco)

	print(
		f"Converted {len(coco['images'])} images and {len(coco['annotations'])} annotations "
		f"into {output_path}"
	)


if __name__ == "__main__":
	main()
