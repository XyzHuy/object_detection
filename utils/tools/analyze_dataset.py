#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def quantiles(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    qs = np.percentile(arr, [0, 5, 25, 50, 75, 95, 100])
    keys = ["min", "p05", "p25", "p50", "p75", "p95", "max"]
    return {key: float(value) for key, value in zip(keys, qs)}


def bucket_area(area_frac):
    if area_frac < 0.01:
        return "tiny_<1%"
    if area_frac < 0.05:
        return "small_1-5%"
    if area_frac < 0.20:
        return "medium_5-20%"
    return "large_>=20%"


def bucket_shape(aspect_ratio):
    if aspect_ratio < 0.5:
        return "tall_<0.5"
    if aspect_ratio < 0.8:
        return "portrait_0.5-0.8"
    if aspect_ratio <= 1.25:
        return "square_0.8-1.25"
    if aspect_ratio <= 2.0:
        return "landscape_1.25-2"
    return "wide_>2"


def load_split(data_root, split):
    path = data_root / "annotations" / f"{split}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_split(data_root, split):
    data = load_split(data_root, split)
    classes = data["classes"]
    images = data["images"]
    image_by_id = {img["id"]: img for img in images}

    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    class_objects = Counter()
    class_images = Counter()
    class_area_bucket = {cls: Counter() for cls in classes}
    class_shape_bucket = {cls: Counter() for cls in classes}
    invalid_boxes = []

    image_widths = []
    image_heights = []
    image_aspects = []
    boxes_per_image = []
    box_widths = []
    box_heights = []
    box_area_fracs = []
    box_aspects = []
    box_area_fracs_by_class = defaultdict(list)
    box_aspects_by_class = defaultdict(list)

    for image in images:
        width = image["width"]
        height = image["height"]
        image_widths.append(width)
        image_heights.append(height)
        image_aspects.append(width / height)
        image_anns = anns_by_image.get(image["id"], [])
        boxes_per_image.append(len(image_anns))

        seen_classes = set()
        for ann in image_anns:
            cls = ann["class"]
            x1, y1, x2, y2 = ann["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                invalid_boxes.append({"image_id": image["id"], "class": cls, "bbox": ann["bbox"]})
                continue

            area_frac = (bw * bh) / (width * height)
            aspect = bw / bh

            class_objects[cls] += 1
            seen_classes.add(cls)
            class_area_bucket[cls][bucket_area(area_frac)] += 1
            class_shape_bucket[cls][bucket_shape(aspect)] += 1
            box_widths.append(bw)
            box_heights.append(bh)
            box_area_fracs.append(area_frac)
            box_aspects.append(aspect)
            box_area_fracs_by_class[cls].append(area_frac)
            box_aspects_by_class[cls].append(aspect)

        for cls in seen_classes:
            class_images[cls] += 1

    total_objects = sum(class_objects.values())
    max_count = max(class_objects.values()) if class_objects else 0
    min_count = min(class_objects.values()) if class_objects else 0

    class_rows = []
    for cls in classes:
        count = class_objects[cls]
        image_count = class_images[cls]
        class_rows.append(
            {
                "split": split,
                "class": cls,
                "objects": count,
                "object_pct": 100 * count / total_objects if total_objects else 0,
                "images": image_count,
                "image_pct": 100 * image_count / len(images) if images else 0,
                "median_area_pct": 100 * np.median(box_area_fracs_by_class[cls])
                if box_area_fracs_by_class[cls]
                else 0,
                "median_aspect_w_h": np.median(box_aspects_by_class[cls])
                if box_aspects_by_class[cls]
                else 0,
            }
        )

    return {
        "split": split,
        "classes": classes,
        "num_images": len(images),
        "num_annotations": len(data["annotations"]),
        "valid_boxes": total_objects,
        "invalid_boxes": invalid_boxes,
        "imbalance_ratio_max_min": max_count / min_count if min_count else math.inf,
        "class_rows": class_rows,
        "image_stats": {
            "width": quantiles(image_widths),
            "height": quantiles(image_heights),
            "aspect_w_h": quantiles(image_aspects),
            "boxes_per_image": quantiles(boxes_per_image),
            "empty_images": sum(1 for n in boxes_per_image if n == 0),
        },
        "box_stats": {
            "width_px": quantiles(box_widths),
            "height_px": quantiles(box_heights),
            "area_pct": {k: 100 * v for k, v in quantiles(box_area_fracs).items()},
            "aspect_w_h": quantiles(box_aspects),
        },
        "area_buckets": {
            cls: dict(class_area_bucket[cls])
            for cls in classes
        },
        "shape_buckets": {
            cls: dict(class_shape_bucket[cls])
            for cls in classes
        },
        "plot_values": {
            "box_area_pct": [100 * v for v in box_area_fracs],
            "box_aspect_w_h": box_aspects,
            "image_aspect_w_h": image_aspects,
            "boxes_per_image": boxes_per_image,
            "class_objects": [class_objects[cls] for cls in classes],
        },
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summaries(out_dir, summaries):
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    import matplotlib.pyplot as plt

    for summary in summaries:
        split = summary["split"]
        values = summary["plot_values"]
        classes = summary["classes"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.ravel()

        axes[0].bar(classes, values["class_objects"], color="#3366aa")
        axes[0].set_title(f"{split}: objects per class")
        axes[0].set_ylabel("objects")
        axes[0].tick_params(axis="x", rotation=30)

        axes[1].hist(values["box_area_pct"], bins=40, color="#55a868")
        axes[1].set_title(f"{split}: bbox area / image area")
        axes[1].set_xlabel("area (%)")
        axes[1].set_ylabel("boxes")

        clipped_aspects = [min(v, 6.0) for v in values["box_aspect_w_h"]]
        axes[2].hist(clipped_aspects, bins=40, color="#c44e52")
        axes[2].set_title(f"{split}: bbox aspect ratio w/h (clipped at 6)")
        axes[2].set_xlabel("w / h")
        axes[2].set_ylabel("boxes")

        axes[3].hist(values["boxes_per_image"], bins=range(0, max(values["boxes_per_image"]) + 2), color="#8172b3")
        axes[3].set_title(f"{split}: boxes per image")
        axes[3].set_xlabel("boxes")
        axes[3].set_ylabel("images")

        fig.tight_layout()
        fig.savefig(out_dir / f"{split}_dataset_distributions.png", dpi=160)
        plt.close(fig)


def print_summary(summary):
    print(f"\n=== {summary['split'].upper()} ===")
    print(f"images: {summary['num_images']}")
    print(f"annotations: {summary['num_annotations']}")
    print(f"valid boxes: {summary['valid_boxes']}")
    print(f"invalid boxes: {len(summary['invalid_boxes'])}")
    print(f"class imbalance max/min objects: {summary['imbalance_ratio_max_min']:.2f}x")
    print("\nclass distribution:")
    for row in summary["class_rows"]:
        print(
            f"  {row['class']:>6}: objects={row['objects']:5d} "
            f"({row['object_pct']:5.1f}%), images={row['images']:4d} "
            f"({row['image_pct']:5.1f}%), median_area={row['median_area_pct']:5.2f}%, "
            f"median_aspect={row['median_aspect_w_h']:4.2f}"
        )
    print("\nbox area % quantiles:", summary["box_stats"]["area_pct"])
    print("box aspect w/h quantiles:", summary["box_stats"]["aspect_w_h"])
    print("boxes/image quantiles:", summary["image_stats"]["boxes_per_image"])
    print("image aspect w/h quantiles:", summary["image_stats"]["aspect_w_h"])
    print(f"empty images: {summary['image_stats']['empty_images']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze object detection dataset annotations.")
    parser.add_argument("--data-root", default="final_public/public", type=Path)
    parser.add_argument("--out-dir", default="dataset_stats", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [summarize_split(args.data_root, split) for split in args.splits]

    all_class_rows = []
    report = {}
    for summary in summaries:
        print_summary(summary)
        all_class_rows.extend(summary["class_rows"])
        report[summary["split"]] = {
            key: value
            for key, value in summary.items()
            if key != "plot_values"
        }

        write_csv(
            args.out_dir / f"{summary['split']}_area_buckets.csv",
            [
                {"split": summary["split"], "class": cls, **summary["area_buckets"][cls]}
                for cls in summary["classes"]
            ],
        )
        write_csv(
            args.out_dir / f"{summary['split']}_shape_buckets.csv",
            [
                {"split": summary["split"], "class": cls, **summary["shape_buckets"][cls]}
                for cls in summary["classes"]
            ],
        )

    write_csv(args.out_dir / "class_distribution.csv", all_class_rows)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if not args.no_plots:
        plot_summaries(args.out_dir, summaries)

    print(f"\nWrote report files to: {args.out_dir}")


if __name__ == "__main__":
    main()
