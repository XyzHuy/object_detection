import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, List, Tuple, Dict
import torchvision.transforms.functional as trans_func
import torchvision.transforms as trans
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
try:
    import albumentations as A
except ImportError:
    A = None


def letterbox_resize(image: Image.Image, target_size: int, target: dict) -> Tuple[Image.Image, dict]:
    """
    Resize ảnh về target_size x target_size bằng cách:
    1. Scale giữ nguyên aspect ratio (fit trong target_size)
    2. Pad hai phía (top/bottom hoặc left/right) cho đủ vuông

    Cập nhật bbox trong target theo đúng tọa độ mới.
    """
    W_orig, H_orig = image.size  # PIL: (W, H)

    scale = target_size / max(W_orig, H_orig)
    W_new = int(round(W_orig * scale))
    H_new = int(round(H_orig * scale))

    # Resize giữ aspect ratio
    image = image.resize((W_new, H_new), Image.BILINEAR)

    # Padding offset để căn giữa
    pad_left = (target_size - W_new) // 2
    pad_top  = (target_size - H_new) // 2

    # Tạo canvas xám và paste ảnh vào giữa
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    canvas.paste(image, (pad_left, pad_top))

    # Scale + shift bounding boxes
    if target["boxes"].numel() > 0:
        boxes = target["boxes"].clone().float()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_left  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_top   # y1, y2
        # Clamp vào trong ảnh
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, target_size)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, target_size)
        target["boxes"] = boxes

    # Lưu meta để có thể unpad lại lúc inference nếu cần
    target["letterbox_meta"] = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "orig_size": (W_orig, H_orig),
    }

    return canvas, target


AREA_BUCKETS = (
    ("tiny_<1%", 0.0, 0.01),
    ("small_1-5%", 0.01, 0.05),
    ("medium_5-20%", 0.05, 0.20),
    ("large_>=20%", 0.20, 1.01),
)


def area_bucket(area_ratio: float) -> str:
    for name, lower, upper in AREA_BUCKETS:
        if lower <= area_ratio < upper:
            return name
    return AREA_BUCKETS[-1][0]


def bucket_bounds(bucket_name: str) -> Tuple[float, float]:
    for name, lower, upper in AREA_BUCKETS:
        if name == bucket_name:
            return lower, upper
    raise KeyError(f"Unknown area bucket: {bucket_name}")


SHAPE_BUCKETS = (
    ("tall_<0.5", 0.0, 0.5),
    ("portrait_0.5-0.8", 0.5, 0.8),
    ("square_0.8-1.25", 0.8, 1.25),
    ("landscape_1.25-2", 1.25, 2.0),
    ("wide_>2", 2.0, float("inf")),
)


def shape_bucket(aspect_ratio: float) -> str:
    for name, lower, upper in SHAPE_BUCKETS:
        if lower <= aspect_ratio < upper:
            return name
    return SHAPE_BUCKETS[-1][0]


def shape_bucket_bounds(bucket_name: str) -> Tuple[float, float]:
    for name, lower, upper in SHAPE_BUCKETS:
        if name == bucket_name:
            return lower, upper
    raise KeyError(f"Unknown shape bucket: {bucket_name}")


class BoxTypeEqualizer:
    """
    Scale whole letterboxed images to synthesize under-represented box-size buckets
    within each class. Down-scaling creates more small/tiny boxes; up-scaling creates
    more medium/large boxes while cropping around the selected object.
    """

    def __init__(
        self,
        classes: List[str],
        images: List[Dict],
        ann_by_image_id: Dict[str, List[Dict]],
        img_size: int,
        p: float = 0.5,
        max_downscale: float = 0.55,
        max_upscale: float = 1.8,
        min_visibility: float = 0.25,
        min_box_size: float = 2.0,
    ):
        self.classes = classes
        self.img_size = img_size
        self.p = p
        self.max_downscale = max_downscale
        self.max_upscale = max_upscale
        self.min_visibility = min_visibility
        self.min_box_size = min_box_size
        self.bucket_names = [bucket[0] for bucket in AREA_BUCKETS]
        self.class_bucket_counts, self.class_bucket_area_ranges = self._compute_bucket_stats(images, ann_by_image_id)
        self.class_bucket_deficits = self._compute_bucket_deficits()
        self.class_bucket_ratios = self._compute_bucket_ratios()
        self.class_imbalance_ratios = self._compute_imbalance_ratios()

    def _compute_bucket_stats(
        self,
        images: List[Dict],
        ann_by_image_id: Dict[str, List[Dict]],
    ) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[str, Tuple[float, float]]]]:
        image_by_id = {image["id"]: image for image in images}
        counts = {
            class_idx: {bucket: 0 for bucket in self.bucket_names}
            for class_idx in range(len(self.classes))
        }
        areas = {
            class_idx: {bucket: [] for bucket in self.bucket_names}
            for class_idx in range(len(self.classes))
        }
        global_areas = {bucket: [] for bucket in self.bucket_names}
        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        for image_id, anns in ann_by_image_id.items():
            image_info = image_by_id.get(image_id)
            if image_info is None:
                continue
            image_width = float(image_info["width"])
            image_height = float(image_info["height"])
            scale = self.img_size / max(image_width, image_height, 1.0)
            canvas_area = float(self.img_size * self.img_size)
            for ann in anns:
                class_idx = class_to_idx.get(ann["class"])
                if class_idx is None:
                    continue
                x1, y1, x2, y2 = ann["bbox"]
                box_area = max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0) * scale * scale
                area_ratio = box_area / canvas_area
                bucket = area_bucket(area_ratio)
                counts[class_idx][bucket] += 1
                areas[class_idx][bucket].append(area_ratio)
                global_areas[bucket].append(area_ratio)

        area_ranges = {}
        for class_idx in range(len(self.classes)):
            area_ranges[class_idx] = {}
            for bucket in self.bucket_names:
                values = areas[class_idx][bucket] or global_areas[bucket]
                area_ranges[class_idx][bucket] = self._area_range_from_values(bucket, values)

        return counts, area_ranges

    def _area_range_from_values(self, bucket: str, values: List[float]) -> Tuple[float, float]:
        lower, upper = bucket_bounds(bucket)
        if values:
            low, high = np.percentile(np.asarray(values, dtype=float), [20, 80])
            low = max(float(low), lower + 1e-6)
            high = min(float(high), upper - 1e-6)
            if low < high:
                return low, high

        if lower <= 0:
            lower = min(upper * 0.10, 1e-4)
        low = lower + (upper - lower) * 0.25
        high = lower + (upper - lower) * 0.85
        return low, high

    def _compute_bucket_deficits(self) -> Dict[int, Dict[str, int]]:
        deficits = {}
        for class_idx, counts in self.class_bucket_counts.items():
            target_count = max(counts.values()) if counts else 0
            deficits[class_idx] = {
                bucket: max(target_count - count, 0)
                for bucket, count in counts.items()
            }
        return deficits

    def _compute_bucket_ratios(self) -> Dict[int, Dict[str, float]]:
        ratios = {}
        for class_idx, counts in self.class_bucket_counts.items():
            total = max(sum(counts.values()), 1)
            ratios[class_idx] = {
                bucket: count / total
                for bucket, count in counts.items()
            }
        return ratios

    def _compute_imbalance_ratios(self) -> Dict[int, float]:
        imbalance = {}
        for class_idx, counts in self.class_bucket_counts.items():
            non_zero_counts = [count for count in counts.values() if count > 0]
            if not non_zero_counts:
                imbalance[class_idx] = 0.0
            else:
                imbalance[class_idx] = max(non_zero_counts) / max(min(non_zero_counts), 1)
        return imbalance

    def stats(self) -> Dict[str, Dict]:
        return {
            self.classes[class_idx]: {
                "counts": self.class_bucket_counts[class_idx],
                "ratios": {
                    bucket: round(ratio, 6)
                    for bucket, ratio in self.class_bucket_ratios[class_idx].items()
                },
                "deficits": self.class_bucket_deficits[class_idx],
                "imbalance_ratio": round(self.class_imbalance_ratios[class_idx], 6),
                "target_area_ranges": {
                    bucket: [round(bounds[0], 6), round(bounds[1], 6)]
                    for bucket, bounds in self.class_bucket_area_ranges[class_idx].items()
                },
            }
            for class_idx in range(len(self.classes))
        }

    def __call__(self, image: Image.Image, target: dict) -> Tuple[Image.Image, dict]:
        if random.random() > self.p or target["boxes"].numel() == 0:
            return image, target

        candidate = self._sample_candidate(target["boxes"], target["labels"])
        if candidate is None:
            return image, target

        box_idx, scale = candidate
        return self._scale_image_and_boxes(image, target, box_idx, scale)

    def _sample_candidate(self, boxes: torch.Tensor, labels: torch.Tensor) -> Optional[Tuple[int, float]]:
        image_area = float(self.img_size * self.img_size)
        candidates = []
        weights = []

        for box_idx, (box, label) in enumerate(zip(boxes, labels)):
            class_idx = int(label)
            current_area = max(float((box[2] - box[0]) * (box[3] - box[1])), 1.0) / image_area
            current_bucket = area_bucket(current_area)
            deficits = self.class_bucket_deficits.get(class_idx, {})

            for target_bucket, deficit in deficits.items():
                if deficit <= 0 or target_bucket == current_bucket:
                    continue
                scale_range = self._scale_range_for_bucket(class_idx, current_area, target_bucket)
                if scale_range is None:
                    continue
                candidates.append((box_idx, scale_range))
                weights.append(float(deficit))

        if not candidates:
            return None

        selected_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
        box_idx, scale_range = candidates[selected_idx]
        return box_idx, random.uniform(scale_range[0], scale_range[1])

    def _scale_range_for_bucket(
        self,
        class_idx: int,
        current_area: float,
        target_bucket: str,
    ) -> Optional[Tuple[float, float]]:
        target_low, target_high = self.class_bucket_area_ranges[class_idx][target_bucket]
        low = (target_low / max(current_area, 1e-8)) ** 0.5
        high = (target_high / max(current_area, 1e-8)) ** 0.5

        if high < 1.0:
            low = max(low, self.max_downscale)
            high = min(high, 0.95)
        elif low > 1.0:
            low = max(low, 1.05)
            high = min(high, self.max_upscale)
        else:
            return None

        if low > high:
            return None
        return low, high

    def _scale_image_and_boxes(
        self,
        image: Image.Image,
        target: dict,
        selected_box_idx: int,
        scale: float,
    ) -> Tuple[Image.Image, dict]:
        size = self.img_size
        new_size = max(int(round(size * scale)), 1)
        resized = image.resize((new_size, new_size), Image.BILINEAR)
        boxes = target["boxes"].clone().float() * scale

        if scale < 1.0:
            offset_x = (size - new_size) // 2
            offset_y = (size - new_size) // 2
            canvas = Image.new("RGB", (size, size), (114, 114, 114))
            canvas.paste(resized, (offset_x, offset_y))
            boxes[:, [0, 2]] += offset_x
            boxes[:, [1, 3]] += offset_y
            image_out = canvas
        else:
            selected = boxes[selected_box_idx]
            center_x = float((selected[0] + selected[2]) * 0.5)
            center_y = float((selected[1] + selected[3]) * 0.5)
            jitter = size * 0.10
            crop_left = center_x - size * 0.5 + random.uniform(-jitter, jitter)
            crop_top = center_y - size * 0.5 + random.uniform(-jitter, jitter)
            max_crop = max(new_size - size, 0)
            crop_left = int(round(min(max(crop_left, 0.0), float(max_crop))))
            crop_top = int(round(min(max(crop_top, 0.0), float(max_crop))))
            image_out = resized.crop((crop_left, crop_top, crop_left + size, crop_top + size))
            boxes[:, [0, 2]] -= crop_left
            boxes[:, [1, 3]] -= crop_top

        original_boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, size)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, size)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        clipped_area = widths.clamp(min=0) * heights.clamp(min=0)
        original_area = (
            (original_boxes[:, 2] - original_boxes[:, 0]).clamp(min=1e-6)
            * (original_boxes[:, 3] - original_boxes[:, 1]).clamp(min=1e-6)
        )
        keep = (
            (widths >= self.min_box_size)
            & (heights >= self.min_box_size)
            & ((clipped_area / original_area) >= self.min_visibility)
        )

        target = dict(target)
        target["boxes"] = boxes[keep]
        target["labels"] = target["labels"][keep]
        target["box_type_equalizer"] = {"scale": scale}
        return image_out, target


class BoxShapeEqualizer:
    """
    Apply anisotropic whole-image scaling to synthesize under-represented
    bbox aspect-ratio buckets within each class. This is independent from
    BoxTypeEqualizer and uses its own train annotation statistics.
    """

    def __init__(
        self,
        classes: List[str],
        ann_by_image_id: Dict[str, List[Dict]],
        img_size: int,
        p: float = 0.5,
        max_axis_scale: float = 1.6,
        min_axis_scale: float = 0.65,
        min_visibility: float = 0.25,
        min_box_size: float = 2.0,
    ):
        self.classes = classes
        self.img_size = img_size
        self.p = p
        self.max_axis_scale = max_axis_scale
        self.min_axis_scale = min_axis_scale
        self.min_visibility = min_visibility
        self.min_box_size = min_box_size
        self.bucket_names = [bucket[0] for bucket in SHAPE_BUCKETS]
        self.class_bucket_counts, self.class_bucket_aspect_ranges = self._compute_bucket_stats(ann_by_image_id)
        self.class_bucket_deficits = self._compute_bucket_deficits()
        self.class_bucket_ratios = self._compute_bucket_ratios()
        self.class_imbalance_ratios = self._compute_imbalance_ratios()

    def _compute_bucket_stats(
        self,
        ann_by_image_id: Dict[str, List[Dict]],
    ) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[str, Tuple[float, float]]]]:
        counts = {
            class_idx: {bucket: 0 for bucket in self.bucket_names}
            for class_idx in range(len(self.classes))
        }
        aspects = {
            class_idx: {bucket: [] for bucket in self.bucket_names}
            for class_idx in range(len(self.classes))
        }
        global_aspects = {bucket: [] for bucket in self.bucket_names}
        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        for anns in ann_by_image_id.values():
            for ann in anns:
                class_idx = class_to_idx.get(ann["class"])
                if class_idx is None:
                    continue
                x1, y1, x2, y2 = ann["bbox"]
                width = max(float(x2 - x1), 0.0)
                height = max(float(y2 - y1), 0.0)
                if width <= 0 or height <= 0:
                    continue
                aspect = width / height
                bucket = shape_bucket(aspect)
                counts[class_idx][bucket] += 1
                aspects[class_idx][bucket].append(aspect)
                global_aspects[bucket].append(aspect)

        aspect_ranges = {}
        for class_idx in range(len(self.classes)):
            aspect_ranges[class_idx] = {}
            for bucket in self.bucket_names:
                values = aspects[class_idx][bucket] or global_aspects[bucket]
                aspect_ranges[class_idx][bucket] = self._aspect_range_from_values(bucket, values)

        return counts, aspect_ranges

    def _aspect_range_from_values(self, bucket: str, values: List[float]) -> Tuple[float, float]:
        lower, upper = shape_bucket_bounds(bucket)
        if values:
            low, high = np.percentile(np.asarray(values, dtype=float), [20, 80])
            low = max(float(low), lower + 1e-6)
            high = float(high) if np.isinf(upper) else min(float(high), upper - 1e-6)
            if low < high:
                return low, high

        if np.isinf(upper):
            return lower * 1.10, lower * 2.00
        if lower <= 0:
            lower = min(upper * 0.10, 0.05)
        low = lower + (upper - lower) * 0.25
        high = lower + (upper - lower) * 0.85
        return low, high

    def _compute_bucket_deficits(self) -> Dict[int, Dict[str, int]]:
        deficits = {}
        for class_idx, counts in self.class_bucket_counts.items():
            target_count = max(counts.values()) if counts else 0
            deficits[class_idx] = {
                bucket: max(target_count - count, 0)
                for bucket, count in counts.items()
            }
        return deficits

    def _compute_bucket_ratios(self) -> Dict[int, Dict[str, float]]:
        ratios = {}
        for class_idx, counts in self.class_bucket_counts.items():
            total = max(sum(counts.values()), 1)
            ratios[class_idx] = {
                bucket: count / total
                for bucket, count in counts.items()
            }
        return ratios

    def _compute_imbalance_ratios(self) -> Dict[int, float]:
        imbalance = {}
        for class_idx, counts in self.class_bucket_counts.items():
            non_zero_counts = [count for count in counts.values() if count > 0]
            if not non_zero_counts:
                imbalance[class_idx] = 0.0
            else:
                imbalance[class_idx] = max(non_zero_counts) / max(min(non_zero_counts), 1)
        return imbalance

    def stats(self) -> Dict[str, Dict]:
        return {
            self.classes[class_idx]: {
                "counts": self.class_bucket_counts[class_idx],
                "ratios": {
                    bucket: round(ratio, 6)
                    for bucket, ratio in self.class_bucket_ratios[class_idx].items()
                },
                "deficits": self.class_bucket_deficits[class_idx],
                "imbalance_ratio": round(self.class_imbalance_ratios[class_idx], 6),
                "target_aspect_ranges": {
                    bucket: [round(bounds[0], 6), round(bounds[1], 6)]
                    for bucket, bounds in self.class_bucket_aspect_ranges[class_idx].items()
                },
            }
            for class_idx in range(len(self.classes))
        }

    def __call__(self, image: Image.Image, target: dict) -> Tuple[Image.Image, dict]:
        if random.random() > self.p or target["boxes"].numel() == 0:
            return image, target

        candidate = self._sample_candidate(target["boxes"], target["labels"])
        if candidate is None:
            return image, target

        box_idx, scale_x, scale_y = candidate
        return self._anisotropic_scale_image_and_boxes(image, target, box_idx, scale_x, scale_y)

    def _sample_candidate(self, boxes: torch.Tensor, labels: torch.Tensor) -> Optional[Tuple[int, float, float]]:
        candidates = []
        weights = []

        for box_idx, (box, label) in enumerate(zip(boxes, labels)):
            width = max(float(box[2] - box[0]), 1e-6)
            height = max(float(box[3] - box[1]), 1e-6)
            current_aspect = width / height
            current_bucket = shape_bucket(current_aspect)
            class_idx = int(label)
            deficits = self.class_bucket_deficits.get(class_idx, {})

            for target_bucket, deficit in deficits.items():
                if deficit <= 0 or target_bucket == current_bucket:
                    continue
                scale_pair = self._axis_scales_for_bucket(class_idx, current_aspect, target_bucket)
                if scale_pair is None:
                    continue
                candidates.append((box_idx, scale_pair[0], scale_pair[1]))
                weights.append(float(deficit))

        if not candidates:
            return None

        selected_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
        return candidates[selected_idx]

    def _axis_scales_for_bucket(
        self,
        class_idx: int,
        current_aspect: float,
        target_bucket: str,
    ) -> Optional[Tuple[float, float]]:
        target_low, target_high = self.class_bucket_aspect_ranges[class_idx][target_bucket]
        target_aspect = random.uniform(target_low, target_high)
        ratio = target_aspect / max(current_aspect, 1e-8)
        if 0.98 <= ratio <= 1.02:
            return None

        scale_x = ratio ** 0.5
        scale_y = 1.0 / max(scale_x, 1e-8)
        scale_x = min(max(scale_x, self.min_axis_scale), self.max_axis_scale)
        scale_y = min(max(scale_y, self.min_axis_scale), self.max_axis_scale)

        achieved_ratio = scale_x / max(scale_y, 1e-8)
        achieved_aspect = current_aspect * achieved_ratio
        if shape_bucket(achieved_aspect) != target_bucket:
            return None
        return scale_x, scale_y

    def _axis_crop_or_pad(self, new_size: int, selected_center: float, target_size: int) -> Tuple[int, int]:
        if new_size <= target_size:
            return 0, (target_size - new_size) // 2

        jitter = target_size * 0.10
        max_crop = new_size - target_size
        crop = selected_center - target_size * 0.5 + random.uniform(-jitter, jitter)
        crop = int(round(min(max(crop, 0.0), float(max_crop))))
        return crop, 0

    def _anisotropic_scale_image_and_boxes(
        self,
        image: Image.Image,
        target: dict,
        selected_box_idx: int,
        scale_x: float,
        scale_y: float,
    ) -> Tuple[Image.Image, dict]:
        size = self.img_size
        new_w = max(int(round(size * scale_x)), 1)
        new_h = max(int(round(size * scale_y)), 1)
        resized = image.resize((new_w, new_h), Image.BILINEAR)

        boxes = target["boxes"].clone().float()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        selected = boxes[selected_box_idx]
        selected_center_x = float((selected[0] + selected[2]) * 0.5)
        selected_center_y = float((selected[1] + selected[3]) * 0.5)
        crop_left, pad_left = self._axis_crop_or_pad(new_w, selected_center_x, size)
        crop_top, pad_top = self._axis_crop_or_pad(new_h, selected_center_y, size)

        crop_right = crop_left + min(new_w, size)
        crop_bottom = crop_top + min(new_h, size)
        cropped = resized.crop((crop_left, crop_top, crop_right, crop_bottom))
        image_out = Image.new("RGB", (size, size), (114, 114, 114))
        image_out.paste(cropped, (pad_left, pad_top))

        boxes[:, [0, 2]] = boxes[:, [0, 2]] - crop_left + pad_left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - crop_top + pad_top

        original_boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, size)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, size)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        clipped_area = widths.clamp(min=0) * heights.clamp(min=0)
        original_area = (
            (original_boxes[:, 2] - original_boxes[:, 0]).clamp(min=1e-6)
            * (original_boxes[:, 3] - original_boxes[:, 1]).clamp(min=1e-6)
        )
        keep = (
            (widths >= self.min_box_size)
            & (heights >= self.min_box_size)
            & ((clipped_area / original_area) >= self.min_visibility)
        )

        target = dict(target)
        target["boxes"] = boxes[keep]
        target["labels"] = target["labels"][keep]
        target["box_shape_equalizer"] = {"scale_x": scale_x, "scale_y": scale_y}
        return image_out, target


class MosaicAugmentation:
    """YOLO-style 4-image mosaic on the final training canvas."""

    def __init__(
        self,
        img_size: int,
        p: float = 0.5,
        min_visibility: float = 0.10,
        min_box_size: float = 2.0,
        fill: Tuple[int, int, int] = (114, 114, 114),
    ):
        self.img_size = img_size
        self.p = p
        self.min_visibility = min_visibility
        self.min_box_size = min_box_size
        self.fill = fill

    def should_apply(self) -> bool:
        return random.random() < self.p

    def __call__(self, dataset: "CustomDataset", idx: int) -> Tuple[Image.Image, dict]:
        size = self.img_size
        base_image_id = dataset.images[idx]["id"]
        indices = [idx] + random.choices(range(len(dataset)), k=3)
        random.shuffle(indices)
        center_x = int(random.uniform(size * 0.25, size * 0.75))
        center_y = int(random.uniform(size * 0.25, size * 0.75))

        mosaic = Image.new("RGB", (size, size), self.fill)
        all_boxes = []
        all_labels = []
        source_ids = []

        for mosaic_idx, sample_idx in enumerate(indices):
            image, target = dataset.load_raw_sample(sample_idx)
            source_ids.append(target["image_id"])
            image, boxes = self._resize_image_and_boxes(image, target["boxes"])
            labels = target["labels"]
            paste_box = self._placement(mosaic_idx, image.size, center_x, center_y)
            if paste_box is None:
                continue

            dst, src = paste_box
            dst_x1, dst_y1, dst_x2, dst_y2 = dst
            src_x1, src_y1, src_x2, src_y2 = src
            crop = image.crop((src_x1, src_y1, src_x2, src_y2))
            mosaic.paste(crop, (dst_x1, dst_y1))

            if boxes.numel() == 0:
                continue

            boxes = boxes.clone().float()
            boxes[:, [0, 2]] += dst_x1 - src_x1
            boxes[:, [1, 3]] += dst_y1 - src_y1
            original_boxes = boxes.clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, size)

            keep = self._valid_boxes(boxes, original_boxes)
            if keep.any():
                all_boxes.append(boxes[keep])
                all_labels.append(labels[keep])

        if all_boxes:
            boxes_out = torch.cat(all_boxes, dim=0)
            labels_out = torch.cat(all_labels, dim=0).long()
        else:
            boxes_out = torch.zeros((0, 4), dtype=torch.float32)
            labels_out = torch.zeros((0,), dtype=torch.int64)

        return mosaic, {
            "boxes": boxes_out,
            "labels": labels_out,
            "image_id": base_image_id,
            "mosaic": {"source_image_ids": source_ids},
        }

    def _resize_image_and_boxes(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
    ) -> Tuple[Image.Image, torch.Tensor]:
        width, height = image.size
        scale = self.img_size / max(width, height, 1)
        new_w = max(int(round(width * scale)), 1)
        new_h = max(int(round(height * scale)), 1)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        boxes = boxes.clone().float()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale
        return image, boxes

    def _placement(
        self,
        mosaic_idx: int,
        image_size: Tuple[int, int],
        center_x: int,
        center_y: int,
    ) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
        size = self.img_size
        width, height = image_size

        if mosaic_idx == 0:
            dst_x1 = max(center_x - width, 0)
            dst_y1 = max(center_y - height, 0)
            dst_x2, dst_y2 = center_x, center_y
            src_x1 = width - (dst_x2 - dst_x1)
            src_y1 = height - (dst_y2 - dst_y1)
            src_x2, src_y2 = width, height
        elif mosaic_idx == 1:
            dst_x1 = center_x
            dst_y1 = max(center_y - height, 0)
            dst_x2 = min(center_x + width, size)
            dst_y2 = center_y
            src_x1 = 0
            src_y1 = height - (dst_y2 - dst_y1)
            src_x2, src_y2 = dst_x2 - dst_x1, height
        elif mosaic_idx == 2:
            dst_x1 = max(center_x - width, 0)
            dst_y1 = center_y
            dst_x2 = center_x
            dst_y2 = min(center_y + height, size)
            src_x1 = width - (dst_x2 - dst_x1)
            src_y1 = 0
            src_x2, src_y2 = width, dst_y2 - dst_y1
        else:
            dst_x1, dst_y1 = center_x, center_y
            dst_x2 = min(center_x + width, size)
            dst_y2 = min(center_y + height, size)
            src_x1, src_y1 = 0, 0
            src_x2, src_y2 = dst_x2 - dst_x1, dst_y2 - dst_y1

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1 or src_x2 <= src_x1 or src_y2 <= src_y1:
            return None
        return (dst_x1, dst_y1, dst_x2, dst_y2), (src_x1, src_y1, src_x2, src_y2)

    def _valid_boxes(self, boxes: torch.Tensor, original_boxes: torch.Tensor) -> torch.Tensor:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        clipped_area = widths.clamp(min=0) * heights.clamp(min=0)
        original_area = (
            (original_boxes[:, 2] - original_boxes[:, 0]).clamp(min=1e-6)
            * (original_boxes[:, 3] - original_boxes[:, 1]).clamp(min=1e-6)
        )
        return (
            (widths >= self.min_box_size)
            & (heights >= self.min_box_size)
            & ((clipped_area / original_area) >= self.min_visibility)
        )



# Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        data_root,
        split: str = "train",
        transforms: Optional[Callable] = None,
        normalize: bool = True,
        img_size: int = 512,         # target size cho letterbox
        box_type_equalizer: bool = False,
        box_type_equalizer_p: float = 0.5,
        box_shape_equalizer: bool = False,
        box_shape_equalizer_p: float = 0.5,
        mosaic: bool = False,
        mosaic_p: float = 0.5,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms
        self.normalize = normalize
        self.img_size = img_size
        self.box_type_equalizer = None
        self.box_shape_equalizer = None
        self.mosaic = (
            MosaicAugmentation(img_size=self.img_size, p=mosaic_p)
            if mosaic and split == "train"
            else None
        )

        annotation_path = self.data_root / "annotations" / f"{self.split}.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file {annotation_path} not found.")

        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.classes: List[str] = data["classes"]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images: List[Dict] = data["images"]

        self.ann_by_image_id: Dict[str, List[Dict]] = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            self.ann_by_image_id.setdefault(img_id, []).append(ann)

        if box_type_equalizer and split == "train":
            self.box_type_equalizer = BoxTypeEqualizer(
                classes=self.classes,
                images=self.images,
                ann_by_image_id=self.ann_by_image_id,
                img_size=self.img_size,
                p=box_type_equalizer_p,
            )
        if box_shape_equalizer and split == "train":
            self.box_shape_equalizer = BoxShapeEqualizer(
                classes=self.classes,
                ann_by_image_id=self.ann_by_image_id,
                img_size=self.img_size,
                p=box_shape_equalizer_p,
            )

        self._normalize = trans.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images)

    def load_raw_sample(self, idx: int) -> Tuple[Image.Image, dict]:
        img_info = self.images[idx]
        img_id   = img_info["id"]

        # Load ảnh
        img_path = self.data_root / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        # Build target
        anns = self.ann_by_image_id.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            xmin, ymin, xmax, ymax = ann["bbox"]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[ann["class"]])

        if boxes:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": img_id}
        return img, target

    def __getitem__(self, idx: int):
        use_mosaic = self.mosaic is not None and self.mosaic.should_apply()
        if use_mosaic:
            img, target = self.mosaic(self, idx)
        else:
            img, target = self.load_raw_sample(idx)

            # Letterbox resize (scale + pad) — trước augmentation
            img, target = letterbox_resize(img, self.img_size, target)

            if self.box_type_equalizer is not None:
                img, target = self.box_type_equalizer(img, target)
            if self.box_shape_equalizer is not None:
                img, target = self.box_shape_equalizer(img, target)

        # Augmentation (albumentations hoặc custom)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # To tensor
        img = trans_func.to_tensor(img)           # (3, H, W), float [0,1]
        if self.normalize:
            img = self._normalize(img)

        return img, target



# collate_fn — đơn giản vì tất cả ảnh đã cùng size sau letterbox


def collate_fn(batch):
    imgs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Build dataloader


def build_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    transforms: Optional[Callable] = None,
    normalize: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    img_size: int = 512,
    seed: Optional[int] = None,
    box_type_equalizer: bool = False,
    box_type_equalizer_p: float = 0.5,
    box_shape_equalizer: bool = False,
    box_shape_equalizer_p: float = 0.5,
    mosaic: bool = False,
    mosaic_p: float = 0.5,
) -> DataLoader:

    dataset = CustomDataset(
        data_root=data_root,
        split=split,
        transforms=transforms,
        normalize=normalize,
        img_size=img_size,
        box_type_equalizer=box_type_equalizer,
        box_type_equalizer_p=box_type_equalizer_p,
        box_shape_equalizer=box_shape_equalizer,
        box_shape_equalizer_p=box_shape_equalizer_p,
        mosaic=mosaic,
        mosaic_p=mosaic_p,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    shuffle = (split == "train")

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=seed_worker if seed is not None else None,
        generator=generator,
    )

    return loader



# Albumentations augmentation (sau letterbox, bbox đã ở tọa độ mới)


def albumentations_transform():
    if A is None:
        raise ImportError(
            "albumentations is not installed. Install it or run training with --no_aug."
        )

    albu = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1),
                 rotate=(-10, 10), p=0.3),
        A.OneOf([
            A.MotionBlur(p=1),
            A.GaussNoise(p=1),
        ], p=0.2),
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.2,
    ))

    def transform(image: Image.Image, target: dict):
        img_np = np.array(image)
        boxes  = target["boxes"].tolist()
        labels = target["labels"].tolist()

        result = albu(image=img_np, bboxes=boxes, labels=labels)

        image_out = Image.fromarray(result["image"])
        new_boxes  = result["bboxes"]
        new_labels = result["labels"]

        if new_boxes:
            target["boxes"]  = torch.tensor(new_boxes,  dtype=torch.float32)
            target["labels"] = torch.tensor(new_labels, dtype=torch.int64)
        else:
            target["boxes"]  = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,),   dtype=torch.int64)

        return image_out, target

    return transform



# Visualize loader (không normalize để dễ hiển thị)


def build_visualized_loader(
    data_root: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    transforms=None,
    img_size: int = 512,
    box_type_equalizer: bool = False,
    box_shape_equalizer: bool = False,
    mosaic: bool = False,
    mosaic_p: float = 0.5,
) -> DataLoader:
    return build_dataloader(
        data_root=data_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        transforms=transforms,
        normalize=False,
        pin_memory=False,
        drop_last=False,
        img_size=img_size,
        box_type_equalizer=box_type_equalizer,
        box_shape_equalizer=box_shape_equalizer,
        mosaic=mosaic,
        mosaic_p=mosaic_p,
    )
