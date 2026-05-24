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



# Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        data_root,
        split: str = "train",
        transforms: Optional[Callable] = None,
        normalize: bool = True,
        img_size: int = 512,         # target size cho letterbox
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms
        self.normalize = normalize
        self.img_size = img_size

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

        self._normalize = trans.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
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

        # Letterbox resize (scale + pad) — trước augmentation
        img, target = letterbox_resize(img, self.img_size, target)

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
) -> DataLoader:

    dataset = CustomDataset(
        data_root=data_root,
        split=split,
        transforms=transforms,
        normalize=normalize,
        img_size=img_size,
    )

    shuffle = (split == "train")
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

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
    )
