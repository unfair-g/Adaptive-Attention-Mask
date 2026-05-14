"""Mapillary Vistas dataset mapped to Cityscapes-19 classes.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.cityscapes import CITYSCAPES_PALETTE


MAPILLARY_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]
MAPILLARY_CLASSES_WITH_BG = MAPILLARY_CLASSES + ["background"]
BACKGROUND_CLASS = 19


MAPILLARY_TO_CITYSCAPES = {
    # ================= flat =================
    "construction--flat--road": 0,
    "construction--flat--service-lane": 0,
    "construction--flat--parking": 0,
    "construction--flat--bike-lane": 0,
    "construction--flat--rail-track": 0,
    "marking--general": 0,
    "marking--crosswalk-zebra": 0,
    "construction--flat--sidewalk": 1,
    "construction--barrier--curb": 1,
    "construction--flat--curb-cut": 1,
    "construction--flat--pedestrian-area": 1,
    "construction--flat--crosswalk-plain": 1,
    # ================= construction =================
    "construction--structure--building": 2,
    "construction--structure--bridge": 2,
    "construction--structure--tunnel": 2,
    "construction--barrier--wall": 3,
    "construction--barrier--fence": 4,
    "construction--barrier--guard-rail": 4,
    "construction--barrier--other-barrier": 4,
    # ================= object =================
    "object--support--pole": 5,
    "object--support--utility-pole": 5,
    "object--street-light": 5,
    "object--traffic-light": 6,
    "object--traffic-sign--front": 7,
    "object--traffic-sign--back": 7,
    "object--support--traffic-sign-frame": 7,
    # ================= nature =================
    "nature--vegetation": 8,
    "nature--terrain": 9,
    "nature--sand": 9,
    "nature--snow": 9,
    "nature--mountain": 9,
    "nature--sky": 10,
    # ================= human =================
    "human--person": 11,
    "human--rider--bicyclist": 12,
    "human--rider--motorcyclist": 12,
    "human--rider--other-rider": 12,
    # ================= vehicle =================
    "object--vehicle--car": 13,
    "object--vehicle--truck": 14,
    "object--vehicle--trailer": 14,
    "object--vehicle--caravan": 14,
    "object--vehicle--other-vehicle": 14,
    "object--vehicle--bus": 15,
    "object--vehicle--on-rails": 16,
    "object--vehicle--motorcycle": 17,
    "object--vehicle--bicycle": 18,
}


IGNORE_CLASSES = {
    "animal--bird",
    "animal--ground-animal",
    "nature--water",
    "object--banner",
    "object--bench",
    "object--bike-rack",
    "object--billboard",
    "object--catch-basin",
    "object--cctv-camera",
    "object--fire-hydrant",
    "object--junction-box",
    "object--mailbox",
    "object--manhole",
    "object--phone-booth",
    "object--pothole",
    "object--trash-can",
    "object--vehicle--boat",
    "object--vehicle--wheeled-slow",
    "void--car-mount",
    "void--ego-vehicle",
    "void--unlabeled",
}


def _normalize_label_text(text):
    if not isinstance(text, str):
        return ""
    s = text.strip().lower()
    for ch in ["_", "/", "(", ")", "[", "]", ",", ".", ":", ";"]:
        s = s.replace(ch, " ")
    s = s.replace("-", " ")
    s = " ".join(s.split())
    return s


# Fallback mapping for config.json formats that store readable class names.
READABLE_TO_CITYSCAPES = {
    "road": 0,
    "service lane": 0,
    "parking": 0,
    "bike lane": 0,
    "rail track": 0,
    "general": 0,
    "lane marking general": 0,
    "crosswalk zebra": 0,
    "zebra": 0,
    "sidewalk": 1,
    "curb": 1,
    "curb cut": 1,
    "pedestrian area": 1,
    "crosswalk plain": 1,
    "building": 2,
    "bridge": 2,
    "tunnel": 2,
    "wall": 3,
    "fence": 4,
    "guard rail": 4,
    "other barrier": 4,
    "pole": 5,
    "utility pole": 5,
    "street light": 5,
    "traffic light": 6,
    "traffic sign": 7,
    "traffic sign front": 7,
    "traffic sign back": 7,
    "traffic sign frame": 7,
    "vegetation": 8,
    "terrain": 9,
    "sand": 9,
    "snow": 9,
    "mountain": 9,
    "sky": 10,
    "person": 11,
    "bicyclist": 12,
    "motorcyclist": 12,
    "other rider": 12,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "trailer": 14,
    "caravan": 14,
    "other vehicle": 14,
    "bus": 15,
    "on rails": 16,
    "motorcycle": 17,
    "bicycle": 18,
}

READABLE_IGNORE_CLASSES = {_normalize_label_text(x) for x in IGNORE_CLASSES}


class MapillaryVistas(Dataset):
    """Mapillary Vistas mapped into Cityscapes 19 classes.

    Directory layout (both variants are supported):
      - {root}/training/image/*.jpg and {root}/training/labels/*.png
      - {root}/training/images/*.jpg and {root}/training/labels/*.png
      - {root}/validation/image/*.jpg and {root}/validation/labels/*.png
      - {root}/validation/images/*.jpg and {root}/validation/labels/*.png
    """

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        img_size=(512, 1024),
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
        self.n_classes = 19
        self.ignore_index = 255

        self.id_to_trainid = self._build_id_to_trainid()
        self.images = []
        self.masks = []
        self._load_file_list()

    def _split_dir_name(self):
        split_name = self.split.lower()
        if split_name == "train":
            return "training"
        if split_name in {"val", "valid", "validation"}:
            return "validation"
        return split_name

    def _build_id_to_trainid(self):
        cfg_path = os.path.join(self.root, "config.json")
        if not os.path.isfile(cfg_path):
            raise RuntimeError(
                f"Mapillary config file not found: {cfg_path}. "
                "Expected label metadata to build ID->trainID mapping."
            )

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        labels = cfg.get("labels", [])
        if not labels:
            raise RuntimeError(f"No 'labels' found in {cfg_path}")

        mapping = {}
        mapped_count = 0
        seen_names = []
        for idx, entry in enumerate(labels):
            # Some Mapillary config.json versions do not provide explicit `id`.
            # In that case the label id is the position in labels[].
            raw_id = entry.get("id", idx)
            candidates = []
            for key in ("name", "readable", "label", "class", "category"):
                value = entry.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)
            seen_names.extend(candidates[:1])

            assigned = None
            for cand in candidates:
                if cand in MAPILLARY_TO_CITYSCAPES:
                    assigned = MAPILLARY_TO_CITYSCAPES[cand]
                    break
                if cand in IGNORE_CLASSES:
                    assigned = self.ignore_index
                    break

                normalized = _normalize_label_text(cand)
                if normalized in READABLE_TO_CITYSCAPES:
                    assigned = READABLE_TO_CITYSCAPES[normalized]
                    break
                if normalized in READABLE_IGNORE_CLASSES:
                    assigned = self.ignore_index
                    break

            if assigned is None:
                assigned = self.ignore_index
            else:
                if assigned != self.ignore_index:
                    mapped_count += 1
            mapping[int(raw_id)] = assigned

        if mapped_count == 0:
            raise RuntimeError(
                "Mapillary->Cityscapes mapping matched 0 classes from config.json. "
                "Please verify label metadata fields. "
                f"Example labels in config.json: {seen_names[:20]}"
            )

        return mapping

    def _resolve_dirs(self):
        base = os.path.join(self.root, self._split_dir_name())
        image_candidates = [
            os.path.join(base, "image"),
            os.path.join(base, "images"),
        ]
        label_dir = os.path.join(base, "labels")

        image_dir = None
        for candidate in image_candidates:
            if os.path.isdir(candidate):
                image_dir = candidate
                break

        if image_dir is None:
            raise RuntimeError(
                f"Mapillary image directory not found under: {base}. "
                "Expected one of: image/, images/"
            )
        if not os.path.isdir(label_dir):
            raise RuntimeError(f"Mapillary label directory not found: {label_dir}")

        return image_dir, label_dir

    def _load_file_list(self):
        image_dir, label_dir = self._resolve_dirs()

        for img_name in sorted(os.listdir(image_dir)):
            img_path = os.path.join(image_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            stem, ext = os.path.splitext(img_name)
            if ext.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            mask_path = os.path.join(label_dir, f"{stem}.png")
            if os.path.isfile(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)

        if len(self.images) == 0:
            raise RuntimeError(
                "No matched image/mask pairs found for Mapillary. "
                f"image_dir={image_dir}, label_dir={label_dir}"
            )

        print(f"Loaded {len(self.images)} Mapillary images for {self.split} split")

    def _convert_label(self, mask):
        mask_np = np.array(mask, dtype=np.int32)
        converted = np.full_like(mask_np, self.ignore_index)

        for raw_id, train_id in self.id_to_trainid.items():
            converted[mask_np == raw_id] = train_id

        return converted

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])

        if self.img_size is not None:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        mask = self._convert_label(mask)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(mask).long()

        return img, mask, index

    def get_original_converted_mask(self, index):
        mask = Image.open(self.masks[index])
        if self.img_size is not None:
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        mask = self._convert_label(mask)
        return torch.from_numpy(np.ascontiguousarray(mask)).long()

    def __len__(self):
        return len(self.images)

    @staticmethod
    def decode_target(mask):
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for label_id, color in enumerate(CITYSCAPES_PALETTE):
            rgb[mask == label_id] = color
        return rgb


class SplitMapillaryVistas(MapillaryVistas):
    """Split MapillaryVistas dataset for class-incremental learning."""

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        selected_labels=[0],
        img_size=(512, 1024),
        old_labels=None,
        mode="current_only",
        background_class=BACKGROUND_CLASS,
    ):
        self.selected_labels = selected_labels
        self.old_labels = old_labels if old_labels is not None else []
        self.mode = mode
        self.background_class = background_class

        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            img_size=img_size,
        )

        if "background" in self.mode:
            self.n_classes = 20

        print(
            f"SplitMapillaryVistas: {len(self.images)} images, "
            f"current_labels={self.selected_labels}, old_labels={self.old_labels}, "
            f"mode={self.mode}, n_classes={self.n_classes}"
        )

    def _mask_labels(self, mask):
        if self.mode == "current_only":
            processed = np.full_like(mask, self.ignore_index)
            for label in self.selected_labels:
                processed[mask == label] = label
            return processed

        if self.mode == "current_and_old":
            processed = np.full_like(mask, self.ignore_index)
            allowed = self.selected_labels + self.old_labels
            for label in allowed:
                processed[mask == label] = label
            return processed

        if self.mode == "unknown_as_background":
            processed = np.full_like(mask, self.background_class)
            for label in self.selected_labels:
                processed[mask == label] = label
            for label in self.old_labels:
                processed[mask == label] = label
            processed[mask == 255] = self.ignore_index
            return processed

        if self.mode == "all_unknown_as_background":
            processed = np.full_like(mask, self.background_class)
            for label in self.selected_labels:
                processed[mask == label] = label
            processed[mask == 255] = self.ignore_index
            return processed

        raise ValueError(
            f"Unknown mode: {self.mode}. "
            "Available: current_only, current_and_old, "
            "unknown_as_background, all_unknown_as_background"
        )

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])

        if self.img_size is not None:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        mask = self._convert_label(mask)
        mask = self._mask_labels(mask)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(mask).long()

        return img, mask, index


class BlurryMapillaryVistas(MapillaryVistas):
    """Mapillary dataset with blurry task boundaries for online CL."""

    def __init__(
        self,
        root,
        labels_order,
        split="train",
        transform=None,
        n_tasks=5,
        scale=500,
        img_size=(512, 1024),
    ):
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            img_size=img_size,
        )
        self.labels_order = labels_order
        self.n_tasks = n_tasks
        self.scale = scale
        self._reorder_dataset()

    def _get_dominant_class(self, mask_path):
        mask = Image.open(mask_path)
        if self.img_size is not None:
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        mask = self._convert_label(mask)
        unique, counts = np.unique(mask[mask != 255], return_counts=True)
        if len(unique) == 0:
            return -1
        return unique[np.argmax(counts)]

    def _reorder_dataset(self):
        class_to_images = {i: [] for i in range(self.n_classes)}
        class_to_masks = {i: [] for i in range(self.n_classes)}

        for img_path, mask_path in zip(self.images, self.masks):
            dominant = self._get_dominant_class(mask_path)
            if dominant >= 0:
                class_to_images[dominant].append(img_path)
                class_to_masks[dominant].append(mask_path)

        step_size = self.n_classes // self.n_tasks
        new_images = []
        new_masks = []

        for task_id in range(self.n_tasks):
            task_classes = self.labels_order[task_id * step_size:(task_id + 1) * step_size]
            task_images = []
            task_masks = []

            for cls in task_classes:
                task_images.extend(class_to_images[cls])
                task_masks.extend(class_to_masks[cls])

            combined = list(zip(task_images, task_masks))
            np.random.shuffle(combined)
            if combined:
                task_images, task_masks = zip(*combined)
                new_images.extend(task_images)
                new_masks.extend(task_masks)

        self.images = list(new_images)
        self.masks = list(new_masks)
        print(f"Reordered {len(self.images)} Mapillary images for blurry CL")
