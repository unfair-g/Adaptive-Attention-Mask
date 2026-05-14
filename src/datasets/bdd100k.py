"""BDD100K dataset implementation for semantic segmentation.
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# 19 evaluation classes aligned with Cityscapes semantics
BDD100K_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

BDD100K_CLASSES_WITH_BG = BDD100K_CLASSES + ["background"]
BACKGROUND_CLASS = 19

# Typical BDD100K semantic label-id -> train-id mapping
# unlabeled(0)->255, classes(1..19)->(0..18)
BDD100K_ID_TO_TRAINID = {0: 255, **{k: k - 1 for k in range(1, 20)}, 255: 255}

# Reuse the standard 19-class Cityscapes-like palette for visualization.
BDD100K_PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32)
]


class BDD100K(Dataset):
    """BDD100K semantic segmentation dataset.

    File pairing::
        00a7ef03-00000000.jpg  <->  00a7ef03-00000000_train_id.png

    Root layout examples (``split`` is ``train`` or ``val``)::

        images/10k/{split}/  or  images/{split}/
        labels/sem_seg/masks/{split}/  or  labels/{split}/

    Expect 19 train ids (0--18) and ignore 255 in ``*_train_id.png``.
    If a mask uses label-id encoding (0 unlabeled, 1..19 classes), it is
    converted to train ids as a fallback.
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

        self.images = []
        self.masks = []
        self._load_file_list()

    def _get_dir(self, candidates):
        for rel in candidates:
            abs_path = os.path.join(self.root, rel)
            if os.path.isdir(abs_path):
                return abs_path
        return None

    def _load_file_list(self):
        """Pair each .jpg with ``{stem}_train_id.png`` in the mask folder."""
        img_dir = self._get_dir(
            [
                os.path.join("images", "10k", self.split),
                os.path.join("images", self.split),
            ]
        )
        mask_dir = self._get_dir(
            [
                os.path.join("labels", "sem_seg", "masks", self.split),
                os.path.join("labels", "sem_seg", self.split),
                # Flat layout: ./data/bdd100k/labels/train next to ./data/bdd100k/images/train
                os.path.join("labels", self.split),
                os.path.join("seg", "labels", self.split),
                os.path.join("masks", self.split),
            ]
        )
        if img_dir is None:
            raise RuntimeError(f"BDD100K image directory not found under: {self.root}")
        if mask_dir is None:
            tried = [
                os.path.join(self.root, "labels", "sem_seg", "masks", self.split),
                os.path.join(self.root, "labels", "sem_seg", self.split),
                os.path.join(self.root, "labels", self.split),
            ]
            raise RuntimeError(
                f"BDD100K mask directory not found under: {self.root}. "
                f"Tried (among others): {tried}"
            )

        for img_name in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            if not img_name.lower().endswith(".jpg"):
                continue

            stem = os.path.splitext(img_name)[0]
            mask_path = os.path.join(mask_dir, f"{stem}_train_id.png")
            if os.path.isfile(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)

        if len(self.images) == 0:
            raise RuntimeError(
                f"No matched image/mask pairs (jpg + stem_train_id.png). "
                f"img_dir={img_dir}, mask_dir={mask_dir}"
            )
        print(f"Loaded {len(self.images)} BDD100K images for {self.split} split")

    def _convert_label(self, mask):
        """Convert BDD100K mask ids to train ids (0-18, 255 ignore)."""
        mask_np = np.array(mask, dtype=np.int32)
        unique_vals = set(np.unique(mask_np).tolist())

        # Already train-id format.
        if unique_vals.issubset(set(range(19)).union({255})):
            return mask_np

        # Label-id format (0 unlabeled, 1..19 semantic classes).
        converted = np.full_like(mask_np, self.ignore_index)
        for raw_id, train_id in BDD100K_ID_TO_TRAINID.items():
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
        """Train-id mask (0--18, 255 ignore) before any incremental label masking.

        Matches ``SplitBDD100K`` / attention & logit-KD use: distinguish native ignore
        (255) from pixels later set to ignore by ``current_only`` / similar modes.
        Does not apply ``target_transform`` (same contract as ``SplitCityscapes``).
        """
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
        for label_id, color in enumerate(BDD100K_PALETTE):
            rgb[mask == label_id] = color
        return rgb


class SplitBDD100K(BDD100K):
    """Split BDD100K dataset for class-incremental learning."""

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
            f"SplitBDD100K: {len(self.images)} images, "
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


class BlurryBDD100K(BDD100K):
    """BDD100K dataset with blurry task boundaries for online CL."""

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
            task_classes = self.labels_order[task_id * step_size: (task_id + 1) * step_size]
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
        print(f"Reordered {len(self.images)} BDD100K images for blurry CL")
