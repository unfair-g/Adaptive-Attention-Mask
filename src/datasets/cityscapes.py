"""Cityscapes dataset implementation for semantic segmentation.
"""
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Cityscapes class definitions
# 19 evaluation classes as per standard Cityscapes benchmark
# Index 19 is reserved for "background/unknown" class in incremental learning
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# Extended class list with background class for incremental learning
CITYSCAPES_CLASSES_WITH_BG = CITYSCAPES_CLASSES + ['background']

# Background class index (used for unknown/unseen classes in incremental learning)
BACKGROUND_CLASS = 19

# Mapping from original IDs to training IDs (0-18, 255 for ignore)
CITYSCAPES_ID_TO_TRAINID = {
    -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
    16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18
}

# Color palette for visualization
CITYSCAPES_PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32)
]


class Cityscapes(Dataset):
    """Base Cityscapes dataset for semantic segmentation.
    
    Args:
        root: Path to Cityscapes data root (containing leftImg8bit and gtFine)
        split: 'train' or 'val'
        transform: Transform to apply to images
        target_transform: Transform to apply to segmentation masks
        img_size: Size to resize images to (default: 512x1024)
    """
    
    def __init__(
        self,
        root,
        split='train',
        transform=None,
        target_transform=None,
        img_size=(512, 1024)
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
        self.n_classes = 19
        self.ignore_index = 255
        
        # Build file lists
        self.images = []
        self.masks = []
        self._load_file_list()
        
    def _load_file_list(self):
        """Load list of image and mask file paths."""
        img_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        mask_dir = os.path.join(self.root, 'gtFine', self.split)
        
        if not os.path.exists(img_dir):
            raise RuntimeError(f"Image directory not found: {img_dir}")
        if not os.path.exists(mask_dir):
            raise RuntimeError(f"Mask directory not found: {mask_dir}")
            
        for city in os.listdir(img_dir):
            city_img_dir = os.path.join(img_dir, city)
            city_mask_dir = os.path.join(mask_dir, city)
            
            if not os.path.isdir(city_img_dir):
                continue
                
            for img_name in os.listdir(city_img_dir):
                if img_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(city_img_dir, img_name)
                    mask_name = img_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    mask_path = os.path.join(city_mask_dir, mask_name)
                    
                    if os.path.exists(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)
                        
        print(f"Loaded {len(self.images)} images for {self.split} split")
    
    def _convert_label(self, mask):
        """Convert original label IDs to training IDs."""
        mask_copy = np.array(mask, dtype=np.int32)
        converted = np.full_like(mask_copy, 255)
        for k, v in CITYSCAPES_ID_TO_TRAINID.items():
            converted[mask_copy == k] = v
        return converted
    
    def __getitem__(self, index):
        """Get image and mask at index."""
        # Load image
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        
        # Resize
        if self.img_size is not None:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        # Convert mask label IDs
        mask = self._convert_label(mask)
        
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
            
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(mask).long()
            
        return img, mask, index
    
    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def decode_target(mask):
        """Decode segmentation mask to RGB image for visualization."""
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for label_id, color in enumerate(CITYSCAPES_PALETTE):
            rgb[mask == label_id] = color
        return rgb


class SplitCityscapes(Cityscapes):
    """Split Cityscapes dataset for class-incremental learning.
    
    Uses ALL images but only keeps labels for current task classes.
    Other class labels are handled based on the mode setting.
    
    Args:
        root: Path to Cityscapes data root
        split: 'train' or 'val'
        transform: Transform to apply to images
        target_transform: Transform to apply to masks
        selected_labels: List of class IDs for current task (only these labels are kept)
        img_size: Size to resize images to
        old_labels: List of old class IDs (from previous tasks)
        mode: Label handling mode:
            - 'current_only': Only keep current task labels, all others become 255 (ignore)
            - 'current_and_old': Keep current + old task labels, future become 255
            - 'unknown_as_background': Current + old labels kept, unknown/future become background class (19)
            - 'all_unknown_as_background': Only current labels kept, all others become background class (19)
        background_class: Index for background class (default: 19)
    """
    
    def __init__(
        self,
        root,
        split='train',
        transform=None,
        target_transform=None,
        selected_labels=[0],
        img_size=(512, 1024),
        old_labels=None,
        mode='current_only',
        background_class=BACKGROUND_CLASS
    ):
        self.selected_labels = selected_labels  # Current task classes
        self.old_labels = old_labels if old_labels is not None else []  # Previous task classes
        self.mode = mode
        self.background_class = background_class
        
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            img_size=img_size
        )
        
        # Update n_classes if using background class
        if 'background' in self.mode:
            self.n_classes = 20  # 19 original + 1 background
        
        print(f"SplitCityscapes: {len(self.images)} images, "
              f"current_labels={self.selected_labels}, old_labels={self.old_labels}, "
              f"mode={self.mode}, n_classes={self.n_classes}")
        
    def _mask_labels(self, mask):
        """Process mask to only keep allowed labels based on mode.
        
        Args:
            mask: numpy array of labels (0-18 for Cityscapes, 255 for original ignore)
            
        Returns:
            processed_mask: mask with labels processed according to mode
        """
        if self.mode == 'current_only':
            # Only keep current task labels, everything else becomes ignore (255)
            processed_mask = np.full_like(mask, self.ignore_index)
            for label in self.selected_labels:
                processed_mask[mask == label] = label
                
        elif self.mode == 'current_and_old':
            # Keep current + old labels, future/unknown labels become ignore (255)
            processed_mask = np.full_like(mask, self.ignore_index)
            allowed_labels = self.selected_labels + self.old_labels
            for label in allowed_labels:
                processed_mask[mask == label] = label
                
        elif self.mode == 'unknown_as_background':
            # Current + old labels kept, unknown/future classes become background (19)
            # Original ignore (255) pixels remain as ignore
            processed_mask = np.full_like(mask, self.background_class)  # Default to background
            
            # Keep current task labels
            for label in self.selected_labels:
                processed_mask[mask == label] = label
            
            # Keep old task labels
            for label in self.old_labels:
                processed_mask[mask == label] = label
            
            # Original ignore pixels (255 from Cityscapes) stay as ignore
            processed_mask[mask == 255] = self.ignore_index
                
        elif self.mode == 'all_unknown_as_background':
            # Only current labels kept, ALL others (including old) become background (19)
            # This is useful for strict class-incremental where old classes are "forgotten"
            processed_mask = np.full_like(mask, self.background_class)  # Default to background
            
            # Only keep current task labels
            for label in self.selected_labels:
                processed_mask[mask == label] = label
            
            # Original ignore pixels (255 from Cityscapes) stay as ignore
            processed_mask[mask == 255] = self.ignore_index
                
        else:
            raise ValueError(f"Unknown mode: {self.mode}. "
                           f"Available: current_only, current_and_old, unknown_as_background, all_unknown_as_background")
            
        return processed_mask
        
    def __getitem__(self, index):
        """Get image and mask at index with label filtering."""
        # Load image
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        
        # Resize
        if self.img_size is not None:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        # Convert original Cityscapes label IDs to training IDs (0-18)
        mask = self._convert_label(mask)
        
        # Apply class-incremental label masking
        # Only keep labels for current task, others become 255 (ignore)
        mask = self._mask_labels(mask)
        
        # Apply transforms
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
        """Return converted train-id mask before incremental label masking.

        This keeps the dataset's native ignore pixels (255) and is useful when
        we need to distinguish original ignore regions from task-masked regions.
        """
        mask = Image.open(self.masks[index])
        if self.img_size is not None:
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        mask = self._convert_label(mask)
        return torch.from_numpy(mask).long()


class BlurryCityscapes(Cityscapes):
    """Cityscapes dataset with blurry task boundaries for online CL.
    
    Implements gradual transition between tasks by mixing samples.
    
    Args:
        root: Path to Cityscapes data root
        labels_order: Order of class labels for incremental learning
        split: 'train' or 'val'
        transform: Transform to apply to images
        n_tasks: Number of incremental tasks
        scale: Scale parameter for blurry boundaries (higher = more mixing)
        img_size: Size to resize images to
    """
    
    def __init__(
        self,
        root,
        labels_order,
        split='train',
        transform=None,
        n_tasks=5,
        scale=500,
        img_size=(512, 1024)
    ):
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            img_size=img_size
        )
        self.labels_order = labels_order
        self.n_tasks = n_tasks
        self.scale = scale
        
        # Reorder dataset based on class dominance
        self._reorder_dataset()
        
    def _get_dominant_class(self, mask_path):
        """Get the dominant (most frequent) class in a mask."""
        mask = Image.open(mask_path)
        if self.img_size is not None:
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        mask = self._convert_label(mask)
        
        # Count pixels per class
        unique, counts = np.unique(mask[mask != 255], return_counts=True)
        if len(unique) == 0:
            return -1
        return unique[np.argmax(counts)]
    
    def _reorder_dataset(self):
        """Reorder dataset for blurry task boundaries."""
        # Group images by dominant class
        class_to_images = {i: [] for i in range(self.n_classes)}
        class_to_masks = {i: [] for i in range(self.n_classes)}
        
        for img_path, mask_path in zip(self.images, self.masks):
            dominant = self._get_dominant_class(mask_path)
            if dominant >= 0:
                class_to_images[dominant].append(img_path)
                class_to_masks[dominant].append(mask_path)
        
        # Reorder based on labels_order with blurry boundaries
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
            
            # Shuffle within task
            combined = list(zip(task_images, task_masks))
            np.random.shuffle(combined)
            if combined:
                task_images, task_masks = zip(*combined)
                new_images.extend(task_images)
                new_masks.extend(task_masks)
        
        self.images = list(new_images)
        self.masks = list(new_masks)
        print(f"Reordered {len(self.images)} images for blurry CL")

