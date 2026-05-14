"""Data augmentation utilities for semantic segmentation.

Provides augmentation transforms that apply consistently to both
images and segmentation masks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchvision import transforms
from torchvision.transforms import functional as TF


class SegmentationAugmentation(nn.Module):
    """Augmentation pipeline for semantic segmentation.
    
    Applies the same geometric transforms to both image and mask,
    and color transforms only to the image.
    
    Args:
        img_size: Target image size (H, W)
        scale: Scale range for random resized crop
        hflip_p: Probability of horizontal flip
        color_jitter: Color jitter parameters
    """
    
    def __init__(
        self,
        img_size=(512, 1024),
        scale=(0.5, 2.0),
        hflip_p=0.5,
        color_jitter=(0.3, 0.3, 0.3, 0.1),
        p_color=0.5
    ):
        super().__init__()
        self.img_size = img_size
        self.scale = scale
        self.hflip_p = hflip_p
        self.color_jitter = color_jitter
        self.p_color = p_color
        
    def forward(self, img, mask):
        """Apply augmentations to image and mask.
        
        Args:
            img: Image tensor (C, H, W)
            mask: Mask tensor (H, W)
            
        Returns:
            aug_img: Augmented image
            aug_mask: Augmented mask
        """
        # Random horizontal flip
        if random.random() < self.hflip_p:
            img = TF.hflip(img)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        
        # Random scale
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        new_h = int(self.img_size[0] * scale_factor)
        new_w = int(self.img_size[1] * scale_factor)
        
        img = F.interpolate(
            img.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(new_h, new_w),
            mode='nearest'
        ).squeeze().long()
        
        # Random crop to target size
        if new_h > self.img_size[0] and new_w > self.img_size[1]:
            top = random.randint(0, new_h - self.img_size[0])
            left = random.randint(0, new_w - self.img_size[1])
            
            img = img[:, top:top + self.img_size[0], left:left + self.img_size[1]]
            mask = mask[top:top + self.img_size[0], left:left + self.img_size[1]]
        else:
            # Pad if smaller than target
            pad_h = max(0, self.img_size[0] - new_h)
            pad_w = max(0, self.img_size[1] - new_w)
            
            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (0, pad_w, 0, pad_h), value=0)
                mask = F.pad(mask.unsqueeze(0), (0, pad_w, 0, pad_h), value=255).squeeze(0)
            
            # Random crop
            h, w = img.shape[-2:]
            top = random.randint(0, max(0, h - self.img_size[0]))
            left = random.randint(0, max(0, w - self.img_size[1]))
            
            img = img[:, top:top + self.img_size[0], left:left + self.img_size[1]]
            mask = mask[top:top + self.img_size[0], left:left + self.img_size[1]]
        
        # Color jitter (image only)
        if random.random() < self.p_color and self.color_jitter is not None:
            brightness, contrast, saturation, hue = self.color_jitter
            
            if brightness > 0:
                factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
                img = TF.adjust_brightness(img, factor)
            
            if contrast > 0:
                factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
                img = TF.adjust_contrast(img, factor)
            
            if saturation > 0:
                factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
                img = TF.adjust_saturation(img, factor)
            
            if hue > 0:
                factor = random.uniform(-hue, hue)
                img = TF.adjust_hue(img, factor)
        
        return img, mask


class BatchSegmentationAugmentation(nn.Module):
    """Batch-level augmentation for segmentation.
    
    Args:
        img_size: Target image size (H, W)
    """
    
    def __init__(self, img_size=(512, 1024)):
        super().__init__()
        self.img_size = img_size
        self.aug = SegmentationAugmentation(img_size=img_size)
        
    def forward(self, imgs, masks):
        """Apply augmentation to batch.
        
        Args:
            imgs: Batch of images (B, C, H, W)
            masks: Batch of masks (B, H, W)
            
        Returns:
            aug_imgs: Augmented images
            aug_masks: Augmented masks
        """
        aug_imgs = []
        aug_masks = []
        
        for img, mask in zip(imgs, masks):
            aug_img, aug_mask = self.aug(img, mask)
            aug_imgs.append(aug_img)
            aug_masks.append(aug_mask)
        
        return torch.stack(aug_imgs), torch.stack(aug_masks)


class PhotoMetricDistortion(nn.Module):
    """Photo-metric distortion for segmentation.
    
    Applies random brightness, contrast, saturation, and hue changes.
    """
    
    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ):
        super().__init__()
        self.brightness_delta = brightness_delta / 255.0
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta / 360.0
        
    def forward(self, img):
        """Apply photo-metric distortion.
        
        Args:
            img: Image tensor (C, H, W) in [0, 1]
            
        Returns:
            distorted: Distorted image
        """
        # Random brightness
        if random.random() < 0.5:
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img = img + delta
            img = torch.clamp(img, 0, 1)
        
        # Mode selection: contrast first or saturation/hue first
        mode = random.randint(0, 1)
        
        if mode == 0:
            # Contrast first
            if random.random() < 0.5:
                factor = random.uniform(*self.contrast_range)
                img = TF.adjust_contrast(img, factor)
            
            # Convert to HSV for saturation and hue
            if random.random() < 0.5:
                factor = random.uniform(*self.saturation_range)
                img = TF.adjust_saturation(img, factor)
            
            if random.random() < 0.5:
                delta = random.uniform(-self.hue_delta, self.hue_delta)
                img = TF.adjust_hue(img, delta)
        else:
            # Saturation and hue first
            if random.random() < 0.5:
                factor = random.uniform(*self.saturation_range)
                img = TF.adjust_saturation(img, factor)
            
            if random.random() < 0.5:
                delta = random.uniform(-self.hue_delta, self.hue_delta)
                img = TF.adjust_hue(img, delta)
            
            if random.random() < 0.5:
                factor = random.uniform(*self.contrast_range)
                img = TF.adjust_contrast(img, factor)
        
        return torch.clamp(img, 0, 1)


class CutMixSegmentation(nn.Module):
    """CutMix augmentation for semantic segmentation.
    
    Mixes regions from two image-mask pairs.
    
    Args:
        alpha: Beta distribution parameter
        p: Probability of applying CutMix
    """
    
    def __init__(self, alpha=1.0, p=0.5):
        super().__init__()
        self.alpha = alpha
        self.p = p
        
    def forward(self, imgs, masks):
        """Apply CutMix to batch.
        
        Args:
            imgs: Batch of images (B, C, H, W)
            masks: Batch of masks (B, H, W)
            
        Returns:
            mixed_imgs: Mixed images
            mixed_masks: Mixed masks
        """
        if random.random() > self.p:
            return imgs, masks
        
        batch_size = imgs.size(0)
        
        # Generate random permutation
        indices = torch.randperm(batch_size)
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get bounding box
        _, _, h, w = imgs.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        mixed_imgs = imgs.clone()
        mixed_masks = masks.clone()
        
        mixed_imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[indices, :, bby1:bby2, bbx1:bbx2]
        mixed_masks[:, bby1:bby2, bbx1:bbx2] = masks[indices, bby1:bby2, bbx1:bbx2]
        
        return mixed_imgs, mixed_masks


class ClassMixSegmentation(nn.Module):
    """ClassMix augmentation for semantic segmentation.
    
    Mixes regions based on class masks rather than random boxes.
    
    Args:
        n_classes: Number of segmentation classes
        p: Probability of applying ClassMix
    """
    
    def __init__(self, n_classes=19, p=0.5, ignore_index=255):
        super().__init__()
        self.n_classes = n_classes
        self.p = p
        self.ignore_index = ignore_index
        
    def forward(self, imgs, masks):
        """Apply ClassMix to batch.
        
        Args:
            imgs: Batch of images (B, C, H, W)
            masks: Batch of masks (B, H, W)
            
        Returns:
            mixed_imgs: Mixed images
            mixed_masks: Mixed masks
        """
        if random.random() > self.p:
            return imgs, masks
        
        batch_size = imgs.size(0)
        
        # Generate random permutation
        indices = torch.randperm(batch_size)
        
        mixed_imgs = imgs.clone()
        mixed_masks = masks.clone()
        
        for i in range(batch_size):
            j = indices[i]
            
            # Get classes present in mask j
            classes_j = torch.unique(masks[j])
            classes_j = classes_j[classes_j != self.ignore_index]
            
            if len(classes_j) == 0:
                continue
            
            # Randomly select half of the classes
            n_select = max(1, len(classes_j) // 2)
            selected = classes_j[torch.randperm(len(classes_j))[:n_select]]
            
            # Create mix mask
            mix_mask = torch.zeros_like(masks[j], dtype=torch.bool)
            for c in selected:
                mix_mask = mix_mask | (masks[j] == c)
            
            # Apply mixing
            mixed_imgs[i] = torch.where(
                mix_mask.unsqueeze(0),
                imgs[j],
                imgs[i]
            )
            mixed_masks[i] = torch.where(
                mix_mask,
                masks[j],
                masks[i]
            )
        
        return mixed_imgs, mixed_masks

