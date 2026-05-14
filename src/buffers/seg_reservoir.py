"""Reservoir buffer for semantic segmentation.

This buffer stores image-mask pairs for experience replay in
online continual semantic segmentation.
"""
import torch
import random as r
import numpy as np
import logging as lg

from src.buffers.buffer import Buffer
from src.utils.utils import get_device

device = get_device()


class SegmentationReservoir(Buffer):
    """Reservoir sampling buffer for semantic segmentation.
    
    Stores image-mask pairs and supports various retrieval strategies
    for replay-based continual learning.
    
    Args:
        max_size: Maximum buffer size (number of image-mask pairs)
        img_size: Image size as (H, W)
        nb_ch: Number of image channels
        n_classes: Number of segmentation classes
        drop_method: Method for selecting which samples to replace
    """
    
    def __init__(
        self,
        max_size=200,
        img_size=(512, 1024),
        nb_ch=3,
        n_classes=19,
        **kwargs
    ):
        # Don't call parent __init__ as we need custom buffer shapes
        super(Buffer, self).__init__()
        
        self.max_size = max_size
        self.n_classes = n_classes
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size * 2)
        self.nb_ch = nb_ch
        self.n_seen_so_far = 0
        self.n_added_so_far = 0
        self.device = get_device()
        self.drop_method = kwargs.get('drop_method', 'random')
        
        # Initialize buffers for images and masks
        self.register_buffer(
            'buffer_imgs',
            torch.FloatTensor(max_size, nb_ch, self.img_size[0], self.img_size[1]).fill_(0)
        )
        self.register_buffer(
            'buffer_masks',
            torch.LongTensor(max_size, self.img_size[0], self.img_size[1]).fill_(255)
        )
        
        # Optional: store class distribution info per sample
        self.register_buffer(
            'buffer_class_counts',
            torch.LongTensor(max_size, n_classes).fill_(0)
        )
        
    def reset(self):
        """Reset buffer state."""
        self.n_seen_so_far = 0
        self.n_added_so_far = 0
        self.buffer_imgs.fill_(0)
        self.buffer_masks.fill_(255)
        self.buffer_class_counts.fill_(0)
        
    def _compute_class_counts(self, mask):
        """Compute per-class pixel counts in a mask."""
        counts = torch.zeros(self.n_classes, dtype=torch.long)
        for c in range(self.n_classes):
            counts[c] = (mask == c).sum()
        return counts
    
    def update(self, imgs, masks, **kwargs):
        """Update buffer with new image-mask pairs using reservoir sampling.
        
        Args:
            imgs: Tensor of images (B, C, H, W)
            masks: Tensor of segmentation masks (B, H, W)
        """
        for img, mask in zip(imgs, masks):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
            
            if self.n_seen_so_far < self.max_size:
                reservoir_idx = self.n_added_so_far
                
            if reservoir_idx < self.max_size:
                if self.drop_method == 'random':
                    self.replace_data(reservoir_idx, img, mask)
                elif self.drop_method == 'class_balanced':
                    self._class_balanced_replace(img, mask)
                else:
                    self.replace_data(reservoir_idx, img, mask)
                    
            self.n_seen_so_far += 1
    
    def replace_data(self, idx, img, mask):
        """Replace data at given index."""
        self.buffer_imgs[idx] = img
        self.buffer_masks[idx] = mask
        self.buffer_class_counts[idx] = self._compute_class_counts(mask)
        self.n_added_so_far = max(self.n_added_so_far, idx + 1)
        
    def _class_balanced_replace(self, img, mask):
        """Replace sample to maintain class balance."""
        class_counts = self._compute_class_counts(mask)
        new_dominant_class = class_counts.argmax().item()
        
        if self.n_added_so_far < self.max_size:
            self.replace_data(self.n_added_so_far, img, mask)
        else:
            # Find sample with most pixels of over-represented class
            total_counts = self.buffer_class_counts[:self.n_added_so_far].sum(dim=0)
            most_common_class = total_counts.argmax().item()
            
            # Find sample with most pixels of that class
            class_pixels = self.buffer_class_counts[:self.n_added_so_far, most_common_class]
            replace_idx = class_pixels.argmax().item()
            
            self.replace_data(replace_idx, img, mask)
    
    def random_retrieve(self, n_imgs=100):
        """Randomly retrieve image-mask pairs from buffer.
        
        Args:
            n_imgs: Number of pairs to retrieve
            
        Returns:
            imgs: Retrieved images
            masks: Retrieved masks
        """
        if self.n_added_so_far < n_imgs:
            lg.debug(f"Requested {n_imgs} but only {self.n_added_so_far} available")
            return (
                self.buffer_imgs[:self.n_added_so_far],
                self.buffer_masks[:self.n_added_so_far]
            )
        
        indices = r.sample(range(min(self.n_added_so_far, self.max_size)), n_imgs)
        return self.buffer_imgs[indices], self.buffer_masks[indices]
    
    def class_balanced_retrieve(self, n_imgs=100, target_classes=None):
        """Retrieve samples balanced across classes.
        
        Args:
            n_imgs: Number of pairs to retrieve
            target_classes: List of classes to focus on (None for all)
            
        Returns:
            imgs: Retrieved images
            masks: Retrieved masks
        """
        if self.n_added_so_far == 0:
            return torch.Tensor(), torch.Tensor()
            
        if target_classes is None:
            target_classes = list(range(self.n_classes))
            
        # Group samples by dominant class
        class_to_indices = {c: [] for c in target_classes}
        for i in range(min(self.n_added_so_far, self.max_size)):
            dominant = self.buffer_class_counts[i].argmax().item()
            if dominant in class_to_indices:
                class_to_indices[dominant].append(i)
        
        # Sample equally from each class
        selected_indices = []
        per_class = max(1, n_imgs // len(target_classes))
        
        for c in target_classes:
            if len(class_to_indices[c]) > 0:
                n_select = min(per_class, len(class_to_indices[c]))
                selected_indices.extend(
                    r.sample(class_to_indices[c], n_select)
                )
        
        # Fill remaining slots randomly
        while len(selected_indices) < n_imgs and len(selected_indices) < self.n_added_so_far:
            idx = r.randint(0, min(self.n_added_so_far, self.max_size) - 1)
            if idx not in selected_indices:
                selected_indices.append(idx)
        
        return self.buffer_imgs[selected_indices], self.buffer_masks[selected_indices]
    
    def get_all(self):
        """Get all stored image-mask pairs."""
        n = min(self.n_added_so_far, self.max_size)
        return self.buffer_imgs[:n], self.buffer_masks[:n]
    
    def get_class_distribution(self):
        """Get distribution of classes in the buffer."""
        if self.n_added_so_far == 0:
            return torch.zeros(self.n_classes)
            
        total_counts = self.buffer_class_counts[:self.n_added_so_far].sum(dim=0).float()
        return total_counts / total_counts.sum()
    
    def contains_class(self, class_id):
        """Check if buffer contains samples with given class.
        
        Args:
            class_id: Class ID to check
            
        Returns:
            bool: True if buffer contains at least one sample with the class
        """
        if self.n_added_so_far == 0:
            return False
        return self.buffer_class_counts[:self.n_added_so_far, class_id].sum() > 0
    
    def retrieve_with_class(self, class_id, n_imgs=100):
        """Retrieve samples containing a specific class.
        
        Args:
            class_id: Class ID to retrieve
            n_imgs: Maximum number of samples
            
        Returns:
            imgs: Retrieved images
            masks: Retrieved masks
        """
        valid_indices = []
        for i in range(min(self.n_added_so_far, self.max_size)):
            if self.buffer_class_counts[i, class_id] > 0:
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return torch.Tensor(), torch.Tensor()
            
        n_select = min(n_imgs, len(valid_indices))
        selected = r.sample(valid_indices, n_select)
        
        return self.buffer_imgs[selected], self.buffer_masks[selected]


class SegmentationLogitsReservoir(SegmentationReservoir):
    """Extended reservoir that also stores model logits for knowledge distillation.
    
    Args:
        max_size: Maximum buffer size
        img_size: Image size as (H, W)
        nb_ch: Number of image channels
        n_classes: Number of segmentation classes
    """
    
    def __init__(
        self,
        max_size=200,
        img_size=(512, 1024),
        nb_ch=3,
        n_classes=19,
        **kwargs
    ):
        super().__init__(
            max_size=max_size,
            img_size=img_size,
            nb_ch=nb_ch,
            n_classes=n_classes,
            **kwargs
        )
        
        # Store logits for knowledge distillation
        # Use smaller spatial resolution to save memory
        logit_size = (img_size[0] // 4, img_size[1] // 4)
        self.register_buffer(
            'buffer_logits',
            torch.FloatTensor(max_size, n_classes, logit_size[0], logit_size[1]).fill_(0)
        )
        self.logit_size = logit_size
        
    def update(self, imgs, masks, logits=None, **kwargs):
        """Update buffer with image-mask-logits triples.
        
        Args:
            imgs: Tensor of images (B, C, H, W)
            masks: Tensor of segmentation masks (B, H, W)
            logits: Tensor of model logits (B, n_classes, H, W)
        """
        if logits is not None:
            # Downsample logits to save memory
            logits = torch.nn.functional.interpolate(
                logits.detach(),
                size=self.logit_size,
                mode='bilinear',
                align_corners=False
            )
        
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
            
            if self.n_seen_so_far < self.max_size:
                reservoir_idx = self.n_added_so_far
                
            if reservoir_idx < self.max_size:
                self.buffer_imgs[reservoir_idx] = img
                self.buffer_masks[reservoir_idx] = mask
                self.buffer_class_counts[reservoir_idx] = self._compute_class_counts(mask)
                if logits is not None:
                    self.buffer_logits[reservoir_idx] = logits[i]
                self.n_added_so_far = max(self.n_added_so_far, reservoir_idx + 1)
                
            self.n_seen_so_far += 1
    
    def random_retrieve_with_logits(self, n_imgs=100):
        """Retrieve image-mask-logits triples.
        
        Args:
            n_imgs: Number of triples to retrieve
            
        Returns:
            imgs, masks, logits: Retrieved data
        """
        if self.n_added_so_far < n_imgs:
            return (
                self.buffer_imgs[:self.n_added_so_far],
                self.buffer_masks[:self.n_added_so_far],
                self.buffer_logits[:self.n_added_so_far]
            )
        
        indices = r.sample(range(min(self.n_added_so_far, self.max_size)), n_imgs)
        return (
            self.buffer_imgs[indices],
            self.buffer_masks[indices],
            self.buffer_logits[indices]
        )

