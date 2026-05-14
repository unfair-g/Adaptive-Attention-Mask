#!/usr/bin/env python3
"""
Reorganize Cityscapes data from flat structure to standard Cityscapes format.

Input format:
    ./data/cityscapes/train/
        aachen_000000_000019_leftImg8bit.png
        aachen_000000_000019_gtFine_polygons.json (or labelIds.png)
        ...

Output format:
    ./data/cityscapes/
        leftImg8bit/
            train/
                aachen/
                    aachen_000000_000019_leftImg8bit.png
                    ...
        gtFine/
            train/
                aachen/
                    aachen_000000_000019_gtFine_labelIds.png
                    ...

Usage:
    python reorganize_cityscapes_data.py --source-dir ./data/cityscapes/train --target-root ./data/cityscapes --split train
"""

import os
import shutil
import json
from pathlib import Path
from PIL import Image
import numpy as np
import sys

def extract_city_name(filename):
    """Extract city name from Cityscapes filename.
    
    Format: {city}_{sequence}_{frame}_{type}.{ext}
    Example: aachen_000000_000019_leftImg8bit.png -> aachen
    """
    parts = filename.split('_')
    if len(parts) >= 1:
        return parts[0]
    return 'unknown'

def reorganize_cityscapes_data(source_dir, target_root, split='train'):
    """
    Reorganize Cityscapes data from flat structure to standard format.
    
    Args:
        source_dir: Source directory containing flat file structure
                    (e.g., ./data/cityscapes/train/)
        target_root: Target root directory (e.g., ./data/cityscapes/)
        split: 'train', 'val', or 'test'
    """
    source_dir = Path(source_dir)
    target_root = Path(target_root)
    
    if not source_dir.exists():
        raise ValueError(f"Source directory not found: {source_dir}")
    
    # Create target directories
    img_target_dir = target_root / 'leftImg8bit' / split
    mask_target_dir = target_root / 'gtFine' / split
    img_target_dir.mkdir(parents=True, exist_ok=True)
    mask_target_dir.mkdir(parents=True, exist_ok=True)
    
    # Track files by city
    city_files = {}
    
    # Scan source directory
    print(f"Scanning source directory: {source_dir}")
    files = list(source_dir.glob('*'))
    print(f"Found {len(files)} files")
    
    for file_path in files:
        if not file_path.is_file():
            continue
            
        filename = file_path.name
        
        # Skip non-image and non-annotation files
        if not (filename.endswith('.png') or filename.endswith('.json')):
            continue
        
        # Extract city name
        city_name = extract_city_name(filename)
        
        if city_name not in city_files:
            city_files[city_name] = {'images': [], 'masks': []}
        
        # Categorize files
        if filename.endswith('_leftImg8bit.png'):
            city_files[city_name]['images'].append(file_path)
        elif filename.endswith('_gtFine_polygons.json'):
            # Note: We need labelIds, but user has polygons.json
            # We'll create a placeholder or try to convert
            city_files[city_name]['masks'].append(file_path)
        elif filename.endswith('_gtFine_labelIds.png'):
            city_files[city_name]['masks'].append(file_path)
    
    # Create city directories and move files
    print(f"\nFound {len(city_files)} cities")
    
    for city_name, files_dict in city_files.items():
        print(f"\nProcessing city: {city_name}")
        
        # Create city directories
        city_img_dir = img_target_dir / city_name
        city_mask_dir = mask_target_dir / city_name
        city_img_dir.mkdir(exist_ok=True)
        city_mask_dir.mkdir(exist_ok=True)
        
        # Move images
        for img_path in files_dict['images']:
            target_path = city_img_dir / img_path.name
            if not target_path.exists():
                shutil.copy2(img_path, target_path)
                print(f"  Copied image: {img_path.name}")
        
        # Handle masks
        for mask_path in files_dict['masks']:
            if mask_path.suffix == '.json':
                # Convert polygons.json to labelIds.png if possible
                # For now, we'll skip JSON files and create a note
                print(f"  Warning: Found polygons.json file: {mask_path.name}")
                print(f"    You may need to convert this to labelIds.png format")
                print(f"    Or use a different dataset loader that supports polygons.json")
            else:
                # Move labelIds.png files
                target_path = city_mask_dir / mask_path.name
                if not target_path.exists():
                    shutil.copy2(mask_path, target_path)
                    print(f"  Copied mask: {mask_path.name}")
    
    print(f"\n{'='*60}")
    print(f"Reorganization complete!")
    print(f"Images: {img_target_dir}")
    print(f"Masks: {mask_target_dir}")
    print(f"{'='*60}")

def convert_polygons_to_labelids(json_path, output_path, cityscapes_id_to_trainid):
    """
    Convert Cityscapes polygons.json to labelIds.png.
    
    Note: This is a simplified version. Full conversion requires
    proper polygon rendering which is complex.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get image dimensions
        img_width = data.get('imgWidth', 2048)
        img_height = data.get('imgHeight', 1024)
        
        # Create label image
        label_img = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Process objects
        for obj in data.get('objects', []):
            label = obj.get('label', '')
            polygon = obj.get('polygon', [])
            
            # Map label to ID (simplified - you may need full mapping)
            # This is a placeholder - actual conversion is more complex
            pass
        
        # Save as PNG
        Image.fromarray(label_img).save(output_path)
        return True
    except Exception as e:
        print(f"Error converting {json_path}: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reorganize Cityscapes data from flat structure to standard format'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default='./data/cityscapes/train',
        help='Source directory with flat file structure'
    )
    parser.add_argument(
        '--target-root',
        type=str,
        default='./data/cityscapes',
        help='Target root directory (will create leftImg8bit and gtFine subdirectories)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving (default: copy)'
    )
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"Cityscapes Data Reorganization")
    print(f"{'='*60}")
    print(f"Source: {args.source_dir}")
    print(f"Target: {args.target_root}")
    print(f"Split: {args.split}")
    print(f"Mode: {'Copy' if args.copy else 'Move'}")
    print(f"{'='*60}\n")
    
    reorganize_cityscapes_data(args.source_dir, args.target_root, args.split)
    
    print(f"\nNote: If you have polygons.json files, you may need to convert them")
    print(f"      to labelIds.png format. Check Cityscapes dataset documentation")
    print(f"      for conversion tools.")

if __name__ == '__main__':
    main()

