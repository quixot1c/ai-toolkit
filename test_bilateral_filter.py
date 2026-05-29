#!/usr/bin/env python3
"""Minimal bilateral filter test script on folder of images."""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Add toolkit to path
sys.path.insert(0, str(Path(__file__).parent))
from toolkit.image_utils import bilateral_filter


def test_bilateral_filter(input_dir, output_dir, diameter=5, sigma_color=0.1, sigma_space=0.1):
    """Apply bilateral filter to all images in input_dir and save to output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    
    for img_file in image_files:
        try:
            # Load image
            img = Image.open(img_file).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Convert to tensor (1, 3, H, W)
            tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            
            # Apply bilateral filter
            filtered = bilateral_filter(tensor, diameter, sigma_color, sigma_space)
            
            # Convert back to image
            filtered_array = (filtered.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            filtered_img = Image.fromarray(filtered_array)
            
            # Save output
            output_file = output_path / img_file.name
            filtered_img.save(output_file)
            print(f"✓ {img_file.name}")
        except Exception as e:
            print(f"✗ {img_file.name}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test bilateral filter on image folder')
    parser.add_argument('input_dir', help='Input image directory')
    parser.add_argument('--output', default='bilateral_output', help='Output directory (default: bilateral_output)')
    parser.add_argument('--diameter', type=int, default=5, help='Filter diameter (default: 5)')
    parser.add_argument('--sigma-color', type=float, default=0.1, help='Sigma color (default: 0.1)')
    parser.add_argument('--sigma-space', type=float, default=0.1, help='Sigma space (default: 0.1)')
    
    args = parser.parse_args()
    
    test_bilateral_filter(args.input_dir, args.output, args.diameter, args.sigma_color, args.sigma_space)
