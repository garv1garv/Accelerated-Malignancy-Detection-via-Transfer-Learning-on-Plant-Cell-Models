"""
Synthetic skin lesion data generator.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️

This script generates synthetic skin-like images for demonstration purposes only.
The generated images are procedurally created and do not represent real skin lesions.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import cv2
from typing import Tuple, List
import random
from tqdm import tqdm


def create_skin_background(size: Tuple[int, int], skin_tone: str = "light") -> np.ndarray:
    """
    Create a synthetic skin background.
    
    Args:
        size: Image size (height, width)
        skin_tone: Skin tone ('light', 'medium', 'dark')
    
    Returns:
        Skin background image
    """
    # Define skin tone colors (RGB)
    skin_tones = {
        'light': [(255, 220, 177), (255, 205, 148), (255, 190, 120)],
        'medium': [(255, 173, 96), (255, 158, 68), (255, 143, 40)],
        'dark': [(141, 85, 36), (121, 65, 16), (101, 45, 0)]
    }
    
    colors = skin_tones[skin_tone]
    
    # Create base image
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Fill with base skin color
    base_color = random.choice(colors)
    img[:, :] = base_color
    
    # Add texture variations
    for _ in range(50):
        x = random.randint(0, size[1] - 1)
        y = random.randint(0, size[0] - 1)
        radius = random.randint(5, 20)
        color_variation = random.randint(-20, 20)
        
        color = tuple(max(0, min(255, c + color_variation)) for c in base_color)
        
        cv2.circle(img, (x, y), radius, color, -1)
    
    # Add noise for texture
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = np.clip(img + noise, 0, 255)
    
    return img


def create_benign_lesion(size: Tuple[int, int]) -> np.ndarray:
    """
    Create a synthetic benign lesion.
    
    Args:
        size: Image size (height, width)
    
    Returns:
        Benign lesion image
    """
    # Create skin background
    img = create_skin_background(size)
    
    # Add benign lesion characteristics
    center_x, center_y = size[1] // 2, size[0] // 2
    
    # Main lesion area (smooth, regular border)
    radius = random.randint(20, 40)
    color = (random.randint(180, 220), random.randint(150, 190), random.randint(120, 160))
    
    # Create smooth circular lesion
    mask = np.zeros(size[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Apply Gaussian blur for smooth edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    # Blend lesion with background
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - mask / 255.0) + color[c] * (mask / 255.0)
    
    # Add some texture variation within the lesion
    for _ in range(10):
        x = random.randint(center_x - radius, center_x + radius)
        y = random.randint(center_y - radius, center_y + radius)
        if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
            small_radius = random.randint(2, 8)
            color_variation = random.randint(-15, 15)
            lesion_color = tuple(max(0, min(255, c + color_variation)) for c in color)
            cv2.circle(img, (x, y), small_radius, lesion_color, -1)
    
    return img.astype(np.uint8)


def create_malignant_lesion(size: Tuple[int, int]) -> np.ndarray:
    """
    Create a synthetic malignant lesion.
    
    Args:
        size: Image size (height, width)
    
    Returns:
        Malignant lesion image
    """
    # Create skin background
    img = create_skin_background(size)
    
    # Add malignant lesion characteristics
    center_x, center_y = size[1] // 2, size[0] // 2
    
    # Irregular shape with multiple components
    num_components = random.randint(2, 4)
    components = []
    
    for i in range(num_components):
        # Random offset from center
        offset_x = random.randint(-30, 30)
        offset_y = random.randint(-30, 30)
        radius = random.randint(15, 35)
        
        components.append({
            'center': (center_x + offset_x, center_y + offset_y),
            'radius': radius
        })
    
    # Create irregular mask
    mask = np.zeros(size[:2], dtype=np.uint8)
    
    for comp in components:
        cv2.circle(mask, comp['center'], comp['radius'], 255, -1)
    
    # Add irregularity by distorting the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    
    # Darker, more irregular color
    color = (random.randint(80, 140), random.randint(60, 120), random.randint(40, 100))
    
    # Blend lesion with background
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - mask / 255.0) + color[c] * (mask / 255.0)
    
    # Add irregular texture and color variations
    for _ in range(20):
        # Random position within lesion area
        x = random.randint(0, size[1] - 1)
        y = random.randint(0, size[0] - 1)
        
        if mask[y, x] > 0:
            small_radius = random.randint(1, 6)
            color_variation = random.randint(-30, 30)
            lesion_color = tuple(max(0, min(255, c + color_variation)) for c in color)
            cv2.circle(img, (x, y), small_radius, lesion_color, -1)
    
    # Add some border irregularity
    for _ in range(15):
        x = random.randint(0, size[1] - 1)
        y = random.randint(0, size[0] - 1)
        
        if 50 < mask[y, x] < 200:  # Border area
            radius = random.randint(1, 4)
            border_color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in color)
            cv2.circle(img, (x, y), radius, border_color, -1)
    
    return img.astype(np.uint8)


def generate_synthetic_dataset(
    data_root: str = "data/synthetic",
    num_samples: int = 120,
    image_size: int = 128,
    seed: int = 42
):
    """
    Generate synthetic skin lesion dataset.
    
    Args:
        data_root: Root directory for the dataset
        num_samples: Total number of samples to generate
        image_size: Size of generated images (square)
        seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print(f"Generating synthetic skin lesion dataset...")
    print(f"  Total samples: {num_samples}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Output directory: {data_root}")
    
    # Create directory structure
    splits = ['train', 'val', 'test']
    classes = ['benign', 'malignant']
    
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(data_root, split, class_name), exist_ok=True)
    
    # Calculate samples per split and class
    samples_per_class = num_samples // 2
    train_samples = int(samples_per_class * 0.6)
    val_samples = int(samples_per_class * 0.2)
    test_samples = samples_per_class - train_samples - val_samples
    
    split_samples = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Train: {train_samples}, Val: {val_samples}, Test: {test_samples}")
    
    # Generate images
    metadata_records = []
    image_id = 0
    
    for class_idx, class_name in enumerate(classes):
        print(f"\nGenerating {class_name} images...")
        
        for split, num_split_samples in split_samples.items():
            print(f"  {split}: {num_split_samples} images")
            
            for i in tqdm(range(num_split_samples), desc=f"{split} {class_name}"):
                # Generate image
                if class_name == 'benign':
                    img_array = create_benign_lesion((image_size, image_size))
                else:
                    img_array = create_malignant_lesion((image_size, image_size))
                
                # Convert to PIL Image
                img = Image.fromarray(img_array)
                
                # Save image
                image_filename = f"{class_name}_{split}_{i:03d}.png"
                image_path = os.path.join(split, class_name, image_filename)
                full_image_path = os.path.join(data_root, image_path)
                
                img.save(full_image_path)
                
                # Add to metadata
                metadata_records.append({
                    'image_id': image_id,
                    'image_path': image_path,
                    'split': split,
                    'class_name': class_name,
                    'label': class_idx,
                    'image_size': image_size
                })
                
                image_id += 1
    
    # Create metadata DataFrames
    for split in splits:
        split_records = [r for r in metadata_records if r['split'] == split]
        df = pd.DataFrame(split_records)
        
        # Save metadata
        metadata_path = os.path.join(data_root, f"{split}_metadata.csv")
        df.to_csv(metadata_path, index=False)
        
        print(f"  {split} metadata saved: {len(split_records)} samples")
    
    # Print summary
    print(f"\n✅ Dataset generation completed!")
    print(f"  Total images: {len(metadata_records)}")
    print(f"  Directory: {data_root}")
    
    # Print class distribution
    for split in splits:
        split_records = [r for r in metadata_records if r['split'] == split]
        benign_count = sum(1 for r in split_records if r['class_name'] == 'benign')
        malignant_count = sum(1 for r in split_records if r['class_name'] == 'malignant')
        print(f"  {split}: {benign_count} benign, {malignant_count} malignant")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic skin lesion dataset")
    parser.add_argument("--data-root", default="data/synthetic", help="Data root directory")
    parser.add_argument("--num-samples", type=int, default=120, help="Total number of samples")
    parser.add_argument("--image-size", type=int, default=128, help="Image size (square)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        data_root=args.data_root,
        num_samples=args.num_samples,
        image_size=args.image_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
