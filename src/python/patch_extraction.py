#!/usr/bin/env python3
"""
Patch Extraction for Retinal Vessel Segmentation

This module provides utilities for extracting image patches from retinal
fundus photographs for feature extraction and analysis. It supports both
regular grid-based sampling and adaptive sampling based on vessel probability.

Author: Retinal Vessel Segmentation Research Team
Date: February 2026
Python Version: 3.7+

Dependencies:
    - opencv-python (cv2)
    - numpy
    - matplotlib (optional, for visualization)
    - scikit-image (optional, for advanced processing)

Example Usage:
    import cv2
    from patch_extraction import extract_patches, visualize_patches
    
    # Load retinal image
    image = cv2.imread('08_test.tif')
    
    # Extract patches
    patches, coords = extract_patches(image, patch_size=32, stride=16)
    
    # Visualize results
    visualize_patches(image, coords, patches[:10])
"""

import cv2
import numpy as np
import os
from typing import Tuple, List, Optional, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class PatchConfig:
    """Configuration for patch extraction parameters"""
    patch_size: int = 32
    stride: int = 16
    normalize: bool = True
    remove_black_patches: bool = True
    min_variance: float = 10.0
    adaptive_sampling: bool = False
    vessel_probability_threshold: float = 0.1


class RetinalPatchExtractor:
    """
    Advanced patch extraction for retinal vessel segmentation
    
    This class provides comprehensive patch extraction capabilities including:
    - Regular grid-based sampling
    - Adaptive sampling based on vessel likelihood
    - Multi-scale patch extraction
    - Quality filtering and normalization
    """
    
    def __init__(self, config: Optional[PatchConfig] = None):
        """
        Initialize the patch extractor
        
        Args:
            config: Patch extraction configuration
        """
        self.config = config or PatchConfig()
        
    def extract_patches(self, 
                       image: np.ndarray, 
                       mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patches from retinal image
        
        Args:
            image: Input retinal image (BGR or grayscale)
            mask: Optional mask to guide patch extraction
            
        Returns:
            patches: Array of extracted patches (N, H, W, C)
            coordinates: Patch center coordinates (N, 2)
        """
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1
            image = np.expand_dims(image, axis=2)
            
        # Calculate number of patches
        n_patches_h = (height - self.config.patch_size) // self.config.stride + 1
        n_patches_w = (width - self.config.patch_size) // self.config.stride + 1
        
        patches = []
        coordinates = []
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch coordinates
                y = i * self.config.stride
                x = j * self.config.stride
                
                # Extract patch
                patch = image[y:y+self.config.patch_size, 
                            x:x+self.config.patch_size]
                
                # Quality check
                if self._is_valid_patch(patch, mask, x, y):
                    if self.config.normalize:
                        patch = self._normalize_patch(patch)
                    
                    patches.append(patch)
                    # Store center coordinates
                    center_x = x + self.config.patch_size // 2
                    center_y = y + self.config.patch_size // 2
                    coordinates.append([center_y, center_x])
        
        return np.array(patches), np.array(coordinates)
    
    def extract_adaptive_patches(self, 
                               image: np.ndarray,
                               vessel_probability: np.ndarray,
                               n_patches: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patches using adaptive sampling based on vessel probability
        
        Args:
            image: Input retinal image
            vessel_probability: Vessel probability map
            n_patches: Number of patches to extract
            
        Returns:
            patches: Extracted patches
            coordinates: Patch coordinates
        """
        if len(image.shape) == 3:
            height, width, _ = image.shape
        else:
            height, width = image.shape
            image = np.expand_dims(image, axis=2)
        
        # Create sampling probability map
        sampling_prob = self._create_sampling_probability(vessel_probability)
        
        patches = []
        coordinates = []
        
        # Sample patch centers based on probability
        valid_coords = self._get_valid_coordinates(height, width)
        flat_prob = sampling_prob.flatten()
        flat_prob = flat_prob / np.sum(flat_prob)  # Normalize
        
        # Sample coordinates
        sampled_indices = np.random.choice(
            len(valid_coords), 
            size=min(n_patches, len(valid_coords)),
            replace=False,
            p=flat_prob[valid_coords]
        )
        
        for idx in sampled_indices:
            coord = valid_coords[idx]
            y, x = np.unravel_index(coord, (height, width))
            
            # Adjust to patch corner
            patch_y = max(0, y - self.config.patch_size // 2)
            patch_x = max(0, x - self.config.patch_size // 2)
            
            # Ensure patch doesn't go out of bounds
            patch_y = min(patch_y, height - self.config.patch_size)
            patch_x = min(patch_x, width - self.config.patch_size)
            
            # Extract patch
            patch = image[patch_y:patch_y+self.config.patch_size,
                         patch_x:patch_x+self.config.patch_size]
            
            if self.config.normalize:
                patch = self._normalize_patch(patch)
            
            patches.append(patch)
            coordinates.append([y, x])
        
        return np.array(patches), np.array(coordinates)
    
    def extract_multiscale_patches(self, 
                                 image: np.ndarray,
                                 scales: List[int] = [16, 32, 64]) -> dict:
        """
        Extract patches at multiple scales
        
        Args:
            image: Input retinal image
            scales: List of patch sizes
            
        Returns:
            Dictionary mapping scales to (patches, coordinates) tuples
        """
        original_patch_size = self.config.patch_size
        results = {}
        
        for scale in scales:
            self.config.patch_size = scale
            patches, coords = self.extract_patches(image)
            results[scale] = (patches, coords)
        
        # Restore original patch size
        self.config.patch_size = original_patch_size
        
        return results
    
    def _is_valid_patch(self, 
                       patch: np.ndarray, 
                       mask: Optional[np.ndarray], 
                       x: int, 
                       y: int) -> bool:
        """
        Check if patch meets quality criteria
        
        Args:
            patch: Image patch
            mask: Optional validity mask
            x, y: Patch coordinates
            
        Returns:
            True if patch is valid
        """
        # Check patch size
        if patch.shape[0] != self.config.patch_size or \
           patch.shape[1] != self.config.patch_size:
            return False
        
        # Check mask if provided
        if mask is not None:
            patch_mask = mask[y:y+self.config.patch_size, 
                             x:x+self.config.patch_size]
            if np.mean(patch_mask) < 0.5:  # Less than 50% valid pixels
                return False
        
        # Remove black patches (common in retinal images)
        if self.config.remove_black_patches:
            if len(patch.shape) == 3:
                gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            else:
                gray_patch = patch.squeeze()
            
            if np.mean(gray_patch) < 10:  # Very dark patch
                return False
        
        # Check variance (avoid uniform regions)
        if len(patch.shape) == 3:
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray_patch = patch.squeeze()
            
        if np.var(gray_patch) < self.config.min_variance:
            return False
        
        return True
    
    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Normalize patch intensities
        
        Args:
            patch: Input patch
            
        Returns:
            Normalized patch
        """
        patch = patch.astype(np.float32)
        
        # Per-channel normalization
        if len(patch.shape) == 3:
            for c in range(patch.shape[2]):
                channel = patch[:, :, c]
                if np.std(channel) > 0:
                    patch[:, :, c] = (channel - np.mean(channel)) / np.std(channel)
        else:
            if np.std(patch) > 0:
                patch = (patch - np.mean(patch)) / np.std(patch)
        
        return patch
    
    def _create_sampling_probability(self, vessel_probability: np.ndarray) -> np.ndarray:
        """
        Create sampling probability map for adaptive patch extraction
        
        Args:
            vessel_probability: Vessel probability map
            
        Returns:
            Sampling probability map
        """
        # Combine vessel probability with uniform sampling
        uniform_prob = np.ones_like(vessel_probability) * 0.1
        vessel_prob = vessel_probability * 0.9
        
        sampling_prob = uniform_prob + vessel_prob
        
        # Apply Gaussian smoothing for spatial coherence
        sampling_prob = cv2.GaussianBlur(sampling_prob, (5, 5), 1.0)
        
        return sampling_prob
    
    def _get_valid_coordinates(self, height: int, width: int) -> np.ndarray:
        """
        Get valid patch center coordinates
        
        Args:
            height, width: Image dimensions
            
        Returns:
            Array of valid linear indices
        """
        half_patch = self.config.patch_size // 2
        
        valid_y = np.arange(half_patch, height - half_patch)
        valid_x = np.arange(half_patch, width - half_patch)
        
        yy, xx = np.meshgrid(valid_y, valid_x, indexing='ij')
        valid_coords = np.ravel_multi_index((yy.flatten(), xx.flatten()), (height, width))
        
        return valid_coords


def visualize_patches(image: np.ndarray, 
                     coordinates: np.ndarray, 
                     patches: np.ndarray,
                     max_patches: int = 20) -> None:
    """
    Visualize extracted patches on the original image
    
    Args:
        image: Original retinal image
        coordinates: Patch center coordinates
        patches: Extracted patches
        max_patches: Maximum number of patches to show
    """
    plt.figure(figsize=(15, 10))
    
    # Show original image with patch locations
    plt.subplot(2, 3, 1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    
    # Plot patch centers
    coords_to_show = coordinates[:max_patches]
    plt.scatter(coords_to_show[:, 1], coords_to_show[:, 0], 
               c='red', s=20, alpha=0.7)
    plt.title('Original Image with Patch Locations')
    plt.axis('off')
    
    # Show individual patches
    n_show = min(max_patches, len(patches))
    for i in range(min(n_show, 15)):  # Show up to 15 patches
        plt.subplot(2, 8, i + 9)
        if len(patches[i].shape) == 3:
            if patches[i].dtype == np.float32:
                # Denormalize for visualization
                patch_vis = (patches[i] - patches[i].min()) / (patches[i].max() - patches[i].min())
                plt.imshow(patch_vis)
            else:
                plt.imshow(cv2.cvtColor(patches[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(patches[i], cmap='gray')
        plt.title(f'Patch {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Demonstration of patch extraction functionality
    """
    # Load example image (update path as needed)
    image_path = '08_test.tif'
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        print("Please update the image path in the script.")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        return
    
    print(f"Loaded image: {image.shape}")
    
    # Initialize patch extractor
    config = PatchConfig(
        patch_size=32,
        stride=16,
        normalize=True,
        remove_black_patches=True,
        min_variance=10.0
    )
    
    extractor = RetinalPatchExtractor(config)
    
    # Extract regular patches
    print("Extracting regular patches...")
    patches, coordinates = extractor.extract_patches(image)
    print(f"Extracted {len(patches)} patches")
    
    # Extract multi-scale patches
    print("Extracting multi-scale patches...")
    multiscale_results = extractor.extract_multiscale_patches(
        image, scales=[16, 32, 64]
    )
    
    for scale, (patches_scale, coords_scale) in multiscale_results.items():
        print(f"Scale {scale}: {len(patches_scale)} patches")
    
    # Visualize results
    print("Visualizing results...")
    visualize_patches(image, coordinates, patches, max_patches=20)
    
    # Save sample patches
    output_dir = 'extracted_patches'
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(min(10, len(patches))):
        patch_path = os.path.join(output_dir, f'patch_{i:03d}.png')
        if len(patches[i].shape) == 3:
            cv2.imwrite(patch_path, patches[i])
        else:
            cv2.imwrite(patch_path, (patches[i] * 255).astype(np.uint8))
    
    print(f"Saved sample patches to '{output_dir}' directory")


if __name__ == "__main__":
    main()
