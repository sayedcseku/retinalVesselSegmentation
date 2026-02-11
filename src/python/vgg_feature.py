#!/usr/bin/env python3
"""
VGG Feature Extraction for Retinal Vessel Segmentation

This module implements VGG-based deep feature extraction for retinal vessel
segmentation, providing pre-trained CNN features as an alternative to 
traditional hand-crafted features.

Author: Retinal Vessel Segmentation Research Team
Date: February 2026
Python Version: 3.7+

Dependencies:
    - tensorflow or pytorch (depending on implementation)
    - numpy
    - opencv-python (cv2)
    - scikit-learn (for dimensionality reduction)
    - matplotlib (for visualization)

Example Usage:
    from vgg_feature import VGGFeatureExtractor
    
    # Initialize feature extractor
    extractor = VGGFeatureExtractor(model_type='vgg16', layer='block4_conv3')
    
    # Extract features from retinal image
    features = extractor.extract_features(image_path)
    
    # Use for vessel segmentation
    segmentation = extractor.segment_vessels(image_path)
"""

import numpy as np
import cv2
import os
from typing import Tuple, List, Optional, Union, Dict
import warnings
from abc import ABC, abstractmethod

# Try to import deep learning frameworks
try:
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16, VGG19
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import UpSampling2D, Conv2D
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. Deep learning features disabled.")

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some features disabled.")


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    @abstractmethod
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from image"""
        pass
    
    @abstractmethod
    def segment_vessels(self, image: np.ndarray) -> np.ndarray:
        """Segment vessels using extracted features"""
        pass


class VGGFeatureExtractor(BaseFeatureExtractor):
    """
    VGG-based feature extraction for retinal vessel segmentation
    
    This class uses pre-trained VGG networks to extract deep features
    from retinal images, which can be used for vessel segmentation
    either directly or as input to other classifiers.
    """
    
    def __init__(self, 
                 model_type: str = 'vgg16',
                 layer: str = 'block4_conv3',
                 input_size: Tuple[int, int] = (224, 224),
                 use_pca: bool = True,
                 pca_components: int = 50):
        """
        Initialize VGG feature extractor
        
        Args:
            model_type: 'vgg16' or 'vgg19'
            layer: Layer name for feature extraction
            input_size: Input image size for VGG
            use_pca: Whether to apply PCA for dimensionality reduction
            pca_components: Number of PCA components
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for VGG feature extraction")
        
        self.model_type = model_type
        self.layer = layer
        self.input_size = input_size
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        # Initialize models
        self._load_model()
        self.pca_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
    def _load_model(self):
        """Load pre-trained VGG model"""
        if self.model_type == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False,
                              input_shape=(*self.input_size, 3))
        elif self.model_type == 'vgg19':
            base_model = VGG19(weights='imagenet', include_top=False,
                              input_shape=(*self.input_size, 3))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create feature extraction model
        try:
            feature_layer = base_model.get_layer(self.layer)
            self.feature_model = Model(inputs=base_model.input,
                                     outputs=feature_layer.output)
        except ValueError:
            print(f"Warning: Layer '{self.layer}' not found. Using last conv layer.")
            # Find last convolutional layer
            conv_layers = [layer for layer in base_model.layers 
                          if 'conv' in layer.name.lower()]
            self.feature_model = Model(inputs=base_model.input,
                                     outputs=conv_layers[-1].output)
        
        # Create segmentation model (simple upsampling approach)
        self._create_segmentation_model()
        
    def _create_segmentation_model(self):
        """Create a simple segmentation model using VGG features"""
        # Get feature model output shape
        feature_shape = self.feature_model.output_shape[1:]
        
        # Create segmentation head
        inputs = tf.keras.Input(shape=feature_shape)
        x = inputs
        
        # Reduce channels
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        
        # Upsample to original resolution
        target_h, target_w = self.input_size
        current_h, current_w = feature_shape[:2]
        
        scale_h = target_h // current_h
        scale_w = target_w // current_w
        
        if scale_h > 1 or scale_w > 1:
            x = UpSampling2D(size=(scale_h, scale_w))(x)
        
        # Final segmentation layer
        x = Conv2D(1, 1, activation='sigmoid', padding='same')(x)
        
        self.segmentation_model = Model(inputs=inputs, outputs=x)
        
        # Compile model (for potential training)
        self.segmentation_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract VGG features from retinal image
        
        Args:
            image: Input retinal image (BGR or RGB)
            
        Returns:
            Extracted features array
        """
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Extract features
        features = self.feature_model.predict(processed_image, verbose=0)
        
        # Reshape features for pixel-wise analysis
        batch_size, height, width, channels = features.shape
        features_reshaped = features.reshape(-1, channels)
        
        # Apply PCA if requested
        if self.use_pca and SKLEARN_AVAILABLE:
            features_reshaped = self._apply_pca(features_reshaped)
        
        return features_reshaped.reshape(height, width, -1)
    
    def extract_patch_features(self, 
                             image: np.ndarray, 
                             patch_size: int = 64,
                             stride: int = 32) -> Dict[str, np.ndarray]:
        """
        Extract features from image patches
        
        Args:
            image: Input retinal image
            patch_size: Size of patches to extract
            stride: Stride for patch extraction
            
        Returns:
            Dictionary containing features and coordinates
        """
        height, width = image.shape[:2]
        
        patches = []
        coordinates = []
        
        # Extract patches
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                
                if patch.shape[:2] == (patch_size, patch_size):
                    patches.append(patch)
                    coordinates.append([y + patch_size//2, x + patch_size//2])
        
        # Convert to batch
        patch_batch = np.array(patches)
        
        # Resize patches to VGG input size if needed
        if patch_size != self.input_size[0]:
            resized_patches = []
            for patch in patch_batch:
                resized = cv2.resize(patch, self.input_size)
                resized_patches.append(resized)
            patch_batch = np.array(resized_patches)
        
        # Preprocess for VGG
        if len(patch_batch.shape) == 3:
            patch_batch = np.expand_dims(patch_batch, -1)
            patch_batch = np.repeat(patch_batch, 3, axis=-1)
        elif patch_batch.shape[-1] == 1:
            patch_batch = np.repeat(patch_batch, 3, axis=-1)
        
        patch_batch = preprocess_input(patch_batch.astype(np.float32))
        
        # Extract features
        patch_features = self.feature_model.predict(patch_batch, verbose=0)
        
        return {
            'features': patch_features,
            'coordinates': np.array(coordinates),
            'patches': patches
        }
    
    def segment_vessels(self, image: np.ndarray) -> np.ndarray:
        """
        Perform vessel segmentation using VGG features
        
        Args:
            image: Input retinal image
            
        Returns:
            Binary vessel segmentation mask
        """
        # Method 1: Use simple segmentation model
        processed_image = self._preprocess_image(image)
        features = self.feature_model.predict(processed_image, verbose=0)
        segmentation = self.segmentation_model.predict(features, verbose=0)
        
        # Resize to original image size
        original_size = image.shape[:2]
        segmentation = cv2.resize(segmentation[0, :, :, 0], 
                                (original_size[1], original_size[0]))
        
        # Apply threshold
        binary_seg = (segmentation > 0.5).astype(np.uint8)
        
        return binary_seg
    
    def segment_vessels_clustering(self, image: np.ndarray) -> np.ndarray:
        """
        Perform vessel segmentation using clustering on VGG features
        
        Args:
            image: Input retinal image
            
        Returns:
            Binary vessel segmentation mask
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for clustering-based segmentation")
        
        # Extract features
        features = self.extract_features(image)
        height, width, n_features = features.shape
        
        # Reshape for clustering
        features_flat = features.reshape(-1, n_features)
        
        # Remove invalid features
        valid_mask = np.all(np.isfinite(features_flat), axis=1)
        valid_features = features_flat[valid_mask]
        
        if len(valid_features) == 0:
            return np.zeros((height, width), dtype=np.uint8)
        
        # Apply clustering (2 clusters: vessel vs background)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(valid_features)
        
        # Create full label array
        full_labels = np.zeros(len(features_flat))
        full_labels[valid_mask] = labels
        
        # Reshape back to image
        label_image = full_labels.reshape(height, width)
        
        # Determine which cluster represents vessels
        # Assume vessels have lower intensity on average
        cluster_means = []
        for cluster_id in range(2):
            cluster_mask = label_image == cluster_id
            if np.any(cluster_mask):
                if len(image.shape) == 3:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = image
                cluster_intensity = np.mean(gray_image[cluster_mask])
                cluster_means.append(cluster_intensity)
            else:
                cluster_means.append(255)
        
        vessel_cluster = np.argmin(cluster_means)
        segmentation = (label_image == vessel_cluster).astype(np.uint8)
        
        return segmentation
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for VGG input
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image batch
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR input from OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            # Grayscale to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        # Resize to VGG input size
        resized = cv2.resize(image_rgb, self.input_size)
        
        # Add batch dimension
        batch = np.expand_dims(resized, axis=0)
        
        # Apply VGG preprocessing
        preprocessed = preprocess_input(batch.astype(np.float32))
        
        return preprocessed
    
    def _apply_pca(self, features: np.ndarray) -> np.ndarray:
        """
        Apply PCA to reduce feature dimensionality
        
        Args:
            features: Input features
            
        Returns:
            PCA-transformed features
        """
        if self.pca_model is None:
            # Fit PCA model
            self.pca_model = PCA(n_components=self.pca_components)
            
            # Scale features first
            features_scaled = self.scaler.fit_transform(features)
            features_pca = self.pca_model.fit_transform(features_scaled)
        else:
            # Transform using existing model
            features_scaled = self.scaler.transform(features)
            features_pca = self.pca_model.transform(features_scaled)
        
        return features_pca
    
    def visualize_features(self, 
                          image: np.ndarray, 
                          save_path: Optional[str] = None) -> None:
        """
        Visualize extracted VGG features
        
        Args:
            image: Input retinal image
            save_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for visualization")
            return
        
        # Extract features
        features = self.extract_features(image)
        
        # Visualize first few feature channels
        n_channels = min(9, features.shape[2])
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f'VGG {self.model_type.upper()} Features - Layer: {self.layer}')
        
        for i in range(n_channels):
            row = i // 3
            col = i % 3
            
            ax = axes[row, col]
            feature_map = features[:, :, i]
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Channel {i+1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for i in range(n_channels, 9):
            row = i // 3
            col = i % 3
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature visualization saved to: {save_path}")
        
        plt.show()


def main():
    """
    Demonstration of VGG feature extraction
    """
    # Check availability
    if not TF_AVAILABLE:
        print("TensorFlow not available. Cannot run VGG feature extraction demo.")
        return
    
    # Load example image
    image_path = '08_test.tif'
    
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found.")
        print("Please update the image path or place a test image in the current directory.")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image from '{image_path}'")
        return
    
    print(f"Loaded image: {image.shape}")
    
    # Initialize VGG feature extractor
    print("Initializing VGG feature extractor...")
    extractor = VGGFeatureExtractor(
        model_type='vgg16',
        layer='block4_conv3',
        use_pca=True,
        pca_components=50
    )
    
    # Extract features
    print("Extracting VGG features...")
    features = extractor.extract_features(image)
    print(f"Extracted features shape: {features.shape}")
    
    # Perform vessel segmentation
    print("Performing vessel segmentation...")
    
    # Method 1: Simple segmentation model
    segmentation1 = extractor.segment_vessels(image)
    print(f"Segmentation 1 shape: {segmentation1.shape}")
    
    # Method 2: Clustering-based segmentation
    if SKLEARN_AVAILABLE:
        segmentation2 = extractor.segment_vessels_clustering(image)
        print(f"Segmentation 2 shape: {segmentation2.shape}")
    
    # Extract patch features
    print("Extracting patch features...")
    patch_results = extractor.extract_patch_features(image, patch_size=64, stride=32)
    print(f"Extracted {len(patch_results['patches'])} patches")
    print(f"Patch features shape: {patch_results['features'].shape}")
    
    # Visualize features
    print("Visualizing features...")
    extractor.visualize_features(image, save_path='vgg_features_visualization.png')
    
    # Save segmentation results
    cv2.imwrite('vgg_segmentation_simple.png', segmentation1 * 255)
    if SKLEARN_AVAILABLE:
        cv2.imwrite('vgg_segmentation_clustering.png', segmentation2 * 255)
    
    print("VGG feature extraction demo completed!")


if __name__ == "__main__":
    main()
