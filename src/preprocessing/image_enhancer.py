"""
Advanced image preprocessing for OCR enhancement.
Implements SOTA preprocessing techniques for meter reading images.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image


class ImageEnhancer:
    """
    Advanced image enhancement for OCR preprocessing.
    
    Implements multiple enhancement techniques:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Denoising (bilateral filtering, non-local means)
    - Deskewing and perspective correction
    - Binarization with adaptive thresholding
    - Shadow removal
    """
    
    def __init__(self, 
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_size: Tuple[int, int] = (8, 8),
                 denoise_strength: int = 10,
                 bilateral_d: int = 9):
        """
        Initialize ImageEnhancer with configurable parameters.
        
        Args:
            clahe_clip_limit: Threshold for contrast limiting
            clahe_tile_size: Size of grid for histogram equalization
            denoise_strength: Strength of denoising filter
            bilateral_d: Diameter of bilateral filter
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.denoise_strength = denoise_strength
        self.bilateral_d = bilateral_d
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size
        )
    
    def enhance(self, image: np.ndarray, 
                apply_clahe: bool = True,
                apply_denoise: bool = True,
                apply_deskew: bool = True,
                apply_binarize: bool = False) -> np.ndarray:
        """
        Apply full enhancement pipeline to image.
        
        Args:
            image: Input image (BGR or grayscale)
            apply_clahe: Whether to apply CLAHE
            apply_denoise: Whether to apply denoising
            apply_deskew: Whether to apply deskewing
            apply_binarize: Whether to apply binarization
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement
        if apply_clahe:
            gray = self.apply_clahe(gray)
        
        # Apply denoising
        if apply_denoise:
            gray = self.apply_denoise(gray)
        
        # Apply deskewing
        if apply_deskew:
            gray = self.deskew(gray)
        
        # Apply binarization
        if apply_binarize:
            gray = self.binarize(gray)
        
        return gray
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for adaptive contrast enhancement."""
        return self.clahe.apply(image)
    
    def apply_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering for edge-preserving denoising.
        
        Bilateral filter reduces noise while preserving edges,
        which is crucial for OCR accuracy.
        """
        return cv2.bilateralFilter(
            image, 
            self.bilateral_d, 
            self.denoise_strength * 2, 
            self.denoise_strength * 2
        )
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image skew using Hough transform.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Deskewed image
        """
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        if lines is None:
            return image
        
        # Calculate average angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
        
        median_angle = np.median(angles)
        
        # Only correct if skew is significant (> 0.5 degrees)
        if abs(median_angle) < 0.5:
            return image
        
        # Rotate image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def binarize(self, image: np.ndarray, 
                 method: str = 'adaptive') -> np.ndarray:
        """
        Apply binarization to image.
        
        Args:
            image: Input grayscale image
            method: 'otsu', 'adaptive', or 'combined'
            
        Returns:
            Binarized image
        """
        if method == 'otsu':
            _, binary = cv2.threshold(
                image, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary
        
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            return binary
        
        elif method == 'combined':
            # Combine Otsu and adaptive thresholding
            _, otsu = cv2.threshold(
                image, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            adaptive = cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            # Take weighted average
            combined = cv2.addWeighted(otsu, 0.5, adaptive, 0.5, 0)
            return combined
        
        else:
            raise ValueError(f"Unknown binarization method: {method}")
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Remove shadows from image using morphological operations.
        
        Useful for meter images taken in varying lighting conditions.
        """
        # Dilate image to remove text
        dilated = cv2.dilate(
            image,
            np.ones((7, 7), np.uint8),
            iterations=1
        )
        
        # Apply median blur to get background
        bg = cv2.medianBlur(dilated, 21)
        
        # Subtract background from original
        diff = 255 - cv2.absdiff(image, bg)
        
        # Normalize
        norm = cv2.normalize(
            diff, None, 
            alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX
        )
        
        return norm
    
    @staticmethod
    def resize_for_ocr(image: np.ndarray, 
                       target_height: int = 64,
                       maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize image to optimal size for OCR.
        
        Args:
            image: Input image
            target_height: Target height in pixels
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        if maintain_aspect:
            aspect_ratio = w / h
            target_width = int(target_height * aspect_ratio)
        else:
            target_width = target_height
        
        resized = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC
        )
        
        return resized
