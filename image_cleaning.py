"""
Image cleaning functions for axial curvature maps
Removes artifacts and standardizes the output
"""

import cv2 as cv
import numpy as np
from typing import Tuple, Dict


def standardize_image(img: np.ndarray, output_size: int = 224) -> np.ndarray:
    """Standardize image to fixed size with circle centered."""
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Threshold to get the circle
    _, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cv.resize(img, (output_size, output_size), interpolation=cv.INTER_AREA)
    
    # Find the largest contour
    largest_contour = max(contours, key=cv.contourArea)
    
    # Fit a minimum enclosing circle
    (x, y), radius = cv.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Create a square crop around the circle
    x1 = max(center[0] - radius, 0)
    y1 = max(center[1] - radius, 0)
    x2 = min(center[0] + radius, img.shape[1])
    y2 = min(center[1] + radius, img.shape[0])
    
    cropped = img[y1:y2, x1:x2]
    
    # Resize to output_size x output_size
    resized = cv.resize(cropped, (output_size, output_size), interpolation=cv.INTER_AREA)
    
    # Create a mask for the circle
    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    cv.circle(mask, (output_size//2, output_size//2), output_size//2, 255, -1)
    
    # Apply mask
    result = cv.bitwise_and(resized, resized, mask=mask)
    
    return result


def clean_axial_map(img: np.ndarray, circle_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Clean axial curvature map by removing artifacts."""
    cir_x = circle_params['cir_x']
    cir_y = circle_params['cir_y']
    cir_radius = circle_params['cir_radius']
    
    # Keep a median filtered version
    median_filtered = cv.medianBlur(img, 23)
    
    # Convert BGR to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Threshold to get black colors
    mask = cv.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 200]))
    
    perc_mask = round(np.mean(mask == 255) * 100)
    
    if perc_mask > 20:
        mask = cv.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 160]))
    
    # Inpaint artifacts
    result = cv.inpaint(img.copy(), mask, 3, cv.INPAINT_NS)
    
    # Create mask for extreme values
    mask1 = np.zeros((result.shape[0], result.shape[1]), dtype=np.uint8)
    
    dark_mask = np.all(result <= 75, axis=2)
    medium_mask = np.logical_and(
        np.all(result[:, :, :2] <= 100, axis=2),
        result[:, :, 2] <= 10
    )
    bright_mask = np.all(result >= 180, axis=2)
    
    mask1[dark_mask | medium_mask | bright_mask] = 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        mask1, connectivity=4
    )
    
    # Remove small artifacts
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if 1 < area < 600:
            x = max(0, stats[i, cv.CC_STAT_LEFT] - 5)
            y = max(0, stats[i, cv.CC_STAT_TOP] - 5)
            w = min(result.shape[1] - x, stats[i, cv.CC_STAT_WIDTH] + 5)
            h = min(result.shape[0] - y, stats[i, cv.CC_STAT_HEIGHT] + 5)
            
            y_coords, x_coords = np.ogrid[y:y+h+2, x:x+w+2]
            dist_squared = (x_coords - cir_x)**2 + (y_coords - cir_y)**2
            mask_circle = dist_squared <= cir_radius**2
            
            temp_mask = np.zeros((result.shape[0], result.shape[1]), dtype=np.uint8)
            
            if mask_circle.all():
                temp_mask[y:y+h+2, x:x+w+2] = 255
            else:
                temp_mask[y:y+h+2, x:x+w+2][mask_circle] = 255
            
            result = cv.inpaint(result, temp_mask, inpaintRadius=3, flags=cv.INPAINT_NS)
    
    # Handle bright spots
    circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv.circle(circle_mask, (cir_x, cir_y), cir_radius, 255, -1)
    mask2 = np.all(result >= 150, axis=2).astype(np.uint8) * 255
    mask2 = cv.bitwise_and(mask2, circle_mask)
    mask2_bool = mask2.astype(bool)
    result[mask2_bool] = median_filtered[mask2_bool]
    
    # Create final circular mask
    circle_mask_3ch = np.zeros_like(img)
    cv.circle(circle_mask_3ch, (cir_x, cir_y), cir_radius, (255, 255, 255), -1)
    
    # Apply circular mask
    result_full = cv.bitwise_and(result, circle_mask_3ch)
    
    # Standardize to 224x224
    result_standardized = standardize_image(result_full)
    
    return result_full, result_standardized