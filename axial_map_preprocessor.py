"""
Main pipeline for processing Pentacam/Oculyzer images
Extracts and cleans axial curvature maps
"""

import cv2 as cv
import numpy as np
import os
from typing import Tuple, Optional, Dict
from pathlib import Path
from constants import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Import from the modules above
# If you saved them as separate files, use:
from constants import *
from ocr_utils import extract_header_text, match_header
from image_cleaning import clean_axial_map, standardize_image


class AxialMapProcessor:
    """Processor for extracting and cleaning axial curvature maps."""
    
    def __init__(self, ocr_method: str = 'auto'):
        self.ocr_method = ocr_method
    
    def get_image_dimensions(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """Get image dimensions and check if supported."""
        height, width = image.shape[:2]
        dim_str = f"{height}_{width}"
        
        if dim_str in SUPPORTED_DIMENSIONS:
            return (height, width)
        return None
    
    def extract_and_match_header(self, image: np.ndarray, 
                                 dimensions: Tuple[int, int]) -> Optional[str]:
        """Extract header text and match against known headers."""
        dim_str = f"{dimensions[0]}_{dimensions[1]}"
        
        if dim_str not in IMAGE_HEADER_COORDS:
            return None
        
        header_coords = IMAGE_HEADER_COORDS[dim_str]
        header_text = extract_header_text(image, header_coords)
        
        matched_header = match_header(header_text, MODEL_HEADERS)
        return matched_header
    
    def get_axial_map_coordinates(self, dimensions: Tuple[int, int], 
                                  header: str) -> Optional[Dict]:
        """Get coordinates for axial map extraction."""
        is_refractive = 'Refractive' in header
        
        coords_dict = REFRACTIVE_MAP_COORDS if is_refractive else SELECTABLE_MAP_COORDS
        
        if dimensions not in coords_dict:
            return None
        
        return coords_dict[dimensions]
    
    def extract_axial_map(self, image: np.ndarray, map_coords: Dict) -> np.ndarray:
        """Extract axial map region from full image."""
        map1 = map_coords['map1']
        axial_map = image[map1['row1']:map1['row2'], 
                         map1['col1']:map1['col2']]
        return axial_map
    
    def process_image(self, image_path: str, output_dir: Optional[str] = None,
                     save_full: bool = True, save_standardized: bool = True) -> Dict:
        """Process a single image: extract and clean axial map."""
        result = {
            'success': False,
            'message': '',
            'dimensions': None,
            'header': None,
            'device': None,
            'axial_map_full': None,
            'axial_map_224': None,
            'output_paths': {}
        }
        
        # Read image
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        if image is None:
            result['message'] = f"Failed to read image: {image_path}"
            return result
        
        # Check dimensions
        dimensions = self.get_image_dimensions(image)
        if dimensions is None:
            result['message'] = f"Unsupported image dimensions: {image.shape[:2]}"
            return result
        
        result['dimensions'] = dimensions
        
        # Extract and match header
        header = self.extract_and_match_header(image, dimensions)
        if header is None:
            result['message'] = "Could not identify image header/type"
            return result
        
        result['header'] = header
        result['device'] = 'pentacam' if header in PENTACAM_HEADERS else 'oculyzer'
        
        # Get axial map coordinates
        map_coords = self.get_axial_map_coordinates(dimensions, header)
        if map_coords is None:
            result['message'] = f"No coordinates defined for dimensions: {dimensions}"
            return result
        
        # Extract axial map
        axial_map_raw = self.extract_axial_map(image, map_coords)
        
        # Clean axial map
        circle_params = map_coords['circle_loc']
        axial_map_full, axial_map_224 = clean_axial_map(axial_map_raw, circle_params)
        
        result['axial_map_full'] = axial_map_full
        result['axial_map_224'] = axial_map_224
        
        # Save outputs if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = Path(image_path).stem
            
            if save_full:
                full_path = os.path.join(output_dir, f"{base_name}_axial_full.png")
                cv.imwrite(full_path, axial_map_full)
                result['output_paths']['full'] = full_path
            
            if save_standardized:
                std_path = os.path.join(output_dir, f"{base_name}_axial_224.png")
                cv.imwrite(std_path, axial_map_224)
                result['output_paths']['standardized'] = std_path
        
        result['success'] = True
        result['message'] = "Successfully processed image"
        
        return result
    
    def process_batch(self, image_paths: list, output_dir: str,
                     save_full: bool = True, save_standardized: bool = True) -> list:
        """Process multiple images."""
        results = []
        
        for img_path in image_paths:
            print(f"Processing: {img_path}")
            result = self.process_image(img_path, output_dir, 
                                       save_full, save_standardized)
            results.append(result)
            
            if result['success']:
                print(f"  ✓ Success: {result['device']} - {result['dimensions']}")
            else:
                print(f"  ✗ Failed: {result['message']}")
        
        return results


def process_single_image(image_path: str, output_dir: str = None) -> Dict:
    """Convenience function to process a single image."""
    processor = AxialMapProcessor()
    return processor.process_image(image_path, output_dir)


def process_directory(input_dir: str, output_dir: str, 
                      pattern: str = "*.png") -> list:
    """Process all images in a directory."""
    from glob import glob
    
    image_paths = glob(os.path.join(input_dir, pattern))
    
    if not image_paths:
        print(f"No images found in {input_dir} matching pattern {pattern}")
        return []
    
    processor = AxialMapProcessor()
    return processor.process_batch(image_paths, output_dir)


# Example usage
if __name__ == "__main__":
    # Process single image
    result = process_single_image(
        image_path="path/to/your/pentacam_image.png",
        output_dir="output"
    )
    
    if result['success']:
        print(f"Processed successfully!")
        print(f"Device: {result['device']}")
        print(f"Dimensions: {result['dimensions']}")
        print(f"Saved to: {result['output_paths']}")
    else:
        print(f"Processing failed: {result['message']}")