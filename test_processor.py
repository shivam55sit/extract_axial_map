import os
from axial_map_preprocessor import process_single_image
from constants import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Check what files are available
print("Current directory:", os.getcwd())
print("\nAvailable image files:")
for file in os.listdir('.'):
    if file.endswith(('.png', '.jpg', '.jpeg')):
        print(f"  - {file}")

# Update this path to your actual image
IMAGE_PATH = r"C:\Users\shivam.prajapati\Documents\lvp-projects\suture_radilaity\pentacam_OD.jpg"

# Check if file exists
if not os.path.exists(IMAGE_PATH):
    print(f"\n❌ Error: File not found at: {IMAGE_PATH}")
    print("\nPlease update IMAGE_PATH variable with correct path")
else:
    print(f"\n✓ Found image: {IMAGE_PATH}")
    
    # Process the image
    print("\nProcessing...")
    result = process_single_image(IMAGE_PATH, "output")
    
    if result['success']:
        print("\n✓ SUCCESS!")
        print(f"  Device: {result['device']}")
        print(f"  Dimensions: {result['dimensions']}")
        print(f"  Header: {result['header']}")
        print(f"\nSaved files:")
        for key, path in result['output_paths'].items():
            print(f"  {key}: {path}")
    else:
        print(f"\n❌ FAILED: {result['message']}")