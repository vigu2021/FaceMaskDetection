import os
import shutil
import random
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#Source files
source_images_dir = 'data/images'
source_annotations_dir = 'data/labels'

# Root directory
output_base = 'data_yolo'
os.makedirs(output_base,exist_ok=True)

#Subdirectory splitting images and labels 
output_images = os.path.join(output_base,'images')
os.makedirs(output_images,exist_ok=True)

output_labels = os.path.join(output_base,'labels')
os.makedirs(output_labels,exist_ok=True)

# Create output directories with different methods and dataset splits
methods = ['resize','resize_pad','pad_resize']
for method in methods:
    output_images_method = os.path.join(output_images,method)
    output_labels_method = os.path.join(output_labels,method)
    # Create method directories
    os.makedirs(output_images_method,exist_ok=True)
    os.makedirs(output_labels_method,exist_ok=True)

    # Create train, val, test subdirectories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_images_method, split),exist_ok=True)
        os.makedirs(os.path.join(output_labels_method, split),exist_ok=True)

logging.info("✅ Directories created successfully!")

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

def validate_directory_and_return_files(directory_path, allowed_extensions=None):
    """
    Validates a directory and returns a list of valid files.

    Args:
        directory_path (str): The path to the directory.
        allowed_extensions (tuple, optional): File extensions to filter. Defaults to None.

    Returns:
        list: List of valid filenames if successful, empty list otherwise.
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        logger.error(f"❌ Directory not found: {directory_path}")
        return []

    # List all files in the directory
    all_files = os.listdir(directory_path)

    # Check if the directory is empty
    if len(all_files) == 0:
        logger.warning(f"⚠️ Directory is empty: {directory_path}")
        return []

    # If no filtering is needed, return all files
    if allowed_extensions is None:
        logger.info(f"✅ Found {len(all_files)} files in {directory_path}")
        return all_files

    # Filter files based on extensions
    valid_files = [f for f in all_files if f.lower().endswith(allowed_extensions)]

    # Handle no valid files found
    if len(valid_files) == 0:
        logger.error(f"❌ No valid files found in {directory_path}. Expected extensions: {allowed_extensions}")
        return []

    # Warn if some files were skipped due to unsupported formats
    if len(valid_files) < len(all_files):
        logger.warning(f"⚠️ Out of {len(all_files)} files, only {len(valid_files)} were valid in {directory_path}")

    logger.info(f"✅ Found {len(valid_files)} valid files in {directory_path}")
    return valid_files


def split_data_set():
    logger.info("Starting data set split")
    pass

    for method in methods:
        method_image_dir = os.path.join(source_images_dir,method)
        method_label_dir = os.path.join(source_annotations_dir,method)

        image_names = validate_directory_and_return_files(method_image_dir,allowed_extensions=('.txt',))
        label_names = validate_directory_and_return_files(method_label_dir,allowed_extensions = ('.jpg','.png'))




if __name__ == "__main__":
    print(len(validate_directory_and_return_files(os.path.join(source_images_dir,'resize'))))
    

        



    
