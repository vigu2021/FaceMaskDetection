import os
import random
import logging
import shutil



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#Source files
source_images_dir = 'data/images'
source_annotations_dir = 'data/annotations'

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
    

    for method in methods:
        method_image_dir = os.path.join(source_images_dir,method)
        method_label_dir = os.path.join(source_annotations_dir,method)

        # Validate and return all files found for images and labels 
        image_names = validate_directory_and_return_files(method_image_dir,allowed_extensions=('.jpg','.png'))
        label_names = validate_directory_and_return_files(method_label_dir,allowed_extensions=('.txt'))

        no_images = len(image_names)
        no_labels = len(label_names)

        if no_images != no_labels:
            logger.warning(f"⚠️ No. of images and labels aren't equal!: They are {no_images} images and {no_labels} labels")

        #Shuffle image names
        random.shuffle(image_names)
        
        train_count = int(train_ratio * no_images)
        val_count = int(val_ratio * no_images )
        test_count = no_images - train_count - val_count

        #Create training,validation and testing sets
        training_set = image_names[:train_count]
        validation_set = image_names[train_count:train_count+val_count]
        testing_set = image_names[train_count + val_count :]

        logger.info(f"Training set: {len(training_set)} files, Validation set: {len(validation_set)} files, Testing set: {len(testing_set)}")
        
        dataset = {
            'train': training_set,
            'val': validation_set,
            'test': testing_set
        }      
        target_image_dir = 'data_yolo/images'  
        target_label_dir = 'data_yolo/labels'
        for split_name,split_data in dataset.items():
            for image_name in split_data:
                image_file_name, extension = os.path.splitext(image_name)
                
                #Source image and label path
                src_image_path = os.path.join(method_image_dir, image_name)
                src_label_path = os.path.join(method_label_dir, image_file_name + '.txt')

                #Target image and label path
                target_image_path = os.path.join(target_image_dir,method,split_name,image_name)
                target_label_path = os.path.join(target_label_dir,method,split_name,image_file_name + '.txt')
                        
                #Copy image
                try:
                    shutil.copyfile(src_image_path,target_image_path)
                    logger.info(f"✅ Copied Image {src_image_path} → {target_image_path}")
                except FileNotFoundError:
                    logger.error(f"❌ Image not found: {src_image_path} → {target_image_path}")
                except Exception as e:
                    logger.error(f"❌ Error copying {src_image_path} due to {e}")
                
                #Copy label
                try:
                    shutil.copyfile(src_label_path,target_label_path)
                    logger.info(f"Copied Image {src_label_path} → {target_label_path}")
                except FileNotFoundError:
                    logger.error(f"❌ Image not found: {src_label_path}")
                
                except Exception as e:
                    logger.error(f"❌ Error copying {src_label_path} due to {e}")
                

        logger.info(f"✅ Dataset split completed!, find images at {target_image_dir} and labels at {target_label_dir}")

                
                
            
                



if __name__ == "__main__":
    split_data_set()
    

        



    
