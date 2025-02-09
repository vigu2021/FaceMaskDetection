import os
import cv2
import numpy as np
import torch
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resize_images(image_folder='data/images', target_size=(224, 224), method='resize'):
    """
    Preprocesses images using one of the following methods:
    - 'resize': Directly resizes the image to the target size (default).
    - 'pad_resize': Pads the image to a square shape first, then resizes to the target size.
    - 'resize_pad': Resizes the image while maintaining the aspect ratio, then pads to reach the target size.

    Parameters:
    ----------
    image_folder : str, optional
        Path to the folder containing images (default is 'data/images').
    target_size : tuple, optional
        Desired size for resizing/padding (default is (224, 224)).
    method : str, optional
        Preprocessing method: 'resize', 'pad_resize', or 'resize_pad' (default is 'resize').

    Returns:
    -------
    list or torch.Tensor or None
        - A list of PyTorch tensors (for 'same' method).
        - A 4D PyTorch tensor (for 'resize', 'pad_resize', or 'resize_pad' methods).
        - None if no images are processed.
    """

    if method not in ['resize', 'pad_resize', 'resize_pad']:
        logger.error(f"Invalid method!: {method}. Please use 'resize', 'pad_resize', or 'resize_pad'.")
        return None
    images_list = []

    # Check if the folder exists
    if not os.path.exists(image_folder):
        logger.error(f"The folder does not exist at: {os.path.abspath(image_folder)}")
        logger.error("Operation stopped! File not found.")
        return None

    image_files = os.listdir(image_folder)

    for image_name in image_files:
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_image_path = os.path.join(image_folder, image_name)

            try:
                # Read the image using OpenCV
                image_decoded = cv2.imread(full_image_path)
                if image_decoded is None:
                    raise ValueError("Image could not be decoded. Possibly corrupted.")

                # Convert from BGR to RGB
                image_rgb = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)

                
                #Resize method
                if method == 'resize':
                    image_resized = cv2.resize(image_rgb, target_size)
                    images_list.append(image_resized)
                    logger.info(f"Processed: {image_name}, Resized to: {image_resized.shape[:2]}")
                # Resize and pad
                elif method == 'resize_pad':
                    image_resized_padded = resize_then_pad(image_rgb,target_size)
                    images_list.append(image_resized_padded)
                    logger.info(f"Processed: {image_name}, Resized to: {image_resized_padded.shape[:2]}")
                    if image_resized_padded.shape[:2] != target_size[:2]:
                        logger.error(f"Dimension don't match: {image_name}, Resized to: {image_resized_padded.shape[:2]}")



                #Resize then pad method

            except Exception as e:
                logger.warning(f"Failed to process {image_name}: {e}")
        else:
            logger.warning(f"Skipping unsupported file format: {image_name}")

    if not images_list:
        logger.error("No images were processed successfully.")
        return None

    # Convert to tensor for 'resize', keep as a list for 'same'
    images_numpy = np.array(images_list)
    images_tensor = torch.from_numpy(images_numpy).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    logger.info(f"🎯 Final Tensor Shape: {images_tensor.shape}, Dtype: {images_tensor.dtype}")
    return images_tensor





def resize_then_pad(image,target_size=(224, 224)):
    """
    Resizes the image while maintaining the aspect ratio, 
    then pads it to reach the target size.

    Parameters:
    ----------
    image : np.ndarray
        The input image (in RGB format),
    target_size : tuple, optional
        The desired output size (width, height), default is (224, 224). Ensure width = height 
    Returns:
    -------
    np.ndarray
        The resized and padded image.
    """ 
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Step 1: Resize while maintaining aspect ratio
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image,(new_width,new_height))

    #Step 2: Calculate padding and pad image
    delta_w = target_width - new_width
    delta_h = target_height - new_height

    top = delta_h // 2 
    bottom = target_height - (top + new_height)
    left = delta_w//2
    right = target_width - (left + new_width)

    final_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, borderType = cv2.BORDER_CONSTANT,value=(0,0,0))
    return final_image




resize_images(method = 'resize_pad')

    




