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

                # Padding then resize method can be implemented here if needed


                #Resize then pad method

            except Exception as e:
                logger.warning(f"Failed to process {image_name}: {e}")
        else:
            logger.warning(f"Skipping unsupported file format: {image_name}")

    if not images_list:
        logger.error("No images were processed successfully.")
        return None

    # Convert to tensor for 'resize', keep as a list for 'same'
    if method == 'resize':
        images_numpy = np.array(images_list)
        images_tensor = torch.from_numpy(images_numpy).permute(0, 3, 1, 2).to(torch.float32) / 255.0
        logger.info(f"🎯 Final Tensor Shape: {images_tensor.shape}, Dtype: {images_tensor.dtype}")
        return images_tensor

    return images_list  # Return list of tensors for 'same' method






