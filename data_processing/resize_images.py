import os
import cv2
import numpy as np
import torch
import logging
from .extract_annotations import extract_annotations  # Import extract_annotations from another file

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def resize_image_with_annotations(image_path, xml_path, target_size=(224, 224), method="resize"):
    """
    Resizes both the image and bounding boxes extracted from the annotation file.

    Parameters:
    ----------
    image_path : str
        Path to the image file.
    xml_path : str
        Path to the XML annotation file.
    target_size : tuple, optional
        Desired size for resizing/padding (default is (224, 224)).
    method : str, optional
        Preprocessing method: 'resize', 'pad_resize', or 'resize_pad' (default is 'resize').

    Returns:
    -------
    tuple (torch.Tensor, dict)
        - A 3D PyTorch tensor (C, H, W) of the resized image.
        - A dictionary containing resized bounding boxes.
    """

    if method not in ["resize", "pad_resize", "resize_pad"]:
        logger.error(f"Invalid method: {method}. Please use 'resize', 'pad_resize', or 'resize_pad'.")
        return None, None

    if not os.path.exists(image_path) or not os.path.exists(xml_path):
        logger.error(f"Image or XML file not found: {image_path} | {xml_path}")
        return None, None

    # Extract bounding boxes and original image size from XML
    annotations = extract_annotations(xml_path)
    if not annotations:
        logger.error("Failed to extract annotations.")
        return None, None

    original_width = annotations["image_size"]["width"]
    original_height = annotations["image_size"]["height"]

    # Read and preprocess the image
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be decoded. Possibly corrupted.")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply chosen resizing method
        if method == "resize":
            scale_x = target_size[0] / original_width
            scale_y = target_size[1] / original_height
            image_resized = cv2.resize(image_rgb, target_size)
            shift_x = shift_y = 0  # No padding, so no shift

        elif method == "resize_pad":
            image_resized, scale_x, scale_y, shift_x, shift_y = resize_then_pad(image_rgb, target_size, original_width, original_height)

        elif method == "pad_resize":
            image_resized, scale_x, scale_y, shift_x, shift_y = pad_then_resize(image_rgb, target_size, original_width, original_height)

        else:
            logger.error(f"Invalid resizing method: {method}")
            return None, None

        logger.info(f"✅ Processed: {image_path}, New Size: {image_resized.shape[:2]}")

        # Scale and adjust bounding boxes
        resized_bboxes = []
        for annotation in annotations["annotations"]:
            label = annotation["label"]

            # Apply scaling first, then shifting
            x_min = int(annotation["coordinates"]["xmin"] * scale_x + shift_x)
            y_min = int(annotation["coordinates"]["ymin"] * scale_y + shift_y)
            x_max = int(annotation["coordinates"]["xmax"] * scale_x + shift_x)
            y_max = int(annotation["coordinates"]["ymax"] * scale_y + shift_y)

            logger.info(f"🔄 Adjusted Bounding Box for {label}: {x_min,y_min} to {x_max,y_max}")
            resized_bboxes.append({
                "label": label,
                "coordinates": {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max}
            })

        # Convert image to tensor
        image_numpy = np.array(image_resized, dtype=np.float32) / 255.0  # Normalize
        image_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1)  # Convert (H, W, C) → (C, H, W)

        return image_tensor, {"image_size": target_size, "annotations": resized_bboxes}

    except Exception as e:
        logger.warning(f"❌ Failed to process {image_path}: {e}")
        return None, None


def resize_then_pad(image, target_size, original_width, original_height):
    """
    Resizes image while maintaining aspect ratio, then pads it.
    Returns the image, scaling factors, and padding shifts.
    """
    scale = min(target_size[0] / original_width, target_size[1] / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Compute padding in absolute pixels
    delta_w = target_size[0] - new_width
    delta_h = target_size[1] - new_height
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)

    # Apply padding
    final_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

   #Shifts  in box position
    shift_x = left 
    shift_y = top 

    return final_image, scale, scale, shift_x, shift_y  # Return absolute shifts


def pad_then_resize(image, target_size, original_width, original_height):
    """
    Pads an image to make it square, then resizes.
    Returns the image, scaling factors, and padding shifts.
    """
    max_side = max(original_width, original_height)
    
    # Compute padding in absolute pixels
    delta_w = max_side - original_width
    delta_h = max_side - original_height
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)

    # Apply padding before resizing
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    final_image = cv2.resize(padded_image, target_size)

    # Compute scale factors
    scale_x = target_size[0] / max_side
    scale_y = target_size[1] / max_side

    # Compute correct shifts in the new coordinate space
    shift_x = left * scale_x
    shift_y = top * scale_y

    return final_image, scale_x, scale_y, shift_x, shift_y


if __name__ == "__main__":
    logging.info(resize_image_with_annotations(image_path="data/images/maksssksksss0.png",
                                               xml_path="data/annotations/maksssksksss0.xml"))
