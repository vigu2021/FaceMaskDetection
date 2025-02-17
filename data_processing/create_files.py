import os
import logging
from .resize_images import resize_image_with_annotations
from .convert_to_yolo import convert_to_yolo_format
from torchvision.transforms.functional import to_pil_image # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_files(method, image_dir='data/images', annotations_dir='data/annotations', override=False):
    # Check if valid method
    valid_methods = ['resize', 'pad_resize', 'resize_pad']
    if method not in valid_methods:
        logger.error(f"{method} is not a valid method in {valid_methods}")
        raise ValueError(f"{method} is not a valid method in {valid_methods}")

    # Check that override is a boolean
    if not isinstance(override, bool):
        logger.error("Override parameter has to be a boolean!")
        return

    # Create directories if they don't exist
    image_full_path = os.path.join(image_dir, method)
    annotation_full_path = os.path.join(annotations_dir, method)
    os.makedirs(image_full_path, exist_ok=True)
    os.makedirs(annotation_full_path, exist_ok=True)

    # Get all image files
    image_files = os.listdir(image_dir)
    annotation_files = os.listdir(annotations_dir)

    for image_name in image_files:
        image, extension = os.path.splitext(image_name)
        image_save_path = os.path.join(image_full_path, f"{image}.jpg")  # Save as JPG

        # If override is True, delete the old file; else, terminate if duplicate exists
        if os.path.exists(image_save_path):
            if override:
                os.remove(image_save_path)  # Delete old image
            else:
                logger.error(f"{image_save_path} already exists! Terminating execution!")
                return

        # Check if image is a valid extension
        valid_extensions = {".jpg", ".png"}
        if extension.lower() not in valid_extensions:
            logger.error(f"Unsupported file type: {image_name}")
            return

        # Ensure corresponding annotation exists as .xml
        source_annotation = os.path.join(annotations_dir, f"{image}.xml")
        if not os.path.exists(source_annotation):
            logger.error(f"{source_annotation} does not exist!")
            return

        # Process image and annotations
        source_image = os.path.join(image_dir, image_name)
        image_bbox = resize_image_with_annotations(source_image, source_annotation, method=method)
        new_image = image_bbox[0]  # Processed image tensor
        yolo_list = convert_to_yolo_format(image_bbox)

        # Convert tensor to image and save as .jpg
        to_pil_image(new_image).save(image_save_path)

        # Save YOLO annotations as .txt
        annotation_save_path = os.path.join(annotation_full_path, f"{image}.txt")
        with open(annotation_save_path, "w") as f:
            for line in yolo_list:
                f.write(" ".join(map(str, line)) + "\n")


        
        


if __name__ == "__main__":
    create_files(method = 'resize_pad')
    create_files(method = 'pad_resize')



    
    
