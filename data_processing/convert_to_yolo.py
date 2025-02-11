import os
from extract_annotations import extract_annotations 
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_yolo_format(xml_file):
    """
    Converts bounding box annotations from an XML file to YOLO format.

    This function extracts image size and bounding box annotations from an XML file 
    (commonly used in Pascal VOC datasets), converts the bounding box coordinates to 
    YOLO format, and returns the formatted data as a list of strings.

    YOLO Format Structure:
    -----------------------
    Each annotation is formatted as:
        <class_id> <x_center> <y_center> <width> <height>

    Where:
        - class_id   : Integer (1 for 'with_mask', 0 for 'without_mask')
        - x_center   : Normalized x-coordinate of the bounding box center (0 to 1)
        - y_center   : Normalized y-coordinate of the bounding box center (0 to 1)
        - width      : Normalized width of the bounding box (0 to 1)
        - height     : Normalized height of the bounding box (0 to 1)

    Parameters:
    -----------
    xml_file : str
        The path to the XML annotation file.

    Returns:
    --------
    list of str or None
        A list of YOLO-formatted annotation strings. Each string corresponds to an object 
        in the image. Returns None if an error occurs during extraction or conversion.
    """
    try:
        annotations = extract_annotations(xml_file)
        image_size = annotations['image_size']
        width,height,depth = image_size['width'],image_size['height'],image_size['depth']
    except KeyError as e:
        logging.debug("{e}: Failed to find keys")
        return 
    except Exception as e:
        logging.error(f"Failed to extract due to {e}")
        return 

    try:
        bounding_boxes = annotations['annotations']
        
        yolo_data = []
        for bounding_box in bounding_boxes:
            label = 1 if bounding_box['label']=='with_mask' else 0 
            coords = bounding_box['coordinates']

            # Convert bounding box to YOLO format
            x_center = ((coords['xmin'] + coords['xmax']) / 2)/ width
            y_center = ((coords['ymin'] + coords['ymax']) / 2)/ height
            bbox_width = (coords['xmax'] -coords['xmin']) / width
            bbox_height = (coords['ymax'] - coords['ymin']) / height

            # YOLO format: class x_center y_center width height
            yolo_format = f"{label} {x_center:.4f} {y_center:.4f} {bbox_width:.4f} {bbox_height:.4f}"
            yolo_data.append(yolo_format)

    except KeyError as e:
        logging.debug(f"{e}: Failed to find bounding box keys.")
        return
    except Exception as e:
        logging.error(f"Failed to convert bounding boxes for {xml_file} due to: {e}")
        return

    return yolo_data








