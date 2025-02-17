import os
from .resize_images import resize_image_with_annotations
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_yolo_format(image_bbox):
    """
    Converts bounding box annotations to the YOLO format.

    This function takes an image and its associated bounding box annotations,
    then converts the bounding box coordinates into the YOLO format, which is:

        class_id x_center y_center width height

    where:
        - `class_id` → Integer representing the class label (e.g., 1 = "with_mask", 0 = "without_mask").
        - `x_center` → Normalized x-coordinate of the bounding box center (relative to image width).
        - `y_center` → Normalized y-coordinate of the bounding box center (relative to image height).
        - `width` → Normalized width of the bounding box.
        - `height` → Normalized height of the bounding box.

    Parameters
    ----------
    image_bbox : tuple
        A tuple `(image_tensor, annotations)`, where:
        - `image_tensor` → The resized image (not used in this function).
        - `annotations` → A dictionary containing:
            - `"image_size"`: Dictionary with `'width'` and `'height'` of the image.
            - `"annotations"`: List of bounding box dictionaries, each with:
                - `'label'`: The class label (e.g., "with_mask").
                - `'coordinates'`: A dictionary with:
                    - `'xmin'`: Minimum x-coordinate of the bounding box.
                    - `'ymin'`: Minimum y-coordinate of the bounding box.
                    - `'xmax'`: Maximum x-coordinate of the bounding box.
                    - `'ymax'`: Maximum y-coordinate of the bounding box.

    Returns
    -------
    list of str
        A list of strings, where each entry represents an object annotation in YOLO format:
        ```
        "class_id x_center y_center width height"
        ```
        - The values are **normalized** between `0` and `1`, ensuring compatibility with YOLO.
"""
    
    # Has to be lenght 2 and a tuple
    if len(image_bbox) != 2 or not isinstance(image_bbox,tuple):
        logging.error(f"❌ Error: Expected a tuple (image_tensor, annotations), but got {type(image_bbox)}")
        raise TypeError()
    
    try:
        image_size = image_bbox[1]['image_size']
        annotations = image_bbox[1]['annotations']
    except KeyError as e:
        logging.error(f"❌ Error: Can't find annotations or size key!")
        raise
    
    height = image_size[0]
    width = image_size[1]

    yolo_format_list = []

    for bbox in annotations:
        class_id = 1 if bbox['label'] == 'with_mask' else 0 # 1 if with_mask else 0
        #Coordinates
        coord = bbox['coordinates']
        x_min = coord['xmin']
        y_min = coord['ymin']
        x_max = coord['xmax']
        y_max = coord['ymax']

        # Yolo expects class_id x_center y_center width height
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        #Normalise from 0 to 1
        x_center /= width
        y_center /= height
        bbox_width /= width
        bbox_height /= height

        yolo_string = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
        yolo_format_list.append(yolo_string)
    
    return yolo_format_list
    

if __name__ == '__main__':
    image_path = "data/images/maksssksksss0.png"
    xml_path = "data/annotations/maksssksksss0.xml"
    image_bbox = resize_image_with_annotations(image_path, xml_path)
    print(convert_to_yolo_format(image_bbox))
