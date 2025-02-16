import os
from .resize_images import resize_image_with_annotations
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_yolo_format(image_bbox):
    
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
