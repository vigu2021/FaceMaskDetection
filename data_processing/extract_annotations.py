import os
import logging 
import xml.etree.ElementTree as ET


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_annotations(xml_file = "data/annotations/maksssksksss0.xml"):
    result = {}  # Dictionary to store the extracted data
    tree = ET.parse(xml_file)
    root = tree.getroot() #Grab the root
    
    # Step 1: Extract size of image
    try: 
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)

        logger.info(f"{xml_file} : Sucessfully extracted size attributes of image!. Height:{height}, width:{width}, depth:{depth}")
        result['image_size'] = {'width': width, 'height': height, 'depth': depth}

    except AttributeError as e:
        logger.error(f"{xml_file}: Missing one of the size elements (width, height, depth) - {e}")
    except FileNotFoundError:
        logger.error(f"{xml_file}: File not found.")
    except ET.ParseError as e:
        logger.error(f"{xml_file}: XML parsing error - {e}")
    except Exception as e:
        logger.error(f"{xml_file}: Unexpected error occurred - {e}")
    
    # Step 2: Extract bounding box position
    
    try:
        bounding_boxes = []

        objects = root.findall('object')
        for object in objects:
            with_mask = object.find('name').text
            bounding_box = object.find('bndbox')
            x_min = int(bounding_box.find('xmin').text)
            y_min = int(bounding_box.find('ymin').text)
            x_max = int(bounding_box.find('xmax').text)
            y_max = int(bounding_box.find('ymax').text)
            logger.info(f"Person {with_mask} and Bounding box at {x_min,y_min} to {x_max,y_max}")


            # Append bounding box data to the list
            bounding_boxes.append({
                'label': with_mask,
                'coordinates': {
                    'xmin': x_min,
                    'ymin': y_min,
                    'xmax': x_max,
                    'ymax': y_max
                }
            })

        result['annotations'] = bounding_boxes
    except AttributeError as e:
        logger.error(f"{xml_file}: Missing one of these attributes elements (x_min,y_min,x_max,y_max and wih_mask) - {e}")
    except FileNotFoundError:
        logger.error(f"{xml_file}: File not found.")
    except ET.ParseError as e:
        logger.error(f"{xml_file}: XML parsing error - {e}")
    except Exception as e:
        logger.error(f"{xml_file}: Unexpected error occurred - {e}")

    return result
    
print(extract_annotations_to_txt())


    

