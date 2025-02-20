import unittest
import os 
from data_processing.convert_to_yolo import convert_to_yolo_format
from data_processing.resize_images import resize_image_with_annotations

#Helper function to read yolo text files
def read_text_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file for testing is missing: {file_path}")

    with open(file_path,"r") as file:
        return [line.strip() for line in file.readlines()]

class TestYOLOOutput(unittest.TestCase):
    def test_resize(self):
        # Path for img and xml file used for testing
        image_file_path = "data/images/source/maksssksksss1.png"
        annotation_file_path =  "data/annotations/source/maksssksksss1.xml"
        
        #Check if image file exists in directory
        if not os.path.exists(image_file_path):
            raise FileNotFoundError(f"Required  image file for testing is missing: {image_file_path}")
        
        #Check if xml file exists in directory
        if not os.path.exists(annotation_file_path):
            raise FileNotFoundError(f"Required annotation file for testing is missing: {annotation_file_path}")
        
        image_bbox = resize_image_with_annotations(image_file_path,annotation_file_path,method = "resize")
        test_tensor = image_bbox[0]

        # Test the shape of image tensor 
        test_tensor_shape = test_tensor.shape
        self.assertEqual(test_tensor_shape, (3, 224, 224), f"Expected shape (3, 224, 224), but got {test_tensor_shape}")

        # Test the yolo output (text annotations) -> This tests that create_files and convert_to_yolo are matching
        yolo_format_list = convert_to_yolo_format(image_bbox)
        text_format_list = read_text_file(r"data\annotations\resize\maksssksksss1.txt")

        self.assertEqual(yolo_format_list,text_format_list,f"Yolo output: {yolo_format_list} is not the same as the text file stored {text_format_list}")

        #Check if resize correctly modifies bounding box position
        actual_output = ['1 0.841518 0.328125 0.084821 0.227679', 
                       '1 0.604911 0.352679 0.093750 0.223214', 
                       '1 0.765625 0.444196 0.040179 0.147321', 
                       '1 0.395089 0.604911 0.075893 0.263393', 
                       '1 0.209821 0.537946 0.053571 0.191964', 
                       '1 0.511161 0.511161 0.075893 0.165179', 
                       '1 0.078125 0.529018 0.058036 0.129464', 
                       '1 0.955357 0.540179 0.071429 0.187500', 
                       '0 0.241071 0.462054 0.071429 0.209821']

        self.assertEqual(actual_output,text_format_list)
    
    def test_resize_pad(self):
        # Path for img and xml file used for testing
        image_file_path = "data/images/source/maksssksksss1.png"
        annotation_file_path =  "data/annotations/source/maksssksksss1.xml"
        
        #Check if image file exists in directory
        if not os.path.exists(image_file_path):
            raise FileNotFoundError(f"Required  image file for testing is missing: {image_file_path}")
        
        #Check if xml file exists in directory
        if not os.path.exists(annotation_file_path):
            raise FileNotFoundError(f"Required annotation file for testing is missing: {annotation_file_path}")
        
        image_bbox = resize_image_with_annotations(image_file_path,annotation_file_path,method = "resize_pad")
        test_tensor = image_bbox[0]

        # Test the shape of image tensor 
        test_tensor_shape = test_tensor.shape
        self.assertEqual(test_tensor_shape, (3, 224, 224), f"Expected shape (3, 224, 224), but got {test_tensor_shape}")

        # Test the yolo output (text annotations) -> This tests that create_files and convert_to_yolo functions are matching
        yolo_format_list = convert_to_yolo_format(image_bbox)
        text_format_list = read_text_file(r"data\annotations\resize_pad\maksssksksss1.txt")

        self.assertEqual(yolo_format_list,text_format_list,f"Yolo output: {yolo_format_list} is not the same as the text file stored {text_format_list}")

        #Check if resize correctly modifies bounding box position
        actual_output = ['1 0.841518 0.430804 0.084821 0.084821', 
                         '1 0.604911 0.439732 0.093750 0.084821', 
                         '1 0.765625 0.475446 0.040179 0.058036', 
                         '1 0.395089 0.537946 0.075893 0.102679', 
                         '1 0.209821 0.511161 0.053571 0.075893', 
                         '1 0.511161 0.502232 0.075893 0.066964', 
                         '1 0.078125 0.508929 0.058036 0.053571', 
                         '1 0.955357 0.513393 0.071429 0.071429', 
                         '0 0.241071 0.482143 0.071429 0.080357']

        self.assertEqual(actual_output,text_format_list)

    def test_pad_resize(self):
        # Path for img and xml file used for testing
        image_file_path = "data/images/source/maksssksksss1.png"
        annotation_file_path =  "data/annotations/source/maksssksksss1.xml"
        
        #Check if image file exists in directory
        if not os.path.exists(image_file_path):
            raise FileNotFoundError(f"Required  image file for testing is missing: {image_file_path}")
        
        #Check if xml file exists in directory
        if not os.path.exists(annotation_file_path):
            raise FileNotFoundError(f"Required annotation file for testing is missing: {annotation_file_path}")
        
        image_bbox = resize_image_with_annotations(image_file_path,annotation_file_path,method = "pad_resize")
        test_tensor = image_bbox[0]

        # Test the shape of image tensor 
        test_tensor_shape = test_tensor.shape
        self.assertEqual(test_tensor_shape, (3, 224, 224), f"Expected shape (3, 224, 224), but got {test_tensor_shape}")

        # Test the yolo output (text annotations) -> This tests that create_files and convert_to_yolo functions are matching
        yolo_format_list = convert_to_yolo_format(image_bbox)
        text_format_list = read_text_file(r"data\annotations\pad_resize\maksssksksss1.txt")

        self.assertEqual(yolo_format_list,text_format_list,f"Yolo output: {yolo_format_list} is not the same as the text file stored {text_format_list}")

        #Check if resize correctly modifies bounding box position
        actual_output = ['1 0.841518 0.430804 0.084821 0.084821', 
                         '1 0.604911 0.441964 0.093750 0.089286', 
                         '1 0.765625 0.475446 0.040179 0.058036', 
                         '1 0.395089 0.537946 0.075893 0.102679', 
                         '1 0.209821 0.511161 0.053571 0.075893', 
                         '1 0.511161 0.502232 0.075893 0.066964', 
                         '1 0.078125 0.511161 0.058036 0.049107', 
                         '1 0.955357 0.513393 0.071429 0.071429', 
                         '0 0.241071 0.484375 0.071429 0.084821']

        self.assertEqual(actual_output,text_format_list)
if __name__ == "__main__":
    text_format_list = read_text_file(r"data\annotations\pad_resize\maksssksksss1.txt")
    print(text_format_list)
    unittest.main()





            




        

        
            



