import unittest
import os 
from data_processing.convert_to_yolo import convert_to_yolo_format
from data_processing.resize_images import resize_images,resize_then_pad,pad_then_resize
class TestYOLOOutput(unittest.TestCase):
    def test_yolo_output_format(self):
        # Path for xml file used for testing
        test_file_path =  "yolo_test.xml"
        
        #Check if XML file exists in directory
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"Required file for testing is missing: {test_file_path}")
        
        yolo_text_output = convert_to_yolo_format(test_file_path)
        
        #Expected output 
        expected_output = [
            "1 0.2162 0.1914 0.0925 0.0931",
            "1 0.4150 0.2086 0.0600 0.0931",
            "1 0.6250 0.1776 0.0600 0.1207",
            "0 0.8738 0.1655 0.0825 0.1172"
        ]
        self.assertEqual(yolo_text_output, expected_output)
    
    #Now test resizing of images
    def test_resize_images(self):

        source_image_file = 'data/images'
        if not os.path.exists(source_image_file):
            raise FileNotFoundError(f"Required file for testing is missing: {source_image_file}")
        
        image_names = os.listdir('data/images')
        no_of_images = len(image_names)

        image_tensor_resize = resize_images(method = 'resize')
        image_tensor_resize_pad = resize_images(method = 'resize_pad')
        image_tensor_pad_resize = resize_images(method = 'pad_resize')
        
        #Test to see if all files are extracted from XML to tensor. 
        self.assertEqual(no_of_images, image_tensor_resize.shape[0])
        self.assertEqual(no_of_images, image_tensor_resize_pad.shape[0])
        self.assertEqual(no_of_images, image_tensor_pad_resize.shape[0])

        #Test to see if dimensions are (224,224)
        self.assertTrue(image_tensor_resize.shape[1] == 224 and image_tensor_resize.shape[2] == 224)
        self.assertTrue(image_tensor_resize_pad.shape[1] == 224 and image_tensor_resize_pad.shape[2] == 224)
        self.assertTrue(image_tensor_pad_resize.shape[1] == 224 and image_tensor_pad_resize.shape[2] == 224)

if __name__ == "__main__":
    unittest.main()




            




        

        
            



