import os 
import logging 
from convert_to_yolo import convert_to_yolo_format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def xml_to_yolo(annotation_folder,output_folder):

    """
    Converts all XML annotation files in a folder to YOLO format and saves them as .txt files.

    Parameters:
    -----------
    annotation_folder : str
        Path to the folder containing XML annotation files.
    
    output_folder : str
        Path to the folder where YOLO formatted .txt files will be saved.

    Returns:
    --------
    None
    """

    if not os.path.exists(annotation_folder):
        logger.error(f"The folder does not exist at: {os.path.abspath(image_folder)}")
        logger.error("Operation stopped! File not found.")
        return None
    annotation_files = os.listdir(annotation_folder)

    for file in annotation_files:
        if file.lower().endswith('.xml'):
            try:
                full_file_path = os.path.join(annotation_folder,file)
                yolo_list = convert_to_yolo_format(full_file_path)
                
                if yolo_list:
                    txt_filename = os.path.splitext(file)[0] + '.txt'
                    output_file_path = os.path.join(output_folder, txt_filename)
                # Write YOLO annotations to the .txt file
                    with open(output_file_path, 'w') as f:
                        for line in yolo_list:
                            f.write(line + '\n')

                    logger.info(f"Successfully converted {file} to YOLO format and saved as {txt_filename}")
                else:
                    logger.warning(f"No YOLO data extracted from {file}")

            except FileNotFoundError:
                logger.error(f"File doesn't exist: {file} in folder {annotation_folder}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing {file}: {e}")   
           
    

# Script to convert xml to yolo format text files
'''
if __name__ == "__main__":
    xml_to_yolo('data/annotations','data/yolo_format')
'''

# Check if number of files is the same as in annotations

print(len(os.listdir('data/yolo_format')))