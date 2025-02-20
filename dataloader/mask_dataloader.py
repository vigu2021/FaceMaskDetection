
import torch
from torch.utils.data import Dataset,DataLoader
import cv2
import os
import logging
import matplotlib.pyplot as plt
from .visualise_images import show_image_with_boxes
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CustomYoloDataset(Dataset):
    '''
    Custom Dataset class for loading images and corresponding YOLO-format annotations.

    This dataset is designed for object detection tasks using YOLO models.
    It loads images and their bounding box annotations from specified directories,
    converts images to tensors, and parses YOLO-formatted labels.

    Attributes:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing annotation files.
        images (list): List of image filenames in the directory.
    '''
    def __init__(self,method,images_dir = 'data/images',labels_dir = 'data/annotations'):
        self.image_dir = os.path.join(images_dir,method)
        self.label_dir = os.path.join(labels_dir,method)
        self.images = [image for image in os.listdir(self.image_dir) if image.endswith(('jpg','png'))]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        # Get name of image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir,img_name)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            logging.error(f"Unable to load image at {img_path}")
            raise FileNotFoundError(f"Unable to load image at {img_path}")
            
        # Convert BGR to RGB and normalize
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).float().permute(2,0,1)/255
        
        label_path = os.path.join(self.label_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')

        print(label_path)
        # Load YOLO annotations
        labels = []
        if os.path.exists(label_path):
            with open(label_path,"r") as f:
                for line in f:
                    try:
                        label = list(map(float, line.strip().split()))
                        if len(label) == 5: # Ensure the line has correct format
                            labels.append(label)
                        else:
                            logger.warning(f"Incorrect label format in {label_path}: {line.strip()}")
                    except ValueError:
                        logger.warning(f"Non-numeric value in label file {label_path}: {line.strip()}")
                        
        #Convert to 2D list to tensor
        labels_tensor = torch.tensor(labels,dtype = torch.float32)
        
        logging.info(f"Suceesfully loaded image:{img_path} and {label_path}")

        return image_tensor, labels_tensor 



def custom_collate_fn(batch):
    """
    Custom collate function to handle varying numbers of annotations per image.

    Args:
        batch (list): List of tuples (image_tensor, labels_tensor).

    Returns:
        tuple: Batched images and list of labels.
    """
    images = [item[0] for item in batch]   
    labels = [item[1] for item in batch] 
    images = torch.stack(images, dim=0)    # Shape: (batch_size, C, H, W)

    return images, labels


if __name__ == "__main__":

    # Initialise the Dataset
    dataset = CustomYoloDataset(method = 'resize')

    # Load the data set
    dataloader = DataLoader(dataset,batch_size=2,shuffle=True,num_workers=1,collate_fn=custom_collate_fn)
    
    # Iterate DataLoader 
    for batch_idx,(images, labels) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")
       
        # Iterate through each image in the batch
        for i, image in enumerate(images):
            # Show image with bounding boxes
            show_image_with_boxes(image, labels[i])

        # Ask user if the batch is correct
        batch_check = input("Is this batch correct? (y/n): ").strip().lower()
        if batch_check != 'y':
            print("Batch marked as incorrect. Exiting...")
            break

        # Ask user if they want to see more examples
        more_examples = input("Do you want to see more examples? (y/n): ").strip().lower()
        if more_examples != 'y':
            print("Terminating the visualization.")
            break
        
            
            




#python -m dataloader.mask_dataloader


