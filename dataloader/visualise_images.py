import torch
import matplotlib.pyplot as plt


def show_image_with_boxes(image_tensor, labels_tensor):
    
    image = image_tensor.permute(1, 2, 0).numpy()  # [C, H, W] → [H, W, C]

    pt, ax = plt.subplots(figsize=(8, 8))  
    ax.imshow(image)  # Display the image

    # Draw bounding boxes
    for label in labels_tensor:
        class_id, x_center, y_center, width, height = label
        img_h,img_w = image.shape[:2]
        
        # Convert YOLO format to pixel coordinates
        x1 = (x_center - width / 2) * img_w
        y1 = (y_center - height / 2) * img_h
        rect_width = width * img_w
        rect_height = height * img_h

        # Draw rectangle for boudning boxes 
        rect = plt.Rectangle((x1, y1), rect_width, rect_height, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)

        ax.text(x1, y1, f'Class: {int(class_id)}', color='red', fontsize=10)

    # Remove axes and display
    ax.axis('off')
    plt.show()