import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def inter_means_threshold(image):
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize threshold with the mean pixel value
    threshold = np.mean(image)
    
    # Iterate until convergence
    while True:
        # Segment the image into two groups
        lower_group = image[image <= threshold]
        upper_group = image[image > threshold]
        
        # Calculate the mean values of the groups
        lower_mean = np.mean(lower_group)
        upper_mean = np.mean(upper_group)
        
        # Compute the new threshold
        new_threshold = (lower_mean + upper_mean) / 2
        
        # Check for convergence
        if abs(new_threshold - threshold) < 1:
            break
        
        threshold = new_threshold
    
    return int(threshold)

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Compute the threshold using the inter-means algorithm
    threshold = inter_means_threshold(image)
    print(f"The computed threshold for {image_path} is: {threshold}")
    
    # Apply the threshold to binarize the image
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # Save the binary image
    output_path = os.path.join(os.path.dirname(image_path), f'/processed/binary_{os.path.basename(image_path)}')
    cv2.imwrite(output_path, binary_image)
    print(f"Binary image saved to {output_path}")

    # Display the original and binary images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Binary Image")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')

    plt.show()

# /content/drive/MyDrive/Academics/MSc/Semester 2/CS5513-Computer Vision/image-segmentation-algorithm/input-images

def main(directory_path):
    # Get a list of all .jpg files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]
    
    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        process_image(image_path)

if __name__ == "__main__":
    directory_path = '/content/drive/MyDrive/Academics/MSc/Semester 2/CS5513-Computer Vision/image-segmentation-algorithm/input-images/'  # Replace with the path to your directory
    main(directory_path)
