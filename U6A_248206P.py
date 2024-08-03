import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def inter_means_threshold(image):

    if len(image.shape) == 3:
        # convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    threshold = np.mean(image) # Initialize threshold with the mean pixel value
    
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

def process_image(image_path,figure_id):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Compute the threshold using the inter-means algorithm
    threshold = inter_means_threshold(image)
    print(f"The computed threshold for {image_path} is: {threshold}")
    
    # Apply the threshold to binarize the image
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # Save binary image to processed directory
    output_path = f'./processed/{os.path.basename(image_path)}'
    cv2.imwrite(output_path, binary_image)
    print(f"Binary image saved to {output_path}")

    # Display imaged
    plt.figure(figure_id,figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')

def main(directory_path):
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]
    # Process each image file
    fig_id = 1
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        process_image(image_path,fig_id)
        fig_id += 1
    plt.show()

if __name__ == "__main__":
    input_directory_path = './input/'
    main(input_directory_path)
