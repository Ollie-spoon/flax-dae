from PIL import Image
import numpy as np

# This python script uses the buffer protocol, through the pillow library
# to convert an image to a numpy array, process the array, and convert it 
# back to an image, all with only a signle copy of the image data in memory.

def image_modify(image_array):
    return image_array

# Open an image file
image = Image.open('example.jpg')

# Convert the image to a numpy array
image_array = np.array(image)

# Apply some processing to the image array (e.g., convert to grayscale)
gray_image_array = image_modify(image_array)

# Convert the processed array back to an image
gray_image = Image.fromarray(gray_image_array.astype('uint8'))

# Save the processed image
gray_image.save('gray_example.jpg')
