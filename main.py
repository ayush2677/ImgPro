import cv2
import tensorflow as tf
from tkinter import Tk, filedialog

def preprocess_image(image):
    # Resize the image to the expected input shape of the model
    resized_image = cv2.resize(image, (256, 256))

    # Normalize the image
    normalized_image = resized_image / 255.0

    # Expand the dimensions to match the expected input shape
    preprocessed_image = tf.expand_dims(normalized_image, axis=0)

    return preprocessed_image

# Load the pre-trained model
model = tf.keras.models.load_model(r'/content/drive/MyDrive/checkpoints/model1_checkpoint.h5')

# Initialize the Tkinter root window
root = Tk()
root.withdraw()  # Hide the root window

# Ask the user to select an image file
file_path = filedialog.askopenfilename(filetypes=[('Image files', '*.jpg;*.jpeg;*.png')])

# Read the original image
original_image = cv2.imread(file_path)

# Pre-process the original image
preprocessed_image = preprocess_image(original_image)

# Perform inference using the loaded model
processed_image = model.predict(preprocessed_image)

# Convert processed image array to appropriate data type
processed_image = (processed_image * 255).astype('uint8')

# Display the original and processed images
cv2.imshow('Original Image', original_image)
cv2.imshow('Processed Image', processed_image[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
