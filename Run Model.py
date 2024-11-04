from datetime import datetime
StartTime = datetime.now()

import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the model
model_path = 'Person_Dog_Rec.h5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess the input image
def load_and_preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize the image to the input shape of the model
    img = cv2.resize(img, (256, 256))
    # Normalize the image
    img = img.astype('float32') / 255.0
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Load and preprocess a sample image
image_path = '1.jpeg'  # Change this to your test image path
input_image = load_and_preprocess_image(image_path)

# Make a prediction
prediction = model.predict(input_image)
predicted_class = (prediction > 0.5).astype(int)  # Thresholding for binary classification

# Output the prediction
if predicted_class[0][0] == 1:
    print("Predicted class: Dog")
    reg="Dog"
else:
    print("Predicted class: Human")
    reg="Human"

print(f'BenchMark Time was:{(datetime.now() - StartTime)}')


plt.imshow(input_image[0])  
plt.axis('off')
plt.title(reg)
plt.show()

