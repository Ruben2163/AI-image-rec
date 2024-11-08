import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the trained generator model
generator = tf.keras.models.load_model('models/generator_model.h5')

# Define the noise dimension (must match the dimension used during training)
noise_dim = 100

# Generate random noise
noise = tf.random.normal([16, noise_dim])  # Generate 16 images

# Generate images using the generator
generated_images = generator(noise, training=False)

# Display the generated images
plt.figure(figsize=(4, 4))
for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
plt.show()