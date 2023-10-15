import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Load an example image
img = image.load_img('1.png')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Initialize ImageDataGenerator for augmentations
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Display the original image
plt.figure(figsize=(1, 1))
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Generate augmented images
i = 0
plt.figure(figsize=(12, 12))
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(3, 3, i + 1)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 9 == 0:
        break
plt.show()
