import streamlit as st
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from PIL import Image  # Use PIL for image processing
import matplotlib.pyplot as plt

# Load the model
model_path = '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/best_model.h5'
model = load_model(model_path)

# Streamlit app
st.title("Emotion Recognition App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    img = Image.open(uploaded_file).convert("RGB")  # Use PIL to open and convert image to RGB
    img = img.resize((64, 64))  # Resize the image
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(img_array)

    # Determine predicted class and confidence
    predicted_class = "Happy" if prediction[0] >= 0.5 else "Sad"
    confidence = prediction[0] if predicted_class == "Happy" else 1 - prediction[0]
    confidence_scalar = float(confidence)

    # Display the image and prediction
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f'Predicted Class: {predicted_class} (Confidence: {confidence_scalar:.2f})')



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.metrics import BinaryAccuracy


train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='elu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[BinaryAccuracy()])

save_dir = '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET'

os.makedirs(save_dir, exist_ok=True)

filepath = os.path.join(save_dir, 'best_model.h5')

checkpoint = ModelCheckpoint(filepath,
                              monitor='val_binary_accuracy',
                              save_best_only=True,
                              mode='max',
                              verbose=1)

model.fit(train_generator, validation_data=validation_generator, epochs= 10, callbacks=[checkpoint])

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')

img_path = '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/validation/happy/10023.jpg'
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
print(f"Predicted Probability: {prediction[0]}")
print(f"Predicted Class: {round(prediction[0][0])}")

model.fit(train_generator, epochs=10, validation_data=validation_generator)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')

from keras.preprocessing import image
import numpy as np

img_path = '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/validation/happy/10023.jpg'
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
print(f"Predicted Probability: {prediction[0]}")
print(f"Predicted Class: {round(prediction[0][0])}")

from PIL import Image

import os
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.metrics import BinaryAccuracy

img_path = '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/validation/happy/10023.jpg'
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0
prediction = model.predict(img_array)

class_names = ['happy', 'sad']

predicted_class_index = np.argmax(prediction[0])
predicted_class = class_names[predicted_class_index]
probability = np.max(prediction)

print(f'The predicted class is: {predicted_class}')
print(f'The probability is: {probability}')

img = Image.open(img_path)
plt.imshow(img)
plt.show()

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/sad/096b101cd34c08b79bb0e6aec2b3afb5.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)

predicted_class = "Happy" if prediction[0] >= 0.5 else "Sad"
confidence = prediction[0] if predicted_class == "Happy" else 1 - prediction[0]
confidence_scalar = float(confidence)

img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted Class: {predicted_class} (Confidence: {confidence_scalar:.2f})')
plt.show()

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/sad/394617911.jpg'
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)

predicted_class = "Happy" if prediction[0] >= 0.5 else "Sad"
confidence = prediction[0] if predicted_class == "Happy" else 1 - prediction[0]
confidence_scalar = float(confidence)

img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted Class: {predicted_class} (Confidence: {confidence_scalar:.2f})')
plt.show()

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = '/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/happy/_happy_jumping_on_beach-40815.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)

predicted_class = "Happy" if prediction[0] >= 0.5 else "Sad"
confidence = prediction[0] if predicted_class == "Happy" else 1 - prediction[0]
confidence_scalar = float(confidence)

img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted Class: {predicted_class} (Confidence: {confidence_scalar:.2f})')
plt.show()
