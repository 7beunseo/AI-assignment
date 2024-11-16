import numpy as np
import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

IMAGE_SIZE = (128,128)

def load_train_data(folder_path):
    X = []
    y = []
    class_names = os.listdir(folder_path)
    print(class_names)

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            X.append(image)
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names


# Load image data
def load_test_data(folder_path):
    X = []
    filenames = []
    for image_name in os.listdir(folder_path):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_name)
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            X.append(image)
            filenames.append(image_name)
    X = np.array(X)
    return X, filenames

# # Load training and testing data
train_folder = './flowers-dataset/train'
test_folder = './flowers-dataset/test'
X_train, y_train, class_names = load_train_data(train_folder)
X_test, test_filenames = load_test_data(test_folder)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 데이터 증강
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Normalize values
# 정규화 수행
X_train = X_train / 255.0
X_test = X_test / 255.0

print("X_train_split.shape:", X_train.shape)
print("y_train_split.shape:", y_train.shape)
print("X_test_split.shape:", X_test.shape)
print("y_test_split.shape:", y_test.shape)

# plt.figure(figsize=(10, 2))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(X_train[i])
#     plt.title(class_names[y_train[i]])  # 이미지에 해당하는 클래스 이름 표시
#     plt.axis('off')
# plt.show()


# 레이블을 one-hot encoding으로 변환
train_labels = keras.utils.to_categorical(y_train, 5)
test_labels = keras.utils.to_categorical(y_test, 5)



print('train_labels.shape (one-hot) =', train_labels.shape)
print('test_labels.shape (one-hot) =', test_labels.shape)

# CNN
# 3채널일 경우 그대로 정규화 수행
# train_images = X_train[:, :, :, np.newaxis] 데이터으 개수, 행 수, 열 수, 채널 수
# test_images = X_test[:, :, :, np.newaxis]

# CNN 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.summary()   # 모델의 구조를 요약하여 살펴보자

model.compile(optimizer='adam',\
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# 증강된 데이터를 사용한 학습
hist = model.fit(datagen.flow(X_train, train_labels, batch_size=16),
                 epochs=150, validation_data=(X_test, test_labels))


plt.plot(hist.history['accuracy'], 'b-')
plt.plot(hist.history['val_accuracy'], 'r--')
plt.show()

test_loss, test_acc = model.evaluate(X_test,  test_labels, verbose=2)
print('테스트 정확도:', test_acc) #  0.6418181657791138


mnist_lbl = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']

images = X_test[:25]
pred = np.argmax(model.predict(images), axis=1)
print('예측값 =', pred)
print('실제값 =', np.argmax(test_labels[:25], axis=1))


def plot_images(images, labels, class_names, images_per_row=5):
    n_images = len(images)
    n_rows = (n_images - 1) // images_per_row + 1
    fig, ax = plt.subplots(n_rows, images_per_row,
                           figsize=(images_per_row * 2, n_rows * 2))

    for i in range(n_rows):
        for j in range(images_per_row):
            if i * images_per_row + j >= n_images:
                break
            img_idx = i * images_per_row + j
            a_image = images[img_idx] * 255.0  # 역정규화 수행
            a_image = a_image.astype(np.uint8)  # 정수형 변환

            axis = ax[i, j] if n_rows > 1 else ax[j]
            axis.imshow(a_image)  # 컬러 이미지 시각화
            axis.set_title(class_names[labels[img_idx]])
            axis.axis('off')

    plt.tight_layout()
    plt.show()


# 수정된 plot_images 함수 호출
plot_images(images, pred, class_names, images_per_row=5)

pred = np.argmax(model.predict(X_test), axis=1)
actual = np.argmax(test_labels, axis=1)  # test_labels를 원래 형태로 변환

print('예측값 =', pred)
print('실제값 =', actual)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(actual, pred)
plt.matshow(conf_mat)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print(conf_mat)