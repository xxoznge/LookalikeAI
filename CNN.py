import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 이미지 파일이 저장된 폴더 경로
image_folder = 'C:/Users/SOJUNG/Desktop/LookalikeAI/croppingImages'

# 이미지 파일 로드 및 전처리
def preprocess_images(image_folder):
    images = []
    labels = []

    for celebrity_folder in os.listdir(image_folder):
        celebrity_folder_path = os.path.join(image_folder, celebrity_folder)
        label = celebrity_folder

        for image_file in os.listdir(celebrity_folder_path):
            image_path = os.path.join(celebrity_folder_path, image_file)

            # 이미지 로드
            image = cv2.imread(image_path)

            if image is not None:
                # 얼굴 이미지 크기 통일 (예: 128x128)
                image = cv2.resize(image, (128, 128))

                # 이미지 데이터 정규화 (0~1 범위로)
                image = image.astype('float32') / 255.0

                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)

# 이미지 데이터 전처리
images, labels = preprocess_images(image_folder)

# 데이터를 train, validation, test 세트로 나누기
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

from keras import layers, models
import matplotlib.pyplot as plt

# 레이블 인코딩
label_binarizer = LabelBinarizer()
y_train_encoded = label_binarizer.fit_transform(y_train)
y_val_encoded = label_binarizer.transform(y_val)
y_test_encoded = label_binarizer.transform(y_test)

# 모델 설계
model = models.Sequential([
    layers.Conv2D(16, (8, 8), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(36, (5, 5), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_binarizer.classes_), activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
batch_size = 16
epochs = 200

history = model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val_encoded))

# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f"Test accuracy: {test_acc}")

# 모델 저장
model.save('face_recognition_model.h5')

# 손실 정보 추출
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_loss, label='Train Loss', marker='o')
plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()