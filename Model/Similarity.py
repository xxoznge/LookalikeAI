import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
from keras.models import Model

# 이미지 파일이 저장된 폴더 경로
image_folder = 'C:/Users/SOJUNG/Desktop/LookalikeAI/croppingImages'

# 미리 학습한 모델 로드
model = load_model('face_recognition_model.h5')

# 모델에서 특성 벡터를 추출하는 새로운 모델 생성
feature_extraction_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# 레이블 인코딩
label_binarizer = LabelBinarizer()

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

# 레이블 인코딩
y_train_encoded = label_binarizer.fit_transform(y_train)
y_val_encoded = label_binarizer.transform(y_val)
y_test_encoded = label_binarizer.transform(y_test)

# 특성 벡터를 각 연예인마다 출력
for i, celebrity in enumerate(label_binarizer.classes_):
    images_for_celebrity = X_train[y_train == celebrity]
    feature_vectors = feature_extraction_model.predict(images_for_celebrity)
    print(f"Feature vectors for {celebrity}:")
    print(feature_vectors)
    print()

your_image_path = 'C:/Users/SOJUNG/Desktop/LookalikeAI/me.jpg'
your_image = cv2.imread(your_image_path)

if your_image is not None:
    your_image = cv2.resize(your_image, (128, 128))
    your_image = your_image.astype('float32') / 255.0
else:
    print("Your image could not be loaded.")

your_feature_vector = feature_extraction_model.predict(np.expand_dims(your_image, axis=0))

from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(your_feature_vector, feature_vectors)

top_n = 1
most_similar_indices = np.argsort(similarities, axis=1)[:, -top_n:]
top_n_celebrities = [label_binarizer.classes_[index] for index in most_similar_indices[0]]

print(f"Your image is most similar to: {', '.join(top_n_celebrities)}")

# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f"Test accuracy: {test_acc}")