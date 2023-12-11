from flask import Flask, render_template, request, abort, jsonify
import cv2
import os
import numpy as np
from keras.models import load_model
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


app = Flask(__name__, template_folder='Fronted', static_folder='Fronted/static')

# 미리 학습한 모델 로드
model = load_model('model.keras')
 
# 모델에서 특성 벡터를 추출하는 새로운 모델 생성
feature_extraction_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# 이미지 파일이 저장된 폴더 경로
image_folder = 'C:/Users/SOJUNG/Desktop/LookalikeAI/croppingImages'

# 레이블 인코딩
label_binarizer = LabelBinarizer()

# 이미지 파일 로드 및 전처리 함수
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

# 유사도를 계산하여 연예인 찾기
def find_similar_celebrity(user_image_path):
    # 이미지 로드 및 전처리
    user_image = cv2.imread(user_image_path)

    if user_image is not None:
        user_image = cv2.resize(user_image, (128, 128))
        user_image = user_image.astype('float32') / 255.0
    else:
        abort(400, description="Your image could not be loaded.")

    user_feature_vector = feature_extraction_model.predict(np.expand_dims(user_image, axis=0))

    # 특성 벡터를 각 연예인마다 출력
    celebrity_similarity_averages = []

    for i, celebrity in enumerate(label_binarizer.classes_):
        images_for_celebrity = X_train[y_train == celebrity]
        feature_vectors = feature_extraction_model.predict(images_for_celebrity)

        # 모든 이미지와의 유사도 계산
        similarities = cosine_similarity(user_feature_vector.reshape(1, -1), feature_vectors)

        # 각 이미지와의 유사도 평균 계산
        num_samples = len(images_for_celebrity)
        if similarities.shape[1] == num_samples:
            total_similarity = np.sum(similarities)
            celebrity_similarity_average = total_similarity / num_samples
            celebrity_similarity_averages.append((celebrity, celebrity_similarity_average))
        else:
            print(f"Error: Number of samples does not match the size of the similarities array.")

    # 가장 유사도 평균이 높은 연예인 찾기
    most_similar_celebrity, highest_similarity_average = max(celebrity_similarity_averages, key=lambda x: x[1])

    return {"most_similar_celebrity": most_similar_celebrity, "highest_similarity_average": highest_similarity_average}

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # Check if the 'image' key is in the request.files dictionary
        if 'file' not in request.files:
            print(request.files)
            abort(400, description="No file part in the request.")

        user_image = request.files['file']
        user_image_path = 'C:/Users/SOJUNG/Desktop/LookalikeAI/test/' + user_image.filename
        user_image.save(user_image_path)

        # 유사도 계산 로직 호출
        result = find_similar_celebrity(user_image_path)

        # 결과 클라이언트에게 전달
        app.logger.info(f"Result: {result}")
        return render_template('result.html', result=result)
        
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
