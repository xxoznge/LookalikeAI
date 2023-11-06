import cv2
import dlib
import os

# 이미지 파일이 들어있는 상위 폴더 경로
image_root_folder = 'C:/Users/SOJUNG/Desktop/LookalikeAI/AutoCrawler/download'

# 얼굴 이미지를 저장할 폴더 경로
output_folder = 'C:/Users/SOJUNG/Desktop/LookalikeAI/croppingImages'

# dlib 얼굴 탐지기 초기화
detector = dlib.get_frontal_face_detector()

# 이미지 확대 비율
scale_factor = 3.0  # 예시로 2.0 배 확대하겠습니다.

def increase_sharpness(image):
    # Unsharp Masking 적용
    gaussian = cv2.GaussianBlur(image, (0, 0), 3)
    sharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharp

# 상위 폴더 내의 모든 하위 폴더 탐색
for celebrity_folder in os.listdir(image_root_folder):
    celebrity_folder_path = os.path.join(image_root_folder, celebrity_folder)

    # 각 연예인 폴더 내의 이미지 파일 경로 얻기
    image_paths = [os.path.join(celebrity_folder_path, file) for file in os.listdir(celebrity_folder_path) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # 얼굴 저장 폴더 생성 (연예인 이름 폴더 내)
    celebrity_output_folder = os.path.join(output_folder, celebrity_folder)
    os.makedirs(celebrity_output_folder, exist_ok=True)

    # 각 이미지 파일에 대한 얼굴 탐지 및 선명도 높이기
    for image_path in image_paths:
        try:
            # 이미지 불러오기
            image = cv2.imread(image_path)

            # 이미지 확대
            large_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

            # 얼굴 탐지
            faces = detector(large_image)

            # 얼굴이 탐지되었는지 확인
            if len(faces) > 0:
                for i, face in enumerate(faces):
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()

                    # 얼굴 부분만 자르기
                    face_image = large_image[y:y+h, x:x+w]

                    # 얼굴 이미지가 비어 있지 않은 경우에만 저장
                    if not face_image is None:
                        # 선명도 높이기
                        sharpened_face = increase_sharpness(face_image)

                        save_path = os.path.join(celebrity_output_folder, f'cropped_face_{i}.jpg')
                        cv2.imwrite(save_path, sharpened_face)
                        print(f'얼굴 이미지 저장: {save_path}')

            else:
                print(f'{image_path}: 얼굴을 찾을 수 없습니다.')

        except Exception as e:
            print(f'에러 발생: {e}')

# 창 닫기
cv2.destroyAllWindows()