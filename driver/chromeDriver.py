import requests
import zipfile
import io

# 선택한 크롬 드라이버 다운로드 URL
download_url = "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/119.0.6045.105/win64/chromedriver-win64.zip"

# 다운로드 요청 보내기
response = requests.get(download_url)

if response.status_code == 200:
    # 다운로드한 파일을 메모리에 로드
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        # 압축 해제할 경로 지정
        extraction_path = "크롬 드라이버를 저장할 디렉토리 경로"
        zip_file.extractall(extraction_path)
        print("크롬 드라이버 다운로드 및 압축 해제 완료.")
else:
    print("다운로드 실패. HTTP 상태 코드:", response.status_code)
