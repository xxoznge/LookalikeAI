// 파일이 선택되었을 때 미리보기를 표시하는 기능
function previewImage() {
    const fileInput = document.getElementById('file');
    const uploadedImageContainer = document.getElementById('uploadedImageContainer');
    const fileNameInput = document.getElementById('fileName');

    // 업로드된 파일이 이미지인 경우에만 처리
    if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();

        reader.onload = function (e) {
            const previewImage = document.createElement('img');
            previewImage.src = e.target.result;
            previewImage.style.maxWidth = '100%';
            uploadedImageContainer.innerHTML = ''; // 이미지 미리보기 갱신
            uploadedImageContainer.appendChild(previewImage);
        };

        reader.readAsDataURL(fileInput.files[0]);
        fileNameInput.value = fileInput.files[0].name; // 파일 이름 업데이트
    }
}

// 파일 업로드 함수
function uploadImage() {
    const fileInput = document.getElementById('file');
    const formData = new FormData();

    // 업로드할 파일을 FormData에 추가
    formData.append('file', fileInput.files[0]);

    // 서버로 파일 업로드 요청
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // 서버에서 받은 데이터로 이미지 업데이트
        const uploadedImageContainer = document.getElementById('uploadedImageContainer');
        const previewImage = document.createElement('img');
        previewImage.src = data.imageUrl;
        previewImage.style.maxWidth = '100%';
        uploadedImageContainer.innerHTML = ''; // 이미지 미리보기 갱신
        uploadedImageContainer.appendChild(previewImage);
    })
    .catch(error => console.error('Error:', error));
}
