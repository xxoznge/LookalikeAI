document.addEventListener('DOMContentLoaded', function () {
  const imageInput = document.getElementById('imageInput');
  const uploadButton = document.getElementById('uploadButton');

  imageInput.addEventListener('change', function () {
      previewImage();
  });

  uploadButton.addEventListener('click', function () {
      uploadImage();
  });
});

function previewImage() {
  const imageInput = document.getElementById('imageInput');
  const imagePreview = document.getElementById('imagePreview');
  const confirmationMessage = document.getElementById('confirmationMessage');

  const file = imageInput.files[0];

  if (file) {
      const reader = new FileReader();

      reader.onload = function (e) {
          const img = document.createElement('img');
          img.src = e.target.result;
          imagePreview.innerHTML = '';
          imagePreview.appendChild(img);
          confirmationMessage.innerText = '사진을 확인하셨나요?';
      };

      reader.readAsDataURL(file);
  } else {
      imagePreview.innerHTML = 'No image selected';
  }
}

function uploadImage() {
  const imageInput = document.getElementById('imageInput');
  const file = imageInput.files[0];

  if (file) {
      const formData = new FormData();
      formData.append('file', file);

      fetch('/', {
          method: 'POST',
          body: formData
      })
      .then(response => response.json())
      .then(data => {
          console.log(data);
          alert('이미지 업로드 성공!');

          // 업로드 성공 시 결과 페이지로 이동
          window.location.href = '/result.html'; // '/result.html'로 변경
      })
      .catch(error => {
          // 오류 처리
          console.error('이미지 업로드 실패:', error);
      });
  }
}
