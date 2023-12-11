document.addEventListener('DOMContentLoaded', function () {
    const userImage = document.getElementById('userImage');
    const resultImage = document.getElementById('resultImage');

    // Get the user-uploaded image from localStorage
    const userImageSrc = localStorage.getItem('userImage');
    if (userImageSrc) {
        userImage.src = userImageSrc;
    } else {
        userImage.alt = 'No User Image';
    }

    // 서버에서 전달한 결과 객체를 사용하여 표시
    const result = JSON.parse('{{ result|tojson|safe }}');
    resultImage.src = `/result_image/${result.most_similar_celebrity}.jpg`;
});
