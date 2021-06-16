const imgUpload = document.getElementById('imgUpload');

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models2'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models2'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models2')
]).then(start)

function start() {
    document.body.append('Loaded');
    imgUpload.addEventListener('change', async () => {
        const img = await faceapi.bufferToImage(imgUpload.files[0]);
        document.body.append(img);
        const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();

        document.body.append(detections.length);
    })
}
