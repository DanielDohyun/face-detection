const imgUpload = document.getElementById('imgUpload');

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models2'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models2'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models2')
]).then(start)

async function start() {
    const container = document.createElement('div');
    container.style.position = 'relative';

    document.body.append(container);
    document.body.append('Loaded');

    const labeledFaceDescriptors = await loadLabeledImages()
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)

    imgUpload.addEventListener('change', async () => {
        const img = await faceapi.bufferToImage(imgUpload.files[0]);
        container.append(img);
        const canvas = faceapi.createCanvasFromMedia(img);
        container.append(canvas);

        const displaySize = { width: img.width, height: img.height };
        faceapi.matchDimensions(canvas, displaySize);

        const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();

        //resize all boxes for detection to be the correct size based on the size that i pass it.
        const resizedDetection = faceapi.resizeResults(detections, displaySize);

        const results = resizedDetection.map(d => faceMatcher.findBestMatch(d.descriptor))

        results.forEach((res, i) => {
            const box = resizedDetection[i].detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, {label: res.toString()});
            drawBox.draw(canvas);
        })
    })
}

function loadLabeledImages() {
    const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark']
    return Promise.all(
        labels.map(async label => {
            const descriptions = []
            for (let i = 1; i <= 2; i++) {
                const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`)
                
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()

                descriptions.push(detections.descriptor)
            }

            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}
