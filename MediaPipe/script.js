import "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js";
import "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js";
import "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js";

const video = document.getElementById("inputVideo");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");

const faceMesh = new FaceMesh({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
});

faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
});

faceMesh.onResults((results) => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  if (results.multiFaceLandmarks.length > 0) {
    const points = results.multiFaceLandmarks[0];
    drawPoint(points[33], "red");
    drawPoint(points[133], "red");
  }
});

function drawPoint({ x, y }, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x * canvas.width, y * canvas.height, 4, 0, 2 * Math.PI);
  ctx.fill();
}

const camera = new Camera(video, {
  onFrame: async () => {
    await faceMesh.send({ image: video });
  },
  width: 640,
  height: 480,
});
camera.start();
