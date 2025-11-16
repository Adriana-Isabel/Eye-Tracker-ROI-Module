import "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js";
import "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js";
import "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js";

const video = document.getElementById("inputVideo");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");

// Setup FaceMesh
const faceMesh = new FaceMesh({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
});

faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true, // includes iris landmarks
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

faceMesh.onResults((results) => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // draw camera feed
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  if (!results.multiFaceLandmarks[0]) return;

  const pts = results.multiFaceLandmarks[0];

  // Draw IRIS landmarks
  drawPoint(pts[468], "red"); // left iris center
  drawPoint(pts[473], "blue"); // right iris center
});

function drawPoint({ x, y }, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x * canvas.width, y * canvas.height, 4, 0, Math.PI * 2);
  ctx.fill();
}

// Setup camera
const camera = new Camera(video, {
  onFrame: async () => {
    await faceMesh.send({ image: video });
  },
  width: 640,
  height: 480,
});
camera.start();
