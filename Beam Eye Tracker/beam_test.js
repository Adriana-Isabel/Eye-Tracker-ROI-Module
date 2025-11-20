const WebSocket = require('ws');

// Replace with Beam's WebSocket URL if different
const ws = new WebSocket('ws://localhost:5050');

ws.on('open', () => console.log('Connected to Beam'));
ws.on('message', (msg) => {
    const data = JSON.parse(msg);
    console.log(data); // x,y gaze points
});
