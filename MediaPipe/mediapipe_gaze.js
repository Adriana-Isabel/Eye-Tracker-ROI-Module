import "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js";
import "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js";
import "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js";

const video = document.getElementById('inputVideo');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d');
const gazeDot = document.getElementById('gazeDot');
const calibrateBtn = document.getElementById('calibrateBtn');
const enableBtn = document.getElementById('enableBtn');
const playBtn = document.getElementById('playBtn');
const stopBtn = document.getElementById('stopBtn');
const arena = document.getElementById('arena');
const statusLine = document.getElementById('statusLine');

let camera = null;
let currentIris = null; // {x,y} normalized
let mapping = null; // function to map normalized iris->page coords

// FaceMesh setup
const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

faceMesh.onResults((results) => {

  // clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // draw the video frame
  if (results.image) {
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
  }

  // no face → stop
  if (!results.multiFaceLandmarks || !results.multiFaceLandmarks[0]) {
    currentIris = null;
    return;
  }

  const pts = results.multiFaceLandmarks[0];

  // iris centers
  const left = pts[468];
  const right = pts[473];

  currentIris = {
    x: (left.x + right.x) / 2,
    y: (left.y + right.y) / 2
  };

  // ------- draw dots on eyes -------
  function drawDot(pt, color) {
    ctx.beginPath();
    ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  drawDot(left, "#00ffea");   // left iris
  drawDot(right, "#ff00aa");  // right iris

  // Optional: draw face mesh
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillRect(p.x * canvas.width, p.y * canvas.height, 1.5, 1.5);
  }
});


// small linear algebra helpers for 3x3 inversion
function invert3x3(m){
  const a=m;
  const det = a[0]* (a[4]*a[8]-a[5]*a[7]) - a[1]*(a[3]*a[8]-a[5]*a[6]) + a[2]*(a[3]*a[7]-a[4]*a[6]);
  if(Math.abs(det) < 1e-9) return null;
  const invDet = 1/det;
  const inv = [];
  inv[0] = (a[4]*a[8]-a[5]*a[7])*invDet;
  inv[1] = (a[2]*a[7]-a[1]*a[8])*invDet;
  inv[2] = (a[1]*a[5]-a[2]*a[4])*invDet;
  inv[3] = (a[5]*a[6]-a[3]*a[8])*invDet;
  inv[4] = (a[0]*a[8]-a[2]*a[6])*invDet;
  inv[5] = (a[2]*a[3]-a[0]*a[5])*invDet;
  inv[6] = (a[3]*a[7]-a[4]*a[6])*invDet;
  inv[7] = (a[1]*a[6]-a[0]*a[7])*invDet;
  inv[8] = (a[0]*a[4]-a[1]*a[3])*invDet;
  return inv;
}

function computeAffine(samples){
  // samples: [{nx,ny, tx,ty}, ...]
  const N = samples.length;
  if(N < 3) return null;
  // build M (Nx3) and bx, by
  const MTM = [0,0,0, 0,0,0, 0,0,0]; // 3x3
  const Mtbx = [0,0,0];
  const Mtby = [0,0,0];
  for(const s of samples){
    const x = s.nx, y = s.ny, w = 1;
    const row = [x,y,w];
    // accumulate MTM = M^T * M
    for(let i=0;i<3;i++) for(let j=0;j<3;j++) MTM[i*3+j] += row[i]*row[j];
    Mtbx[0] += row[0]*s.tx; Mtbx[1] += row[1]*s.tx; Mtbx[2] += row[2]*s.tx;
    Mtby[0] += row[0]*s.ty; Mtby[1] += row[1]*s.ty; Mtby[2] += row[2]*s.ty;
  }
  const inv = invert3x3(MTM);
  if(!inv) return null;
  // solution a = inv(MTM) * M^T * b  -> a is length 3
  function mulInvVec(invMat, vec){
    return [
      invMat[0]*vec[0] + invMat[1]*vec[1] + invMat[2]*vec[2],
      invMat[3]*vec[0] + invMat[4]*vec[1] + invMat[5]*vec[2],
      invMat[6]*vec[0] + invMat[7]*vec[1] + invMat[8]*vec[2]
    ];
  }
  const ax = mulInvVec(inv, Mtbx);
  const ay = mulInvVec(inv, Mtby);
  // mapping function
  return function(norm){
    const X = ax[0]*norm.x + ax[1]*norm.y + ax[2];
    const Y = ay[0]*norm.x + ay[1]*norm.y + ay[2];
    return { x: X, y: Y };
  };
}

// calibration utilities
function createCalDots(){
  const overlay = document.getElementById('calibrateOverlay');
  overlay.innerHTML = '';
  overlay.style.display = 'flex';
  const positions = [
    [0.1,0.1],[0.5,0.1],[0.9,0.1],
    [0.1,0.5],[0.5,0.5],[0.9,0.5],
    [0.1,0.9],[0.5,0.9],[0.9,0.9]
  ];
  for(const [nx,ny] of positions){
    const d = document.createElement('div');
    d.style.position='absolute'; d.style.width='36px'; d.style.height='36px'; d.style.borderRadius='50%';
    d.style.left = `${Math.round(window.innerWidth*nx - 18)}px`;
    d.style.top = `${Math.round(window.innerHeight*ny - 18)}px`;
    d.style.background='#60a5fa'; d.style.boxShadow='0 8px 20px rgba(0,0,0,0.6)';
    overlay.appendChild(d);
  }
}

async function runCalibration(){
  if(!camera) { alert('Enable camera first'); return; }
  statusLine.textContent = 'Calibration: follow dots';
  const overlay = document.getElementById('calibrateOverlay');
  createCalDots();
  const dots = Array.from(overlay.children);
  const samples = [];
  for(let i=0;i<dots.length;i++){
    const dot = dots[i];
    dot.style.transform='scale(1.3)';
    const centerX = parseFloat(dot.style.left) + 18; const centerY = parseFloat(dot.style.top) + 18;
    // collect iris samples for 700ms
    const collected = [];
    const t0 = performance.now();
    while(performance.now() - t0 < 700){
      if(currentIris) collected.push({nx: currentIris.x, ny: currentIris.y});
      await new Promise(r=>setTimeout(r,40));
    }
    // average
    if(collected.length === 0){
      // mark and continue (could repeat), but push an invalid sample
      console.warn('No iris samples for dot', i);
    } else {
      const avgX = collected.reduce((s,v)=>s+v.nx,0)/collected.length;
      const avgY = collected.reduce((s,v)=>s+v.ny,0)/collected.length;
      samples.push({nx: avgX, ny: avgY, tx: centerX, ty: centerY});
    }
    dot.style.transform='scale(1)';
    await new Promise(r=>setTimeout(r,200));
  }
  overlay.style.display='none'; overlay.innerHTML='';
  if(samples.length < 3){ alert('Calibration failed — not enough samples'); statusLine.textContent='Status: calibration failed'; return; }
  mapping = computeAffine(samples);
  if(!mapping){ alert('Could not compute mapping'); statusLine.textContent='Status: calibration failed'; return; }
  statusLine.textContent='Status: calibrated';
  calibrateBtn.disabled = false;
}

// Enable camera and start FaceMesh
async function enableCamera(){
  if(camera) return;
  statusLine.textContent = 'Status: starting camera...';
  camera = new Camera(video, {
    onFrame: async () => { await faceMesh.send({image: video}); },
    width: 640, height: 480
  });
  await camera.start();
  statusLine.textContent = 'Status: camera ready';
  calibrateBtn.disabled = false;
  gazeDot.style.display = 'block';
}

enableBtn.onclick = async ()=>{ try{ await enableCamera(); }catch(e){ alert('Camera permission required.'); console.error(e); } };
calibrateBtn.onclick = async ()=>{ await runCalibration(); };

// simple play test harness (build random targets, detect fixation)
function randBetween(min,max){ return Math.round(min + Math.random()*(max-min)); }
function buildTrials(n){ const rect = arena.getBoundingClientRect(); const arr=[]; const margin=60; for(let i=0;i<n;i++){ arr.push({id:i+1,x:randBetween(margin,rect.width-margin-90),y:randBetween(margin,rect.height-margin-90)}); } return arr; }

let trials = [], currentIndex = 0, records = [], running = false, gazeLoop = null, fixationTimer = null, lastPrediction = null;
let gazeX = NaN, gazeY = NaN, gazeAlpha = 0.18; // smoothing

function showTarget(t){ arena.innerHTML=''; const el = document.createElement('div'); el.className='target'; el.style.left = t.x + 'px'; el.style.top = t.y + 'px'; el.textContent = t.id; arena.appendChild(el); return el; }

function startTest(){
  if(!camera){ alert('Enable camera first'); return; }
  records = []; trials = buildTrials(parseInt(document.getElementById('trialCount').value,10)||8);
  currentIndex = 0; running = true; playBtn.disabled = true; stopBtn.disabled = false; enableBtn.disabled = true; statusLine.textContent = 'Running test...';
  showNextTrial();
  gazeLoop = setInterval(()=>{
    if(!currentIris && !isFinite(gazeX)) return;
    const tgt = arena.querySelector('.target'); if(!tgt) return;
    // map using mapping or fallback
    let mapped = null;
    if(mapping && currentIris) mapped = mapping(currentIris);
    else if(currentIris){ mapped = { x: currentIris.x * window.innerWidth, y: currentIris.y * window.innerHeight }; }
    if(!mapped) return;
    // smoothing
    if(!isFinite(gazeX)){ gazeX = mapped.x; gazeY = mapped.y; }
    gazeX += (mapped.x - gazeX) * gazeAlpha; gazeY += (mapped.y - gazeY) * gazeAlpha;
    gazeDot.style.left = Math.round(gazeX) + 'px'; gazeDot.style.top = Math.round(gazeY) + 'px'; gazeDot.style.display = 'block';

    const r = tgt.getBoundingClientRect(); const TOL = 18;
    const inside = (gazeX >= (r.left - TOL) && gazeX <= (r.right + TOL) && gazeY >= (r.top - TOL) && gazeY <= (r.bottom + TOL));
    if(inside){ if(!tgt.classList.contains('highlight')) tgt.classList.add('highlight'); if(!fixationTimer){ fixationTimer = { startPerf: performance.now(), startWall: Date.now() }; } else { const dur = performance.now() - fixationTimer.startPerf; if(dur >= (parseInt(document.getElementById('fixDur').value,10)||350)){ const endWall = Date.now(); const fixationDur = Math.round(performance.now() - fixationTimer.startPerf); records.push({ trial: trials[currentIndex].id, duration_ms: fixationDur, start_ISO: new Date(fixationTimer.startWall).toISOString(), end_ISO: new Date(endWall).toISOString() }); fixationTimer = null; tgt.remove(); setTimeout(()=>{ currentIndex++; currentIndex < trials.length ? showNextTrial() : finishTest(); }, 500); } } }
    else { fixationTimer = null; const t = arena.querySelector('.target'); if(t) t.classList.remove('highlight'); }
  }, 50);
}

function showNextTrial(){ const t = trials[currentIndex]; if(!t){ finishTest(); return; } showTarget(t); statusLine.textContent = `Trial ${currentIndex+1}/${trials.length}`; }

function finishTest(){ running = false; clearInterval(gazeLoop); gazeLoop = null; playBtn.disabled = false; stopBtn.disabled = true; enableBtn.disabled = false; statusLine.textContent = `Test finished — ${records.length} fixations recorded`; arena.innerHTML=''; if(records.length){ const header = 'trial,start_ISO,end_ISO,duration_ms\n'; const rows = records.map(r=>`${r.trial},${r.start_ISO},${r.end_ISO},${r.duration_ms}`).join('\n'); const csv = header + rows; const blob = new Blob([csv], {type:'text/csv'}); const url = URL.createObjectURL(blob); const dl = document.createElement('a'); dl.href = url; dl.download = `fixations_${Date.now()}.csv`; dl.click(); dl.remove(); } }

playBtn.onclick = ()=> startTest();
stopBtn.onclick = ()=> { if(running) finishTest(); };

// small animation loop to keep gazeDot smooth when currentIris updates outside interval
(function animate(){ try{ if(currentIris && mapping){ const mapped = mapping(currentIris); if(isFinite(mapped.x) && isFinite(mapped.y)){ if(!isFinite(gazeX)){ gazeX = mapped.x; gazeY = mapped.y; } gazeX += (mapped.x - gazeX) * gazeAlpha; gazeY += (mapped.y - gazeY) * gazeAlpha; gazeDot.style.left = Math.round(gazeX) + 'px'; gazeDot.style.top = Math.round(gazeY) + 'px'; gazeDot.style.display = 'block'; } } }catch(e){} requestAnimationFrame(animate); })();
