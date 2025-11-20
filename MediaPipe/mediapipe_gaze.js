import "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js";
import "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js";
import "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js";

/* -----------------------
   DOM refs
   ----------------------- */
const video = document.getElementById('inputVideo');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d');
const gazeDot = document.getElementById('gazeDot');
const calibrateBtn = document.getElementById('calibrateBtn');
const enableBtn = document.getElementById('enableBtn');
const playBtn = document.getElementById('playBtn');
const stopBtn = document.getElementById('stopBtn');
const downloadBtn = document.getElementById('downloadBtn');
const arena = document.getElementById('arena');
const statusLine = document.getElementById('statusLine');
const vgVideo = document.getElementById('vgVideo');
const fixationIndicator = document.getElementById('fixationIndicator');
const dbgRawEl = document.getElementById('dbgRaw');
const dbgSmEl = document.getElementById('dbgSm');
const dbgConfEl = document.getElementById('dbgConf');
const dbgInsideEl = document.getElementById('dbgInside');
const dbgSourceSel = document.getElementById('dbgSource');
const dbgConfThreshInput = document.getElementById('dbgConfThresh');
const gazeSmoothInput = document.getElementById('gazeSmooth');
const gazeAlphaValueEl = document.getElementById('gazeAlphaValue');
const showGazeDotCheckbox = document.getElementById('showGazeDot');
const calibrateOverlay = document.getElementById('calibrateOverlay');
const trialCountInput = document.getElementById('trialCount');
const fixDurInput = document.getElementById('fixDur');

/* -----------------------
   State
   ----------------------- */
let camera = null;
let currentIris = null; // {x,y} normalized in FaceMesh image coords
let mapping = null;     // function(norm)->page coords
let videoRect = null;   // bounding rect of the preview video (used for calibration & fallback)
let videoMirrored = false; // whether the preview video is visually flipped (scaleX(-1))
let csvUrl = null;

/* gaze state */
let gazeX = NaN, gazeY = NaN;
let gazeAlpha = parseFloat(gazeSmoothInput?.value || 0.18);
let trials = [], currentIndex = 0, records = [], running = false, gazeLoop = null;
let fixationTimer = null;

/* -----------------------
   FaceMesh
   ----------------------- */
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
  // draw frame to hidden canvas (optional)
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (results.image) ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  if (!results.multiFaceLandmarks || !results.multiFaceLandmarks[0]) {
    currentIris = null;
    if (dbgConfEl) dbgConfEl.textContent = '-';
    return;
  }

  const pts = results.multiFaceLandmarks[0];

  // iris center landmarks (MediaPipe refined)
  const left = pts[468];
  const right = pts[473];

  currentIris = {
    x: (left.x + right.x) / 2,
    y: (left.y + right.y) / 2
  };

  // debug conf (we currently use presence as 1.0)
  if (dbgConfEl) dbgConfEl.textContent = '1.00';

  // draw small debug dots (on canvas)
  const drawDot = (pt, color) => {
    ctx.beginPath();
    ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  };
  drawDot(left, "#00ffea");
  drawDot(right, "#ff00aa");

  // optional mesh points (comment out if noisy)
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    ctx.fillStyle = "rgba(255,255,255,0.06)";
    ctx.fillRect(p.x * canvas.width, p.y * canvas.height, 1.2, 1.2);
  }
});

/* -----------------------
   Math helpers
   ----------------------- */
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
  // samples: [{nx,ny, tx,ty}, ...]  (tx/ty page coords)
  const N = samples.length;
  if(N < 3) return null;
  const MTM = [0,0,0, 0,0,0, 0,0,0];
  const Mtbx = [0,0,0];
  const Mtby = [0,0,0];
  for(const s of samples){
    const x = s.nx, y = s.ny, w = 1;
    const row = [x,y,w];
    for(let i=0;i<3;i++) for(let j=0;j<3;j++) MTM[i*3+j] += row[i]*row[j];
    Mtbx[0] += row[0]*s.tx; Mtbx[1] += row[1]*s.tx; Mtbx[2] += row[2]*s.tx;
    Mtby[0] += row[0]*s.ty; Mtby[1] += row[1]*s.ty; Mtby[2] += row[2]*s.ty;
  }
  const inv = invert3x3(MTM);
  if(!inv) return null;
  function mulInvVec(invMat, vec){
    return [
      invMat[0]*vec[0] + invMat[1]*vec[1] + invMat[2]*vec[2],
      invMat[3]*vec[0] + invMat[4]*vec[1] + invMat[5]*vec[2],
      invMat[6]*vec[0] + invMat[7]*vec[1] + invMat[8]*vec[2]
    ];
  }
  const ax = mulInvVec(inv, Mtbx);
  const ay = mulInvVec(inv, Mtby);
  return function(norm){
    const X = ax[0]*norm.x + ax[1]*norm.y + ax[2];
    const Y = ay[0]*norm.x + ay[1]*norm.y + ay[2];
    return { x: X, y: Y };
  };
}

/* -----------------------
   Calibration helpers
   ----------------------- */
function updateVideoRect(){
  // prefer preview element vgVideo; fallback to input video
  videoRect = (vgVideo && vgVideo.getBoundingClientRect && vgVideo.getBoundingClientRect().width > 0)
    ? vgVideo.getBoundingClientRect()
    : (video.getBoundingClientRect ? video.getBoundingClientRect() : {left:0,top:0,width:window.innerWidth,height:window.innerHeight});
  // detect if the preview video is visually mirrored (CSS transform scaleX(-1))
  try{
    const cs = vgVideo ? window.getComputedStyle(vgVideo) : null;
    const t = cs ? (cs.transform || cs.getPropertyValue('transform') || '') : '';
    // common forms: 'none', 'matrix(-1, 0, 0, 1, 0, 0)' or 'scaleX(-1)'
    videoMirrored = !!(t && (t.indexOf('-1') !== -1 || t.indexOf('scaleX(-1)') !== -1));
  }catch(e){ videoMirrored = false; }
}

function createCalDots(){
  updateVideoRect();
  calibrateOverlay.innerHTML = '';
  calibrateOverlay.style.display = 'block';  // show overlay
  calibrateOverlay.style.background = 'rgba(0,0,0,0.8)';

  // overlay is fixed, so dots inside are absolute relative to overlay
  calibrateOverlay.style.position = 'fixed';
  calibrateOverlay.style.inset = '0';
  calibrateOverlay.style.pointerEvents = 'none';

  // 3x3 grid, positions relative to videoRect
  const positions = [
    [0.1,0.1],[0.5,0.1],[0.9,0.1],
    [0.1,0.5],[0.5,0.5],[0.9,0.5],
    [0.1,0.9],[0.5,0.9],[0.9,0.9]
  ];

  for(const [nx,ny] of positions){
    const d = document.createElement('div');
    d.className = 'cal-dot';
    d.style.position = 'absolute';  // relative to overlay
    const cx = nx * videoRect.width;
    const cy = ny * videoRect.height;
    d.style.left = `${cx}px`;
    d.style.top = `${cy}px`;
    calibrateOverlay.appendChild(d);
    // center by subtracting half width/height
    const rect = d.getBoundingClientRect();
    d.style.left = `${cx - rect.width/2}px`;
    d.style.top  = `${cy - rect.height/2}px`;
  }
}


async function runCalibration(){
  if(!camera) { alert('Enable camera first'); return; }
  statusLine.textContent = 'Calibration: follow dots';
  createCalDots();
  const dots = Array.from(calibrateOverlay.children);
  const samples = [];
  // ensure cleanup even if something goes wrong while sampling
  try{
    for(let i=0;i<dots.length;i++){
      const dot = dots[i];
      try{
        dot.style.transform='scale(1.25)';
        const rect = dot.getBoundingClientRect();
        const centerX = rect.left + rect.width/2;
        const centerY = rect.top + rect.height/2;
        const collected = [];
        const t0 = performance.now();
        while(performance.now() - t0 < 700){
          if(currentIris) collected.push({nx: currentIris.x, ny: currentIris.y});
          await new Promise(r=>setTimeout(r,40));
        }
        if(collected.length === 0){
          console.warn('No iris samples for dot', i);
        } else {
          // simple median-based outlier rejection per axis
          const xs = collected.map(s=>s.nx).sort((a,b)=>a-b);
          const ys = collected.map(s=>s.ny).sort((a,b)=>a-b);
          const medianX = xs[Math.floor(xs.length/2)];
          const medianY = ys[Math.floor(ys.length/2)];
          // keep near-median samples
          const filt = collected.filter(s => Math.abs(s.nx - medianX) < 0.12 && Math.abs(s.ny - medianY) < 0.12);
          const use = filt.length >= 3 ? filt : collected;
          const avgX = use.reduce((s,v)=>s+v.nx,0)/use.length;
          const avgY = use.reduce((s,v)=>s+v.ny,0)/use.length;
          // if preview is mirrored horizontally, invert the normalized x when computing page X
          const nxForPage = videoMirrored ? (1 - avgX) : avgX;
          const tx = Math.round(videoRect.left + nxForPage * videoRect.width);
          const ty = Math.round(videoRect.top  + avgY * videoRect.height);
          samples.push({nx: avgX, ny: avgY, tx: tx, ty: ty});
        }
      }finally{
        // always reset visual state for this dot
        try{ dots[i].style.transform = 'scale(1)'; }catch(e){}
      }
      await new Promise(r=>setTimeout(r,200));
    }
  }finally{
    // final cleanup: hide and clear overlay; ensure any stray dots are removed
    setTimeout(()=>{
      try{ calibrateOverlay.style.display = 'none'; }catch(e){}
      try{ calibrateOverlay.innerHTML = ''; }catch(e){}
    }, 60);
  }
  if(samples.length < 3){
    alert('Calibration failed — not enough samples');
    statusLine.textContent='Status: calibration failed';
    return;
  }
  // Try identity and flipped-x variants and pick best by RMSE
  console.log('Calibration samples:', samples);
  const tryVariants = [];
  // variant A: identity (use samples as stored)
  tryVariants.push({ name: 'identity', samples: samples });
  // variant B: flip normalized x when fitting (treat nx' = 1 - nx)
  tryVariants.push({ name: 'flipX', samples: samples.map(s => ({ nx: 1 - s.nx, ny: s.ny, tx: s.tx, ty: s.ty })) });

  let best = null;
  for(const v of tryVariants){
    const m = computeAffine(v.samples);
    if(!m) { console.warn('computeAffine returned null for variant', v.name); continue; }
    // compute RMSE over samples
    let se = 0, n=0;
    for(const s of v.samples){ const p = m({ x: s.nx, y: s.ny }); if(p && isFinite(p.x) && isFinite(p.y)){ const dx = p.x - s.tx, dy = p.y - s.ty; se += dx*dx + dy*dy; n++; } }
    const rmse = n>0 ? Math.sqrt(se/n) : Infinity;
    console.log(`variant ${v.name} rmse=${rmse}`);
    if(!best || rmse < best.rmse){ best = { name: v.name, mapping: m, rmse }; }
  }
  if(!best || !best.mapping){
    alert('Could not compute mapping');
    statusLine.textContent='Status: calibration failed';
    return;
  }
  mapping = best.mapping;
  console.log('Chosen mapping variant:', best.name, 'rmse=', best.rmse);
  statusLine.textContent=`Status: calibrated (${best.name}, rmse ${Math.round(best.rmse)})`;
  calibrateBtn.disabled = false;
}

/* -----------------------
   Camera
   ----------------------- */
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
  // keep gaze dot hidden until we have a valid mapped gaze
  try{ gazeDot.style.display = 'none'; }catch(e){}
  try{
    if(video && video.srcObject && vgVideo){ vgVideo.srcObject = video.srcObject; vgVideo.play().catch(()=>{}); }
  }catch(e){}
  updateVideoRect();
}

enableBtn.onclick = async ()=>{ try{ await enableCamera(); }catch(e){ alert('Camera permission required.'); console.error(e); } };
calibrateBtn.onclick = async ()=>{ await runCalibration(); };

/* -----------------------
   Trials / fixation test
   ----------------------- */
function randBetween(min,max){ return Math.round(min + Math.random()*(max-min)); }
function buildTrials(n){
  const rect = arena.getBoundingClientRect();
  const arr=[]; const margin=60;
  for(let i=0;i<n;i++){
    arr.push({id:i+1,x:randBetween(margin,rect.width-margin-90),y:randBetween(margin,rect.height-margin-90)});
  }
  return arr;
}

function showTarget(t){
  arena.innerHTML='';
  const el = document.createElement('div');
  el.className='target';
  el.style.position='absolute';
  el.style.left = t.x + 'px';
  el.style.top = t.y + 'px';
  el.textContent = t.id;
  arena.appendChild(el);
  return el;
}

function startTest(){
  if(!camera){ alert('Enable camera first'); return; }
  records = []; trials = buildTrials(parseInt(trialCountInput?.value || 8,10) || 8);
  currentIndex = 0; running = true; playBtn.disabled = true; stopBtn.disabled = false; enableBtn.disabled = true;
  statusLine.textContent = 'Running test...';
  showNextTrial();
  // keep using the single animate loop for gaze updates; but we still use interval for logic checks
  gazeLoop = setInterval(()=> {
    if(!isFinite(gazeX) && !currentIris) return;
    const tgt = arena.querySelector('.target'); if(!tgt) return;

    const dbgConfThresh = parseFloat(dbgConfThreshInput?.value || '0.05');
    const dbgSource = dbgSourceSel?.value || 'smoothed';

    // pick source for test: 'raw' (not used) or 'smoothed' (gazeX/gazeY)
    const gx = gazeX, gy = gazeY;
    const r = tgt.getBoundingClientRect(); const TOL = 18;
    const inside = (gx >= (r.left - TOL) && gx <= (r.right + TOL) && gy >= (r.top - TOL) && gy <= (r.bottom + TOL));

    if(inside){
      if(!tgt.classList.contains('highlight')) tgt.classList.add('highlight');
      if(fixationIndicator){ fixationIndicator.textContent = 'Fixation: Yes'; fixationIndicator.style.background = 'rgba(16,185,129,0.12)'; fixationIndicator.style.color = '#9ef6d7'; }
      if(!fixationTimer){ fixationTimer = { startPerf: performance.now(), startWall: Date.now() }; }
      else {
        const dur = performance.now() - fixationTimer.startPerf;
        if(dur >= (parseInt(fixDurInput?.value,10)||350)){
          const endWall = Date.now();
          const fixationDur = Math.round(performance.now() - fixationTimer.startPerf);
          records.push({ trial: trials[currentIndex].id, duration_ms: fixationDur, start_ISO: new Date(fixationTimer.startWall).toISOString(), end_ISO: new Date(endWall).toISOString() });
          fixationTimer = null; tgt.remove();
          if(fixationIndicator){ fixationIndicator.textContent = 'Fixation: No'; fixationIndicator.style.background = 'rgba(255,255,255,0.04)'; fixationIndicator.style.color = '#ddd'; }
          setTimeout(()=>{ currentIndex++; currentIndex < trials.length ? showNextTrial() : finishTest(); }, 500);
        }
      }
    } else {
      fixationTimer = null;
      if(fixationIndicator){ fixationIndicator.textContent = 'Fixation: No'; fixationIndicator.style.background = 'rgba(255,255,255,0.04)'; fixationIndicator.style.color = '#ddd'; }
      const t = arena.querySelector('.target'); if(t) t.classList.remove('highlight');
    }

    // update debug UI
    try{
      const raw = currentIris ? { x: Math.round((currentIris.x||0)*(videoRect?.width||window.innerWidth) + (videoRect?.left||0)), y: Math.round((currentIris.y||0)*(videoRect?.height||window.innerHeight) + (videoRect?.top||0)) } : null;
      const sm = (isFinite(gazeX) && isFinite(gazeY)) ? { x: Math.round(gazeX), y: Math.round(gazeY) } : null;
      if(dbgRawEl) dbgRawEl.textContent = raw ? `${raw.x},${raw.y}` : '-';
      if(dbgSmEl) dbgSmEl.textContent = sm ? `${sm.x},${sm.y}` : '-';
      if(dbgInsideEl) dbgInsideEl.textContent = inside ? 'YES' : 'no';
    }catch(e){}
  }, 50);
}

function showNextTrial(){ const t = trials[currentIndex]; if(!t){ finishTest(); return; } showTarget(t); statusLine.textContent = `Trial ${currentIndex+1}/${trials.length}`; }

function finishTest(){
  running = false;
  clearInterval(gazeLoop);
  gazeLoop = null;
  playBtn.disabled = false;
  stopBtn.disabled = true;
  enableBtn.disabled = false;
  statusLine.textContent = `Test finished — ${records.length} fixations recorded`;
  arena.innerHTML='';
  if(records.length){
    const header = 'trial,start_ISO,end_ISO,duration_ms\n';
    const rows = records.map(r=>`${r.trial},${r.start_ISO},${r.end_ISO},${r.duration_ms}`).join('\n');
    const csv = header + rows;
    const blob = new Blob([csv], {type:'text/csv'});
    if(csvUrl) URL.revokeObjectURL(csvUrl);
    csvUrl = URL.createObjectURL(blob);
    if(downloadBtn){
      downloadBtn.style.display = 'inline-block';
      downloadBtn.onclick = ()=>{ const a = document.createElement('a'); a.href = csvUrl; a.download = `fixations_${Date.now()}.csv`; a.click(); a.remove(); };
    }
  }
}

playBtn.onclick = ()=> startTest();
stopBtn.onclick = ()=> { if(running) finishTest(); };

/* download button */
if(downloadBtn){
  downloadBtn.addEventListener('click', ()=>{
    if(csvUrl){ const a = document.createElement('a'); a.href = csvUrl; a.download = `fixations_${Date.now()}.csv`; a.click(); a.remove(); }
  });
}

/* smoothing UI */
try{
  if(gazeSmoothInput){
    gazeSmoothInput.addEventListener('input', (e)=>{
      const v = parseFloat(e.target.value);
      if(!isNaN(v)){ gazeAlpha = v; if(gazeAlphaValueEl) gazeAlphaValueEl.textContent = v.toFixed(2); }
    });
  }
}catch(e){}

/* -----------------------
   Single animation loop (stable)
   ----------------------- */
function animateGaze(){
  try{
    // ensure video rect is up-to-date
    updateVideoRect();

    if (currentIris){
      // compute mapped coordinates (use mapping if calibrated; fallback to videoRect mapping)
      let mapped = null;
      if (mapping) {
        mapped = mapping(currentIris);
      } else if (videoRect) {
        // account for mirrored preview when using direct videoRect fallback mapping
        const nx = currentIris.x;
        const nxForPage = videoMirrored ? (1 - nx) : nx;
        mapped = { x: videoRect.left + nxForPage * videoRect.width, y: videoRect.top + currentIris.y * videoRect.height };
      } else {
        // last resort: window mapping (less accurate)
        mapped = { x: currentIris.x * window.innerWidth, y: currentIris.y * window.innerHeight };
      }

      if (mapped && isFinite(mapped.x) && isFinite(mapped.y)){
        if(!isFinite(gazeX)){ gazeX = mapped.x; gazeY = mapped.y; }
        gazeX += (mapped.x - gazeX) * gazeAlpha;
        gazeY += (mapped.y - gazeY) * gazeAlpha;

        if (gazeDot && (showGazeDotCheckbox ? showGazeDotCheckbox.checked : true)){
          try{ gazeDot.style.left = Math.round(gazeX) + 'px'; gazeDot.style.top = Math.round(gazeY) + 'px'; gazeDot.style.display = 'block'; }catch(e){}
        }
      } else {
        // no valid mapped gaze — hide dot to avoid it sitting at 0,0
        if (gazeDot) try{ gazeDot.style.display = 'none'; }catch(e){}
      }
    }
  }catch(e){
    console.error('animateGaze error', e);
  }
  requestAnimationFrame(animateGaze);
}
animateGaze();
