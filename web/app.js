/**
 * Walkability AI — Client-Side ONNX Runtime Web Execution
 * 4-Class Semantic Segmentation (BiSeNetV2)
 */

// Model & Canvas configurations
const MODEL_PATH = './model.onnx';
const INPUT_WIDTH = 1024;
const INPUT_HEIGHT = 512;
const WALKING_ZONE_Y_START = Math.floor(INPUT_HEIGHT * 0.60); // Bottom 40%

// Palette: [R, G, B]
const PALETTE = [
  [0, 200, 0],     // 0: Walkable (Green)
  [128, 128, 128], // 1: Non-walkable (Gray)
  [220, 0, 0],     // 2: Dynamic Obstacle (Red)
  [255, 165, 0],   // 3: Structural Obstacle (Orange)
];

// App State
let session = null;
let isModelReady = false;
let isCameraActive = false;
let currentMode = 'image'; // 'image' | 'camera'
let voiceEnabled = true;
let voiceLanguage = 'en';
let lastSpokenTime = 0;
let lastAlertState = null;
let currentAlpha = 0.65;
let animationFrameId = null;

// DOM Elements
const baseCanvas = document.getElementById('baseCanvas');
const overlayCanvas = document.getElementById('overlayCanvas');
const baseCtx = baseCanvas.getContext('2d');
const overlayCtx = overlayCanvas.getContext('2d');
const webcam = document.getElementById('webcam');
const fileInput = document.getElementById('fileInput');
const placeholder = document.getElementById('placeholder');
const spinner = document.getElementById('spinner');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const opacitySlider = document.getElementById('opacitySlider');
const opacityVal = document.getElementById('opacityVal');

// Metrics DOM
const metricLatency = document.getElementById('metricLatency');
const metricFps = document.getElementById('metricFps');
const pctWalkable = document.getElementById('pctWalkable');
const pctNonWalkable = document.getElementById('pctNonWalkable');
const pctDynamic = document.getElementById('pctDynamic');
const pctStructural = document.getElementById('pctStructural');
const barWalkable = document.getElementById('barWalkable');
const barNonWalkable = document.getElementById('barNonWalkable');
const barDynamic = document.getElementById('barDynamic');
const barStructural = document.getElementById('barStructural');

// Safety Alert DOM
const safetyAlertBanner = document.getElementById('safetyAlertBanner');
const safetyAlertIcon = document.getElementById('safetyAlertIcon');
const safetyAlertTitle = document.getElementById('safetyAlertTitle');
const safetyAlertDesc = document.getElementById('safetyAlertDesc');
const btnVoiceToggle = document.getElementById('btnVoiceToggle');
const voiceLang = document.getElementById('voiceLang');

// Set Canvas dimensions
baseCanvas.width = INPUT_WIDTH;
baseCanvas.height = INPUT_HEIGHT;
overlayCanvas.width = INPUT_WIDTH;
overlayCanvas.height = INPUT_HEIGHT;
overlayCanvas.style.opacity = currentAlpha;

// Offscreen canvas for pre-processing
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = INPUT_WIDTH;
offscreenCanvas.height = INPUT_HEIGHT;
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

// --- 1. Model Loading ---
async function initModel() {
  try {
    statusText.innerText = 'Loading Model (24 MB)...';
    
    // Configure ONNX Runtime Web
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    ort.env.wasm.simd = true;

    // Create session (tries WebGL for GPU acceleration, falls back to WASM)
    let providerUsed = 'webgl';
    try {
      session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ['webgl']
      });
    } catch (gpuErr) {
      console.warn('WebGL provider unavailable or failed, falling back to WASM:', gpuErr);
      providerUsed = 'wasm';
      session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ['wasm']
      });
    }

    isModelReady = true;
    statusBadge.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/30';
    statusBadge.innerHTML = `
      <span class="relative flex h-2 w-2">
        <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
      </span>
      <span>Model Ready (${providerUsed.toUpperCase()})</span>
    `;

    console.log(`✅ BiSeNetV2 ONNX model loaded successfully (${providerUsed}).`);
    // Load first sample automatically once model is ready
    loadSample(1);
  } catch (err) {
    console.error('Failed to load ONNX model:', err);
    statusBadge.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-red-500/10 text-red-300 border border-red-500/30';
    statusText.innerText = 'Model Load Failed';
  }
}

// --- 2. Image Preprocessing (ImageNet Normalization) ---
function preprocess(imageSource) {
  offscreenCtx.drawImage(imageSource, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  const imgData = offscreenCtx.getImageData(0, 0, INPUT_WIDTH, INPUT_HEIGHT).data;

  // Planar Float32Array: shape [1, 3, 512, 1024]
  const floatArr = new Float32Array(3 * INPUT_HEIGHT * INPUT_WIDTH);
  const planeSize = INPUT_HEIGHT * INPUT_WIDTH;

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < planeSize; i++) {
    const r = imgData[i * 4] / 255.0;
    const g = imgData[i * 4 + 1] / 255.0;
    const b = imgData[i * 4 + 2] / 255.0;

    // Planar format: RRR... GGG... BBB...
    floatArr[i] = (r - mean[0]) / std[0];
    floatArr[planeSize + i] = (g - mean[1]) / std[1];
    floatArr[2 * planeSize + i] = (b - mean[2]) / std[2];
  }

  return new ort.Tensor('float32', floatArr, [1, 3, INPUT_HEIGHT, INPUT_WIDTH]);
}

// --- 3. Inference & Postprocessing ---
async function runInference(imageSource) {
  if (!isModelReady || !session) return;

  const startTime = performance.now();

  try {
    const inputTensor = preprocess(imageSource);
    const results = await session.run({ input: inputTensor });
    const outputData = results.output.data; // [1, 4, 512, 1024]

    const latency = Math.round(performance.now() - startTime);
    metricLatency.innerText = `${latency} ms`;
    metricFps.innerText = `${(1000 / Math.max(1, latency)).toFixed(1)} FPS`;

    renderSegmentation(outputData);
  } catch (err) {
    console.error('Inference error:', err);
  }
}

// --- 4. Render Colored Segmentation Mask ---
function renderSegmentation(outputData) {
  const planeSize = INPUT_HEIGHT * INPUT_WIDTH;
  const overlayData = overlayCtx.createImageData(INPUT_WIDTH, INPUT_HEIGHT);
  const data = overlayData.data;

  const alphaInt = 255; // Full opacity in buffer; overlayCanvas.style.opacity controls visual blending
  const counts = [0, 0, 0, 0];
  let walkingZoneObstaclePixels = 0;
  let totalWalkingZonePixels = 0;

  for (let i = 0; i < planeSize; i++) {
    // 4 channel logits for pixel i
    const l0 = outputData[i];
    const l1 = outputData[planeSize + i];
    const l2 = outputData[2 * planeSize + i];
    const l3 = outputData[3 * planeSize + i];

    // Argmax
    let maxLogit = l0;
    let cls = 0;
    if (l1 > maxLogit) { maxLogit = l1; cls = 1; }
    if (l2 > maxLogit) { maxLogit = l2; cls = 2; }
    if (l3 > maxLogit) { maxLogit = l3; cls = 3; }

    counts[cls]++;

    // Check if pixel falls in the bottom 40% walking zone
    const y = Math.floor(i / INPUT_WIDTH);
    if (y >= WALKING_ZONE_Y_START) {
      totalWalkingZonePixels++;
      if (cls === 2 || cls === 3) {
        walkingZoneObstaclePixels++;
      }
    }

    const rgb = PALETTE[cls];
    const idx = i * 4;
    data[idx] = rgb[0];
    data[idx + 1] = rgb[1];
    data[idx + 2] = rgb[2];
    data[idx + 3] = alphaInt;
  }

  overlayCtx.putImageData(overlayData, 0, 0);

  // Update telemetry percentages
  const totalPx = planeSize;
  const p0 = ((counts[0] / totalPx) * 100).toFixed(1);
  const p1 = ((counts[1] / totalPx) * 100).toFixed(1);
  const p2 = ((counts[2] / totalPx) * 100).toFixed(1);
  const p3 = ((counts[3] / totalPx) * 100).toFixed(1);

  pctWalkable.innerText = `${p0}%`;
  pctNonWalkable.innerText = `${p1}%`;
  pctDynamic.innerText = `${p2}%`;
  pctStructural.innerText = `${p3}%`;

  barWalkable.style.width = `${p0}%`;
  barNonWalkable.style.width = `${p1}%`;
  barDynamic.style.width = `${p2}%`;
  barStructural.style.width = `${p3}%`;

  // Evaluate Walking Zone Safety
  const obstacleRatio = walkingZoneObstaclePixels / Math.max(1, totalWalkingZonePixels);
  evaluateSafetyAlert(obstacleRatio);
}

// --- 5. Safety Alert & TTS Voice Alerts ---
function evaluateSafetyAlert(obstacleRatio) {
  // Threshold: if > 1.8% of walking zone is an obstacle
  const isHazard = obstacleRatio > 0.018;

  if (isHazard) {
    safetyAlertBanner.className = 'mt-2 p-3 rounded-xl border flex items-center gap-3 transition-colors bg-red-500/15 border-red-500/40 text-red-300 animate-pulse';
    safetyAlertIcon.innerText = '⚠️';
    safetyAlertTitle.innerText = 'Hazard Detected!';
    safetyAlertDesc.innerText = `Obstacle detected directly in the walking zone (${(obstacleRatio * 100).toFixed(1)}% area).`;
    triggerVoiceAlert('warning');
  } else {
    safetyAlertBanner.className = 'mt-2 p-3 rounded-xl border flex items-center gap-3 transition-colors bg-emerald-500/10 border-emerald-500/30 text-emerald-300';
    safetyAlertIcon.innerText = '🟢';
    safetyAlertTitle.innerText = 'Path Clear';
    safetyAlertDesc.innerText = 'Immediate walking path is safe to navigate.';
    triggerVoiceAlert('clear');
  }
}

function triggerVoiceAlert(type) {
  if (!voiceEnabled || !window.speechSynthesis) return;

  const now = Date.now();
  // Don't repeat speech within 3.5 seconds unless state changed from clear -> warning
  if (now - lastSpokenTime < 3500 && lastAlertState === type) return;

  lastSpokenTime = now;
  lastAlertState = type;

  window.speechSynthesis.cancel(); // Clear pending utterances

  let message = '';
  if (voiceLanguage === 'hi') {
    message = type === 'warning' ? 'सावधान! आगे बाधा है।' : 'रास्ता साफ़ है।';
  } else {
    message = type === 'warning' ? 'Warning! Obstacle detected ahead.' : 'Path clear.';
  }

  const utterance = new SpeechSynthesisUtterance(message);
  utterance.lang = voiceLanguage === 'hi' ? 'hi-IN' : 'en-US';
  utterance.rate = 1.05;
  window.speechSynthesis.speak(utterance);
}

// --- 6. User Actions: Samples & Uploads ---
function loadSample(num) {
  placeholder.classList.add('hidden');
  spinner.classList.remove('hidden');

  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    baseCtx.drawImage(img, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);
    runInference(img).finally(() => {
      spinner.classList.add('hidden');
    });
  };
  img.src = `samples/sample${num}.jpg`;
}

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  placeholder.classList.add('hidden');
  spinner.classList.remove('hidden');

  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      baseCtx.drawImage(img, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);
      runInference(img).finally(() => {
        spinner.classList.add('hidden');
      });
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
});

// --- 7. Live Camera Mode ---
async function startCamera() {
  try {
    placeholder.classList.add('hidden');
    spinner.classList.remove('hidden');

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
      audio: false
    });

    webcam.srcObject = stream;
    await webcam.play();

    isCameraActive = true;
    spinner.classList.add('hidden');
    cameraLoop();
  } catch (err) {
    alert('Unable to access camera: ' + err.message);
    setMode('image');
  }
}

function stopCamera() {
  isCameraActive = false;
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  if (webcam.srcObject) {
    webcam.srcObject.getTracks().forEach(track => track.stop());
    webcam.srcObject = null;
  }
}

async function cameraLoop() {
  if (!isCameraActive) return;

  baseCtx.drawImage(webcam, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  await runInference(webcam);

  animationFrameId = requestAnimationFrame(cameraLoop);
}

// Mode Switching Tabs
const tabImage = document.getElementById('tabImage');
const tabCamera = document.getElementById('tabCamera');

function setMode(mode) {
  currentMode = mode;
  if (mode === 'camera') {
    tabCamera.className = 'px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all bg-emerald-500 text-slate-950 shadow-md';
    tabImage.className = 'px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold text-slate-400 hover:text-white transition-all';
    startCamera();
  } else {
    tabImage.className = 'px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all bg-emerald-500 text-slate-950 shadow-md';
    tabCamera.className = 'px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold text-slate-400 hover:text-white transition-all';
    stopCamera();
    loadSample(1);
  }
}

tabImage.addEventListener('click', () => setMode('image'));
tabCamera.addEventListener('click', () => setMode('camera'));

// Opacity Slider
opacitySlider.addEventListener('input', (e) => {
  currentAlpha = parseFloat(e.target.value) / 100.0;
  opacityVal.innerText = `${e.target.value}%`;
  overlayCanvas.style.opacity = currentAlpha;
});

// Voice Settings
btnVoiceToggle.addEventListener('click', () => {
  voiceEnabled = !voiceEnabled;
  if (voiceEnabled) {
    btnVoiceToggle.innerText = '🔊 Enabled';
    btnVoiceToggle.className = 'px-3 py-1 rounded-lg text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 transition';
  } else {
    btnVoiceToggle.innerText = '🔇 Muted';
    btnVoiceToggle.className = 'px-3 py-1 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400 border border-slate-700 transition';
    if (window.speechSynthesis) window.speechSynthesis.cancel();
  }
});

voiceLang.addEventListener('change', (e) => {
  voiceLanguage = e.target.value;
});

function testSpeech() {
  lastSpokenTime = 0;
  lastAlertState = null;
  triggerVoiceAlert('warning');
}

// Start Initialization
initModel();
