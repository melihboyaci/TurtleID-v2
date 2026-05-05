/* ═══════════════════════════════════════════════════════════
   TurtleID — app.js
   Vanilla JS — no external libraries
   ═══════════════════════════════════════════════════════════ */

"use strict";

/* ── Sabitler ────────────────────────────────────────────────── */
const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10 MB dosya sınırı
const PIPELINE_DELAY_MS = 800; // pipeline adım gecikmesi (ms)

/* ── Tab sistemi ──────────────────────────────────────────── */
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    document
      .querySelectorAll(".tab-btn")
      .forEach((b) => b.classList.remove("active"));
    document
      .querySelectorAll(".tab-panel")
      .forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(target).classList.add("active");
  });
});

/* ── DB Badge yükleme ─────────────────────────────────────── */
async function loadDbCount() {
  try {
    const res = await fetch("/api/turtles");
    if (!res.ok) return;
    const data = await res.json();
    const el = document.getElementById("db-count");
    if (el) el.textContent = data.count;
  } catch (_) {
    /* sessizce geç */
  }
}

loadDbCount();

/* ═══════════════════════════════════════════════════════════
   KIRPMA BİLEŞENİ (saf Canvas/JS — library yok)
   ═══════════════════════════════════════════════════════════ */

/**
 * Canvas üzerinde interaktif crop kutusu oluşturur.
 * Zoom: fare tekerleği veya butonlar. Pan: Shift+sürükle.
 * Rect IMAGE piksel koordinatlarında saklanır (scale-bağımsız).
 *
 * @param {HTMLCanvasElement} canvas - Hedef canvas elementi
 * @returns {{ setImage, getCroppedBlob, reset, zoomIn, zoomOut, zoomReset, onZoom }}
 */
function createCropComponent(canvas) {
  const ctx = canvas.getContext("2d");
  const MIN_SIZE = 50; // görsel pikseli
  const HANDLE_SIZE = 10; // canvas display pikseli
  const MAX_ZOOM = 8; // 8× yakınlaştırma limiti

  let img = null;
  let rect = null; // { x, y, w, h } — IMAGE piksel koordinatları

  // Viewport: görüntülenen görsel alanı (image piksel)
  let viewX = 0,
    viewY = 0,
    viewW = 0,
    viewH = 0;

  // Etkileşim durumu
  let drawing = false,
    dragging = false,
    resizing = false,
    panning = false;
  let activeHandle = null;
  let dragStart = null; // image px
  let rectStart = null; // image px rect snapshot
  let panStart = null; // { vx, vy, cx, cy }
  let _onZoom = null; // zoom callback

  /* ── Koordinat dönüşümleri ── */
  function canvasToImg(cx, cy) {
    return {
      x: viewX + (cx * viewW) / canvas.width,
      y: viewY + (cy * viewH) / canvas.height,
    };
  }
  function imgToCanvas(ix, iy) {
    return {
      x: ((ix - viewX) * canvas.width) / viewW,
      y: ((iy - viewY) * canvas.height) / viewH,
    };
  }

  /* ── Rect → canvas display koordinatları ── */
  function getRectCanvas() {
    if (!rect) return null;
    const tl = imgToCanvas(rect.x, rect.y);
    const br = imgToCanvas(rect.x + rect.w, rect.y + rect.h);
    return { x: tl.x, y: tl.y, w: br.x - tl.x, h: br.y - tl.y };
  }

  /* ── Handle köşeleri ── */
  function getHandles(rc) {
    const hs = HANDLE_SIZE / 2;
    return {
      tl: { x: rc.x - hs, y: rc.y - hs },
      tr: { x: rc.x + rc.w - hs, y: rc.y - hs },
      bl: { x: rc.x - hs, y: rc.y + rc.h - hs },
      br: { x: rc.x + rc.w - hs, y: rc.y + rc.h - hs },
    };
  }

  function hitHandle(cx, cy) {
    const rc = getRectCanvas();
    if (!rc) return null;
    const h = getHandles(rc);
    for (const [key, pos] of Object.entries(h)) {
      if (
        cx >= pos.x &&
        cx <= pos.x + HANDLE_SIZE &&
        cy >= pos.y &&
        cy <= pos.y + HANDLE_SIZE
      )
        return key;
    }
    return null;
  }

  function inRectCanvas(cx, cy) {
    const rc = getRectCanvas();
    if (!rc) return false;
    return cx >= rc.x && cx <= rc.x + rc.w && cy >= rc.y && cy <= rc.y + rc.h;
  }

  /* ── Viewport sınırla ── */
  function clampViewport() {
    const ar = img.naturalHeight / img.naturalWidth;
    viewW = Math.max(
      img.naturalWidth / MAX_ZOOM,
      Math.min(img.naturalWidth, viewW),
    );
    viewH = viewW * ar;
    viewX = Math.max(0, Math.min(img.naturalWidth - viewW, viewX));
    viewY = Math.max(0, Math.min(img.naturalHeight - viewH, viewY));
  }

  /* ── Çizim ── */
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img) {
      ctx.drawImage(
        img,
        viewX,
        viewY,
        viewW,
        viewH,
        0,
        0,
        canvas.width,
        canvas.height,
      );
    }
    const rc = getRectCanvas();
    if (!rc) return;

    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(0, 0, canvas.width, rc.y);
    ctx.fillRect(0, rc.y + rc.h, canvas.width, canvas.height - rc.y - rc.h);
    ctx.fillRect(0, rc.y, rc.x, rc.h);
    ctx.fillRect(rc.x + rc.w, rc.y, canvas.width - rc.x - rc.w, rc.h);

    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(rc.x, rc.y, rc.w, rc.h);

    ctx.strokeStyle = "rgba(0,255,136,0.3)";
    ctx.lineWidth = 0.5;
    const x1 = rc.x + rc.w / 3,
      x2 = rc.x + (rc.w * 2) / 3;
    const y1 = rc.y + rc.h / 3,
      y2 = rc.y + (rc.h * 2) / 3;
    [
      [rc.x, y1, rc.x + rc.w, y1],
      [rc.x, y2, rc.x + rc.w, y2],
      [x1, rc.y, x1, rc.y + rc.h],
      [x2, rc.y, x2, rc.y + rc.h],
    ].forEach(([sx, sy, ex, ey]) => {
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
    });

    ctx.fillStyle = "#00ff88";
    Object.values(getHandles(rc)).forEach((pos) => {
      ctx.fillRect(pos.x, pos.y, HANDLE_SIZE, HANDLE_SIZE);
    });
  }

  /* ── Fare pozisyonu (CSS ölçeklemesine göre düzeltilmiş) ── */
  function getPos(e) {
    const r = canvas.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: ((cx - r.left) * canvas.width) / r.width,
      y: ((cy - r.top) * canvas.height) / r.height,
    };
  }

  /* ── Fare olay dinleyicileri ── */
  canvas.addEventListener("mousedown", onDown);
  canvas.addEventListener("mousemove", onMove);
  canvas.addEventListener("mouseup", onUp);
  canvas.addEventListener("mouseleave", onUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  function onDown(e) {
    if (!img) return;
    const cp = getPos(e);
    const ip = canvasToImg(cp.x, cp.y);

    if (e.shiftKey || e.button === 1) {
      panning = true;
      panStart = { vx: viewX, vy: viewY, cx: cp.x, cy: cp.y };
      canvas.style.cursor = "grabbing";
      return;
    }

    const handle = hitHandle(cp.x, cp.y);
    if (handle) {
      resizing = true;
      activeHandle = handle;
      rectStart = { ...rect };
      dragStart = ip;
      return;
    }

    if (rect && inRectCanvas(cp.x, cp.y)) {
      dragging = true;
      dragStart = ip;
      rectStart = { ...rect };
      return;
    }

    drawing = true;
    dragStart = ip;
    rect = { x: ip.x, y: ip.y, w: 0, h: 0 };
  }

  function onMove(e) {
    if (!img) return;
    const cp = getPos(e);
    const ip = canvasToImg(cp.x, cp.y);

    if (panning) {
      const dx = ((cp.x - panStart.cx) * viewW) / canvas.width;
      const dy = ((cp.y - panStart.cy) * viewH) / canvas.height;
      viewX = panStart.vx - dx;
      viewY = panStart.vy - dy;
      clampViewport();
      draw();
      return;
    }

    if (drawing) {
      rect.w = ip.x - dragStart.x;
      rect.h = ip.y - dragStart.y;
      draw();
      return;
    }

    if (resizing && rectStart) {
      const dx = ip.x - dragStart.x,
        dy = ip.y - dragStart.y;
      const r = { ...rectStart };
      if (activeHandle === "tl") {
        r.x += dx;
        r.y += dy;
        r.w -= dx;
        r.h -= dy;
      } else if (activeHandle === "tr") {
        r.y += dy;
        r.w += dx;
        r.h -= dy;
      } else if (activeHandle === "bl") {
        r.x += dx;
        r.w -= dx;
        r.h += dy;
      } else if (activeHandle === "br") {
        r.w += dx;
        r.h += dy;
      }
      rect = r;
      draw();
      return;
    }

    if (dragging && rectStart) {
      rect.x = rectStart.x + (ip.x - dragStart.x);
      rect.y = rectStart.y + (ip.y - dragStart.y);
      draw();
      return;
    }

    const handle = hitHandle(cp.x, cp.y);
    if (handle) {
      canvas.style.cursor =
        handle === "tl" || handle === "br" ? "nwse-resize" : "nesw-resize";
    } else if (rect && inRectCanvas(cp.x, cp.y)) {
      canvas.style.cursor = "move";
    } else {
      canvas.style.cursor = e.shiftKey ? "grab" : "crosshair";
    }
  }

  function onUp() {
    if (drawing) {
      drawing = false;
      if (rect.w < 0) {
        rect.x += rect.w;
        rect.w = Math.abs(rect.w);
      }
      if (rect.h < 0) {
        rect.y += rect.h;
        rect.h = Math.abs(rect.h);
      }
      if (rect.w < MIN_SIZE || rect.h < MIN_SIZE) rect = null;
      draw();
    }
    dragging = resizing = panning = false;
    activeHandle = dragStart = rectStart = panStart = null;
    if (img) canvas.style.cursor = "crosshair";
  }

  function onWheel(e) {
    if (!img) return;
    e.preventDefault();
    const cp = getPos(e);
    const ip = canvasToImg(cp.x, cp.y);

    const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    const newViewW = Math.max(
      img.naturalWidth / MAX_ZOOM,
      Math.min(img.naturalWidth, viewW * factor),
    );
    const newViewH = (newViewW * img.naturalHeight) / img.naturalWidth;

    viewX = ip.x - (cp.x / canvas.width) * newViewW;
    viewY = ip.y - (cp.y / canvas.height) * newViewH;
    viewW = newViewW;
    viewH = newViewH;
    clampViewport();
    draw();
    _onZoom && _onZoom(_zoomText());
  }

  /* ── Zoom yardımcısı ── */
  function _doZoom(factor) {
    if (!img) return;
    const cx = canvas.width / 2,
      cy = canvas.height / 2;
    const ip = canvasToImg(cx, cy);
    const newViewW = Math.max(
      img.naturalWidth / MAX_ZOOM,
      Math.min(img.naturalWidth, viewW * factor),
    );
    const newViewH = (newViewW * img.naturalHeight) / img.naturalWidth;
    viewX = ip.x - (cx / canvas.width) * newViewW;
    viewY = ip.y - (cy / canvas.height) * newViewH;
    viewW = newViewW;
    viewH = newViewH;
    clampViewport();
    draw();
    _onZoom && _onZoom(_zoomText());
  }

  function _zoomText() {
    if (!img) return "1.0×";
    return `${(img.naturalWidth / viewW).toFixed(1)}×`;
  }

  /* ── Public API ── */
  function setImage(file) {
    const reader = new FileReader();
    reader.onload = (ev) => {
      const image = new Image();
      image.onload = () => {
        img = image;
        rect = null;
        const MAX_W = canvas.parentElement.clientWidth || 860;
        const s = Math.min(1, MAX_W / image.naturalWidth);
        canvas.width = Math.round(image.naturalWidth * s);
        canvas.height = Math.round(image.naturalHeight * s);
        viewX = 0;
        viewY = 0;
        viewW = image.naturalWidth;
        viewH = image.naturalHeight;
        draw();
        _onZoom && _onZoom("1.0×");
      };
      image.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  }

  function getCroppedBlob() {
    return new Promise((resolve, reject) => {
      if (!img || !rect) {
        reject(new Error("Kırpma alanı seçilmedi."));
        return;
      }
      const rw = Math.abs(rect.w),
        rh = Math.abs(rect.h);
      if (rw < MIN_SIZE || rh < MIN_SIZE) {
        reject(
          new Error(`Kırpma alanı çok küçük (min ${MIN_SIZE}×${MIN_SIZE} px).`),
        );
        return;
      }
      const sx = Math.round(rect.w >= 0 ? rect.x : rect.x + rect.w);
      const sy = Math.round(rect.h >= 0 ? rect.y : rect.y + rect.h);
      const off = document.createElement("canvas");
      off.width = Math.round(rw);
      off.height = Math.round(rh);
      off
        .getContext("2d")
        .drawImage(
          img,
          sx,
          sy,
          Math.round(rw),
          Math.round(rh),
          0,
          0,
          off.width,
          off.height,
        );
      off.toBlob(
        (blob) =>
          blob ? resolve(blob) : reject(new Error("Blob üretilemedi.")),
        "image/jpeg",
        0.92,
      );
    });
  }

  function reset() {
    img = null;
    rect = null;
    viewX = viewY = viewW = viewH = 0;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    _onZoom && _onZoom("1.0×");
  }

  return {
    setImage,
    getCroppedBlob,
    reset,
    zoomIn: () => _doZoom(1 / 1.5),
    zoomOut: () => _doZoom(1.5),
    zoomReset: () => {
      if (!img) return;
      viewX = 0;
      viewY = 0;
      viewW = img.naturalWidth;
      viewH = img.naturalHeight;
      clampViewport();
      draw();
      _onZoom && _onZoom("1.0×");
    },
    onZoom: (fn) => {
      _onZoom = fn;
    },
  };
}

/* ── Ortak yardımcılar ──────────────────────────────────────────── */

/**
 * Dosya tipini ve boyutunu doğrular.
 * @param {File} file
 * @returns {string|null} Hata mesajı ya da geçerliyse null
 */
function validateImageFile(file) {
  if (!file.type.match(/^image\/(jpeg|png)$/))
    return "Sadece JPG ve PNG desteklenir.";
  if (file.size > MAX_FILE_BYTES)
    return `Dosya boyutu ${MAX_FILE_BYTES / 1024 / 1024} MB'ı geçemez.`;
  return null;
}

/**
 * Drag-drop, click ve dosya-seçimi olaylarını drop zone'a bağlar.
 * @param {HTMLElement}      dropZone
 * @param {HTMLInputElement} fileInput
 * @param {(file: File) => void} onFile
 */
function wireDragDrop(dropZone, fileInput, onFile) {
  ["dragenter", "dragover"].forEach((ev) =>
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    }),
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropZone.addEventListener(ev, () => dropZone.classList.remove("dragover")),
  );
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) onFile(fileInput.files[0]);
  });
}

/**
 * Zoom butonlarını ve zoom seviyesi göstergesini crop bileşenine bağlar.
 * @param {string} prefix        - Element ID öneki ("id" veya "reg")
 * @param {object} cropComponent - createCropComponent() döndürdüğü nesne
 */
function wireZoomButtons(prefix, cropComponent) {
  const levelEl = document.getElementById(`${prefix}-zoom-level`);
  cropComponent.onZoom((txt) => {
    if (levelEl) levelEl.textContent = txt;
  });
  document
    .getElementById(`${prefix}-zoom-in`)
    .addEventListener("click", () => cropComponent.zoomIn());
  document
    .getElementById(`${prefix}-zoom-out`)
    .addEventListener("click", () => cropComponent.zoomOut());
  document
    .getElementById(`${prefix}-zoom-reset`)
    .addEventListener("click", () => cropComponent.zoomReset());
}

/* ═══════════════════════════════════════════════════════════
   TANI TABI
   ═══════════════════════════════════════════════════════════ */

const idDropZone = document.getElementById("id-drop-zone");
const idFileInput = document.getElementById("file-input");
const idCropZone = document.getElementById("id-crop-zone");
const idCropCanvas = document.getElementById("crop-canvas");
const idCropBtn = document.getElementById("id-crop-btn");
const idResetBtn = document.getElementById("id-reset-btn");
const idAlert = document.getElementById("id-alert");
const pipelineEl = document.getElementById("pipeline");
const resultCard = document.getElementById("result-card");

const idCrop = createCropComponent(idCropCanvas);

wireZoomButtons("id", idCrop);

/* ── Hata/uyarı gösterici ────────────────────────────────── */
function showAlert(el, msg) {
  el.textContent = msg;
  el.classList.add("visible");
}

function hideAlert(el) {
  el.classList.remove("visible");
}

/* ── Drag & Drop ─────────────────────────────────────────── */
wireDragDrop(idDropZone, idFileInput, loadImageForId);

function loadImageForId(file) {
  const err = validateImageFile(file);
  if (err) {
    showAlert(idAlert, err);
    return;
  }
  hideAlert(idAlert);
  pipelineEl.classList.remove("visible");
  resultCard.classList.remove("visible");
  resultCard.className = "result-card";
  idCropZone.classList.add("visible");
  idCrop.setImage(file);
}

idResetBtn.addEventListener("click", () => {
  idCrop.reset();
  idCropZone.classList.remove("visible");
  pipelineEl.classList.remove("visible");
  resultCard.classList.remove("visible");
  resultCard.className = "result-card";
  hideAlert(idAlert);
  idFileInput.value = "";
});

/* ── Pipeline adım tanımları ─────────────────────────────── */
const PIPELINE_STEPS = [
  {
    key: "audit",
    name: "AuditWorker",
    msg: "Dosya doğrulanıyor...",
    doneMsg: "Dosya geçerli.",
  },
  {
    key: "head",
    name: "HeadDetectionWorker",
    msg: "Gemini doğruluyor...",
    doneMsg: "Kafa profili onaylandı.",
  },
  {
    key: "preprocess",
    name: "PreprocessingWorker",
    msg: "Tensör hazırlanıyor...",
    doneMsg: "Tensör hazır.",
  },
  {
    key: "recognition",
    name: "RecognitionWorker",
    msg: "Embedding üretiliyor...",
    doneMsg: "Embedding tamamlandı.",
  },
  {
    key: "evaluation",
    name: "EvaluationWorker",
    msg: "Veritabanı taranıyor...",
    doneMsg: "Eşleşme hesaplandı.",
  },
  {
    key: "reporting",
    name: "ReportingWorker",
    msg: "Rapor oluşturuluyor...",
    doneMsg: "Rapor hazır.",
  },
];

function resetPipeline() {
  PIPELINE_STEPS.forEach((s) => {
    const el = document.getElementById(`step-${s.key}`);
    if (!el) return;
    el.className = "step";
    el.querySelector(".step-icon").textContent = "○";
    el.querySelector(".step-msg").textContent = "—";
  });
}

function activateStep(key, msg) {
  const el = document.getElementById(`step-${key}`);
  if (!el) return;
  el.className = "step active";
  el.querySelector(".step-icon").innerHTML = '<span class="spinner"></span>';
  el.querySelector(".step-msg").textContent = msg;
}

function completeStep(key, msg, failed = false) {
  const el = document.getElementById(`step-${key}`);
  if (!el) return;
  el.className = failed ? "step failed" : "step done";
  el.querySelector(".step-icon").textContent = failed ? "✗" : "✓";
  el.querySelector(".step-msg").textContent = msg;
}

/**
 * Pipeline log satırlarını parse ederek hangi adımların geçtiğini
 * tespit eder.
 * @param {string[]} logLines
 * @returns {Set<string>} - Tamamlanan supervisor adım adları (küçük harf)
 */
function parseCompletedSteps(logLines) {
  const completed = new Set();
  logLines.forEach((line) => {
    if (line.includes("AuditWorker") && line.includes("completed"))
      completed.add("audit");
    if (line.includes("HeadDetectionWorker") && line.includes("completed"))
      completed.add("head");
    if (line.includes("PreprocessingWorker") && line.includes("completed"))
      completed.add("preprocess");
    if (line.includes("RecognitionWorker") && line.includes("completed"))
      completed.add("recognition");
    if (line.includes("EvaluationWorker") && line.includes("completed"))
      completed.add("evaluation");
    if (line.includes("ReportingWorker") && line.includes("completed"))
      completed.add("reporting");
  });
  return completed;
}

/**
 * Pipeline adımlarını log verisine göre animasyonlu şekilde oynatır.
 * @param {string[]} logLines
 * @param {boolean} overallSuccess
 * @returns {Promise<void>}
 */
async function animatePipeline(logLines, overallSuccess) {
  const completed = parseCompletedSteps(logLines);

  for (let i = 0; i < PIPELINE_STEPS.length; i++) {
    const s = PIPELINE_STEPS[i];
    activateStep(s.key, s.msg);
    await delay(PIPELINE_DELAY_MS);
    const isDone = completed.has(s.key);
    /* Son adım pipeline başarısıyla tamamlandıysa hep done */
    if (overallSuccess) {
      completeStep(s.key, s.doneMsg, false);
    } else if (isDone) {
      completeStep(s.key, s.doneMsg, false);
    } else {
      completeStep(s.key, "Başarısız.", true);
      break;
    }
  }
}

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/* ── Tanı butonu ─────────────────────────────────────────── */
/**
 * Kırpılmış görseli /api/identify'a gönderir.
 * @returns {Promise<object>} API yanıt nesnesi
 */
async function cropAndIdentify() {
  const blob = await idCrop.getCroppedBlob();
  const form = new FormData();
  form.append("image", blob, "crop.jpg");
  const res = await fetch("/api/identify", { method: "POST", body: form });
  return res.json();
}

/** Tanı akışının tüm UI geçişlerini yönetir. */
async function runIdentifyFlow() {
  hideAlert(idAlert);
  resetPipeline();
  pipelineEl.classList.add("visible");
  resultCard.classList.remove("visible");
  resultCard.className = "result-card";
  idCropBtn.disabled = true;
  idCropBtn.innerHTML =
    '<span class="btn-loading"><span class="spinner"></span> Analiz ediliyor...</span>';

  try {
    const data = await cropAndIdentify();
    await animatePipeline(data.mission_log || [], data.success);
    renderResult(data);
  } catch (err) {
    const msg = /seçilmedi|küçük/.test(err.message || "")
      ? "Lütfen kafa bölgesini seçin (min 50×50 px)."
      : "Sunucuya bağlanılamadı. Lütfen sunucunun çalıştığından emin olun.";
    showAlert(idAlert, msg);
  } finally {
    idCropBtn.disabled = false;
    idCropBtn.textContent = "KIRP VE TANI";
  }
}

idCropBtn.addEventListener("click", runIdentifyFlow);

/**
 * API yanıtına göre sonuç kartını doldurur ve gösterir.
 * @param {object} data - /api/identify yanıtı
 */
function renderResult(data) {
  resultCard.classList.remove("visible");
  resultCard.className = "result-card";

  if (!data.success) {
    /* Hata kartı */
    const errMsg = data.error || "Bilinmeyen hata.";
    let userMsg = errMsg;
    if (
      errMsg.toLowerCase().includes("kafa") ||
      errMsg.toLowerCase().includes("head")
    ) {
      userMsg =
        "Kaplumbağa kafası tespit edilemedi. Lütfen yan profil fotoğrafı yükleyin.";
    }
    resultCard.innerHTML = `
      <div class="result-inner">
        <div class="result-header">
          <span class="badge" style="background:var(--danger);box-shadow:0 0 8px var(--danger)"></span>
          TANI BAŞARISIZ
        </div>
        <div class="result-error-msg">${escHtml(userMsg)}</div>
      </div>`;
    resultCard.classList.add("result-error", "visible");
    return;
  }

  const scorePct = data.score ?? 0;
  const status = data.status ?? "YENİ_BİREY";
  const identity = data.identity ?? "Kayıtlı değil";

  resultCard.innerHTML = `
    <div class="result-inner">
      <div class="result-header">
        <span class="badge"></span>
        KİMLİK TESPİT EDİLDİ
      </div>
      <div class="result-name">${escHtml(identity.toUpperCase())}</div>
      <div class="result-species">Caretta caretta</div>
      <div class="result-score">
        <div class="score-label">BENZERLİK SKORU</div>
        <div class="score-bar-wrap">
          <div class="score-bar-bg">
            <div class="score-bar-fill" id="score-fill" style="width:0%"></div>
          </div>
          <div class="score-pct" id="score-pct">%0</div>
        </div>
      </div>
      <div class="result-status ${escHtml(status)}">${escHtml(status.replace("_", " "))}</div>
    </div>`;
  resultCard.classList.add("visible");

  /* Progress bar animasyonu */
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.getElementById("score-fill").style.width = `${scorePct}%`;
      document.getElementById("score-pct").textContent =
        `%${scorePct.toFixed(1)}`;
    });
  });
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ═══════════════════════════════════════════════════════════
   KAYIT TABI
   ═══════════════════════════════════════════════════════════ */

const regDropZone = document.getElementById("reg-drop-zone");
const regFileInput = document.getElementById("reg-file-input");
const regCropZone = document.getElementById("reg-crop-zone");
const regCropCanvas = document.getElementById("reg-crop-canvas");
const regCropBtn = document.getElementById("reg-crop-btn");
const regResetBtn = document.getElementById("reg-reset-btn");
const regSubmitBtn = document.getElementById("reg-submit-btn");
const regAlert = document.getElementById("reg-alert");
const regName = document.getElementById("reg-name");
const regNotes = document.getElementById("reg-notes");

const regCrop = createCropComponent(regCropCanvas);

wireZoomButtons("reg", regCrop);
let regBlob = null;

wireDragDrop(regDropZone, regFileInput, loadImageForReg);

function loadImageForReg(file) {
  const err = validateImageFile(file);
  if (err) {
    showAlert(regAlert, err);
    return;
  }
  hideAlert(regAlert);
  regBlob = null;
  regCropZone.classList.add("visible");
  regCrop.setImage(file);
}

regResetBtn.addEventListener("click", () => {
  regCrop.reset();
  regCropZone.classList.remove("visible");
  regBlob = null;
  hideAlert(regAlert);
  regFileInput.value = "";
  regSubmitBtn.disabled = true;
});

/* "Kırp" butonu — önce kırpma confirm edilsin */
regCropBtn.addEventListener("click", async () => {
  hideAlert(regAlert);
  try {
    regBlob = await regCrop.getCroppedBlob();
    regSubmitBtn.disabled = false;
    showAlert(
      regAlert,
      "✓ Kırpma tamamlandı. Formu doldurup KAYDET butonuna basın.",
    );
    regAlert.className = "alert alert-success visible";
  } catch (_) {
    showAlert(regAlert, "Lütfen kafa bölgesini seçin (min 50×50 px).");
    regAlert.className = "alert alert-error visible";
  }
});

/* Kaydet */
regSubmitBtn.addEventListener("click", async () => {
  hideAlert(regAlert);

  if (!regBlob) {
    showAlert(regAlert, "Lütfen görsel yükleyin ve kafa bölgesini kırpın.");
    regAlert.className = "alert alert-error visible";
    return;
  }

  const name = regName.value.trim();
  if (!name) {
    showAlert(regAlert, "Lütfen bir isim girin.");
    regAlert.className = "alert alert-error visible";
    return;
  }

  const profileEl = document.querySelector('input[name="profile"]:checked');
  if (!profileEl) {
    showAlert(regAlert, "Lütfen profil tarafı seçin (Sağ / Sol).");
    regAlert.className = "alert alert-error visible";
    return;
  }

  regSubmitBtn.disabled = true;
  regSubmitBtn.innerHTML =
    '<span class="btn-loading"><span class="spinner"></span> Kaydediliyor...</span>';

  const form = new FormData();
  form.append("image", regBlob, "register.jpg");
  form.append("name", name);
  form.append("profile", profileEl.value);
  form.append("notes", regNotes.value.trim());

  try {
    const res = await fetch("/api/register", { method: "POST", body: form });
    const data = await res.json();

    if (data.success) {
      regAlert.className = "alert alert-success visible";
      regAlert.textContent = `✓ ${data.message} (${data.species ?? ""})`;
      loadDbCount();
    } else {
      regAlert.className = "alert alert-error visible";
      regAlert.textContent = data.message || "Kayıt başarısız.";
    }
  } catch (_) {
    regAlert.className = "alert alert-error visible";
    regAlert.textContent = "Sunucuya bağlanılamadı.";
  }

  regSubmitBtn.disabled = false;
  regSubmitBtn.textContent = "KAYDET";
});
