/* core.js — ขั้นตอนเตรียมรูป + ตรวจคุณภาพ เขียนให้ตรงกับ Python (PIL) ที่ใช้ตอนเทรน/เซิร์ฟเวอร์
 * ใช้ได้ทั้งในเบราว์เซอร์และ Node (สำหรับทดสอบ) */
(function (root) {
  'use strict';

  // ===== ค่าเกณฑ์ ต้องตรงกับ quality.py =====
  const CFG = {
    MIN_SHARPNESS: 10.0,
    MIN_CONTRAST: 10.0,
    MIN_BRIGHTNESS: 45.0,
    MIN_CONFIDENCE: 0.50,
  };
  const MEAN = [0.485, 0.456, 0.406];
  const STD = [0.229, 0.224, 0.225];

  // ===== ย่อรูปแบบเดียวกับ PIL (antialias, fixed-point 8-bit) =====
  const PREC = 22;
  const ONE = 1 << PREC;
  const HALF = 1 << (PREC - 1);

  const FILTERS = {
    bilinear: { support: 1.0, f: (x) => { x = Math.abs(x); return x < 1 ? 1 - x : 0; } },
    bicubic: {
      support: 2.0,
      f: (x) => {
        const a = -0.5; x = Math.abs(x);
        if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
        if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
        return 0;
      },
    },
  };

  function coeffs(inSize, outSize, filt) {
    const scale = inSize / outSize;
    const fscale = Math.max(scale, 1);
    const support = filt.support * fscale;
    const ss = 1 / fscale;
    const ksize = Math.ceil(support) * 2 + 1;
    const bounds = new Int32Array(outSize * 2);
    const kk = new Int32Array(outSize * ksize);
    const tmp = new Float64Array(ksize);
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      let ww = 0;
      for (let x = 0; x < xmax; x++) {
        const w = filt.f((x + xmin - center + 0.5) * ss);
        tmp[x] = w; ww += w;
      }
      for (let x = 0; x < xmax; x++) {
        const w = ww !== 0 ? tmp[x] / ww : tmp[x];
        kk[xx * ksize + x] = Math.trunc(w < 0 ? -0.5 + w * ONE : 0.5 + w * ONE);
      }
      bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
    }
    return { bounds, kk, ksize };
  }

  const clip8 = (v) => { v = Math.floor(v / ONE); return v < 0 ? 0 : v > 255 ? 255 : v; };

  /** rgba: Uint8ClampedArray/Uint8Array (w*h*4) -> Uint8Array RGB (outW*outH*3) */
  function resizeRGB(rgba, w, h, outW, outH, kind) {
    const filt = FILTERS[kind];
    let cur = new Uint8Array(w * h * 3);
    for (let i = 0, j = 0; i < w * h * 4; i += 4, j += 3) {
      cur[j] = rgba[i]; cur[j + 1] = rgba[i + 1]; cur[j + 2] = rgba[i + 2];
    }
    let cw = w, ch = h;
    if (outW !== cw) {                       // แนวนอนก่อน
      const { bounds, kk, ksize } = coeffs(cw, outW, filt);
      const out = new Uint8Array(outW * ch * 3);
      for (let y = 0; y < ch; y++) {
        for (let xx = 0; xx < outW; xx++) {
          const xmin = bounds[xx * 2], n = bounds[xx * 2 + 1], kb = xx * ksize;
          for (let c = 0; c < 3; c++) {
            let s = HALF;
            for (let x = 0; x < n; x++) s += cur[(y * cw + xmin + x) * 3 + c] * kk[kb + x];
            out[(y * outW + xx) * 3 + c] = clip8(s);
          }
        }
      }
      cur = out; cw = outW;
    }
    if (outH !== ch) {                       // แล้วแนวตั้ง
      const { bounds, kk, ksize } = coeffs(ch, outH, filt);
      const out = new Uint8Array(cw * outH * 3);
      for (let yy = 0; yy < outH; yy++) {
        const ymin = bounds[yy * 2], n = bounds[yy * 2 + 1], kb = yy * ksize;
        for (let x = 0; x < cw; x++) {
          for (let c = 0; c < 3; c++) {
            let s = HALF;
            for (let y = 0; y < n; y++) s += cur[((ymin + y) * cw + x) * 3 + c] * kk[kb + y];
            out[(yy * cw + x) * 3 + c] = clip8(s);
          }
        }
      }
      cur = out; ch = outH;
    }
    return cur;
  }

  /** เตรียมรูปเข้าโมเดล: Resize(bilinear) -> ToTensor -> Normalize  => Float32Array CHW */
  function preprocess(rgba, w, h, size) {
    const rgb = resizeRGB(rgba, w, h, size, size, 'bilinear');
    const n = size * size;
    const out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        out[c * n + i] = (rgb[i * 3 + c] / 255 - MEAN[c]) / STD[c];
      }
    }
    return out;
  }

  // ===== ตรวจคุณภาพ (ตรงกับ quality.measure) =====
  function measureQuality(rgba, w, h) {
    const S = 512;
    const rgb = resizeRGB(rgba, w, h, S, S, 'bicubic');
    const N = S * S;
    const g = new Float32Array(N);
    const mask = new Uint8Array(N);
    let cnt = 0;
    for (let i = 0; i < N; i++) {
      g[i] = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2];
      if (g[i] > 20) { mask[i] = 1; cnt++; }
    }
    if (cnt / N < 0.2) return { sharpness: 0, contrast: 0, brightness: 0 };

    // contrast / brightness ในพื้นที่ mask
    let sum = 0;
    for (let i = 0; i < N; i++) if (mask[i]) sum += g[i];
    const mean = sum / cnt;
    let sq = 0;
    for (let i = 0; i < N; i++) if (mask[i]) { const d = g[i] - mean; sq += d * d; }
    const contrast = Math.sqrt(sq / cnt);

    // erosion 9x9 ด้วย integral image (นอกภาพนับเป็น False)
    const W1 = S + 1;
    const ii = new Int32Array(W1 * W1);
    for (let y = 0; y < S; y++) {
      let row = 0;
      for (let x = 0; x < S; x++) {
        row += mask[y * S + x];
        ii[(y + 1) * W1 + (x + 1)] = ii[y * W1 + (x + 1)] + row;
      }
    }
    const r = 4;
    const inner = (y, x) => {
      if (y - r < 0 || x - r < 0 || y + r > S - 1 || x + r > S - 1) return false;
      const y0 = y - r, y1 = y + r + 1, x0 = x - r, x1 = x + r + 1;
      return ii[y1 * W1 + x1] - ii[y0 * W1 + x1] - ii[y1 * W1 + x0] + ii[y0 * W1 + x0] === 81;
    };
    let n = 0, ls = 0, lq = 0;
    const lap = [];
    for (let y = 1; y < S - 1; y++) {
      for (let x = 1; x < S - 1; x++) {
        if (!inner(y, x)) continue;
        const i = y * S + x;
        const v = 4 * g[i] - g[i - S] - g[i + S] - g[i - 1] - g[i + 1];
        lap.push(v); ls += v; n++;
      }
    }
    let sharp = 0;
    if (n > 0) {
      const lm = ls / n;
      for (let k = 0; k < n; k++) { const d = lap[k] - lm; lq += d * d; }
      sharp = lq / n;
    }
    const r1 = (v) => Math.round(v * 10) / 10;
    return { sharpness: r1(sharp), contrast: r1(contrast), brightness: r1(mean) };
  }

  function softmax(logits) {
    let m = -Infinity;
    for (const v of logits) if (v > m) m = v;
    const e = Array.from(logits, (v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  function assess(q, probs) {
    const reasons = [];
    if (q.sharpness < CFG.MIN_SHARPNESS) reasons.push('ภาพเบลอหรือไม่คมชัด');
    if (q.contrast < CFG.MIN_CONTRAST) reasons.push('ภาพหม่น คอนทราสต์ต่ำ');
    if (q.brightness < CFG.MIN_BRIGHTNESS) reasons.push('ภาพมืดเกินไป');
    if (Math.max(...probs) < CFG.MIN_CONFIDENCE) reasons.push('โมเดลไม่แน่ใจ (ความเชื่อมั่นต่ำ)');
    return { quality: q, assessable: reasons.length === 0, reasons };
  }

  const api = { CFG, resizeRGB, preprocess, measureQuality, softmax, assess };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.DRCore = api;
})(typeof self !== 'undefined' ? self : this);
