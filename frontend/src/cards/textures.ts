import * as THREE from 'three';
import { COL_HEX, COL_DARK } from './constants';

const FONT = "Nunito, 'Arial Black', sans-serif";

function rr(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function glyphNumber(ctx: CanvasRenderingContext2D, t: string, color: number, s: number) {
  ctx.font = `900 ${s}px ${FONT}`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
  ctx.strokeStyle = COL_DARK[color]; ctx.lineWidth = s * 0.18;
  ctx.strokeText(t, 0, s * 0.06);
  ctx.fillStyle = COL_HEX[color];
  ctx.fillText(t, 0, s * 0.06);
}

function drawSkipMark(ctx: CanvasRenderingContext2D, color: number, s: number) {
  ctx.lineCap = 'round';
  for (const [style, lw] of [[COL_DARK[color], s * 0.2], [COL_HEX[color], s * 0.13]] as [string, number][]) {
    ctx.strokeStyle = style; ctx.lineWidth = lw;
    ctx.beginPath(); ctx.arc(0, 0, s * 0.42, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-s * 0.31, s * 0.31); ctx.lineTo(s * 0.31, -s * 0.31); ctx.stroke();
  }
}

function drawReverseMark(ctx: CanvasRenderingContext2D, color: number, s: number) {
  const arrow = (dir: number, cy: number) => {
    const L = s * 0.88, H = s * 0.17, head = s * 0.3;
    ctx.beginPath();
    ctx.moveTo(-dir * L / 2, cy - H / 2);
    ctx.lineTo(dir * (L / 2 - head), cy - H / 2);
    ctx.lineTo(dir * (L / 2 - head), cy - head * 0.72);
    ctx.lineTo(dir * L / 2, cy);
    ctx.lineTo(dir * (L / 2 - head), cy + head * 0.72);
    ctx.lineTo(dir * (L / 2 - head), cy + H / 2);
    ctx.lineTo(-dir * L / 2, cy + H / 2);
    ctx.closePath();
    ctx.fillStyle = COL_HEX[color];
    ctx.strokeStyle = COL_DARK[color];
    ctx.lineWidth = s * 0.045;
    ctx.fill(); ctx.stroke();
  };
  arrow(1, -s * 0.2); arrow(-1, s * 0.2);
}

function drawCenterGlyph(ctx: CanvasRenderingContext2D, type: number, color: number, s: number) {
  if (type <= 9) glyphNumber(ctx, String(type), color, s);
  else if (type === 10) drawSkipMark(ctx, color, s * 1.1);
  else if (type === 11) drawReverseMark(ctx, color, s * 1.02);
  else glyphNumber(ctx, '+2', color, s * 0.82);
}

function miniChip(ctx: CanvasRenderingContext2D, x: number, y: number, type: number, color: number) {
  const w = 52;
  rr(ctx, x, y, w, w, 12); ctx.fillStyle = '#fdfdfb'; ctx.fill();
  ctx.save(); ctx.translate(x + w / 2, y + w / 2);
  drawCenterGlyph(ctx, type, color, w * 0.62);
  ctx.restore();
}

function drawColorFace(ctx: CanvasRenderingContext2D, color: number, type: number) {
  ctx.fillStyle = '#fdfdfb'; ctx.fillRect(0, 0, 256, 384);
  rr(ctx, 7, 7, 242, 370, 26); ctx.fillStyle = COL_HEX[color]; ctx.fill();
  rr(ctx, 7, 7, 242, 370, 26); ctx.strokeStyle = 'rgba(0,0,0,0.22)'; ctx.lineWidth = 3; ctx.stroke();
  ctx.save(); ctx.translate(128, 192); ctx.rotate(-0.5);
  ctx.beginPath(); ctx.ellipse(0, 0, 88, 150, 0, 0, Math.PI * 2);
  ctx.fillStyle = '#fdfdfb'; ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.10)'; ctx.lineWidth = 2; ctx.stroke();
  drawCenterGlyph(ctx, type, color, 116);
  ctx.restore();
  miniChip(ctx, 18, 16, type, color);
  ctx.save(); ctx.translate(256, 384); ctx.rotate(Math.PI); miniChip(ctx, 18, 16, type, color); ctx.restore();
}

function drawWildFace(ctx: CanvasRenderingContext2D, plus4: boolean) {
  ctx.fillStyle = '#fdfdfb'; ctx.fillRect(0, 0, 256, 384);
  rr(ctx, 7, 7, 242, 370, 26); ctx.fillStyle = '#14161d'; ctx.fill();
  const g = ctx.createLinearGradient(7, 0, 249, 0);
  g.addColorStop(0, '#e0453a'); g.addColorStop(0.33, '#f2c53d');
  g.addColorStop(0.55, '#34c759'); g.addColorStop(0.78, '#3b82f6'); g.addColorStop(1, '#a855f7');
  rr(ctx, 7, 7, 242, 370, 26); ctx.strokeStyle = g; ctx.lineWidth = 7; ctx.stroke();
  rr(ctx, 17, 17, 222, 350, 21); ctx.strokeStyle = 'rgba(255,255,255,0.28)'; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.save(); ctx.translate(128, 192); ctx.rotate(-0.5);
  ctx.beginPath(); ctx.ellipse(0, 0, 88, 150, 0, 0, Math.PI * 2);
  ctx.save(); ctx.clip();
  ctx.fillStyle = COL_HEX[0]; ctx.fillRect(-90, -155, 90, 155);
  ctx.fillStyle = COL_HEX[1]; ctx.fillRect(0, -155, 90, 155);
  ctx.fillStyle = COL_HEX[2]; ctx.fillRect(0, 0, 90, 155);
  ctx.fillStyle = COL_HEX[3]; ctx.fillRect(-90, 0, 90, 155);
  ctx.restore();
  ctx.strokeStyle = 'rgba(0,0,0,0.4)'; ctx.lineWidth = 3; ctx.stroke();
  ctx.font = plus4 ? `900 84px ${FONT}` : `900 62px ${FONT}`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
  ctx.strokeStyle = '#000'; ctx.lineWidth = 10;
  ctx.strokeText(plus4 ? '+4' : 'WILD', 0, 4);
  ctx.fillStyle = '#fdfdfb';
  ctx.fillText(plus4 ? '+4' : 'WILD', 0, 4);
  ctx.restore();
  const chip = (x: number, y: number) => {
    rr(ctx, x, y, 52, 52, 12); ctx.fillStyle = '#14161d'; ctx.fill();
    const g2 = ctx.createLinearGradient(x, y, x + 52, y + 52);
    g2.addColorStop(0, '#e0453a'); g2.addColorStop(0.5, '#f2c53d'); g2.addColorStop(1, '#3b82f6');
    rr(ctx, x, y, 52, 52, 12); ctx.strokeStyle = g2; ctx.lineWidth = 4; ctx.stroke();
    ctx.font = `900 26px ${FONT}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = '#fff'; ctx.fillText(plus4 ? '+4' : 'W', x + 26, y + 28);
  };
  chip(18, 16);
  ctx.save(); ctx.translate(256, 384); ctx.rotate(Math.PI); chip(18, 16); ctx.restore();
}

function drawBack(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = '#fdfdfb'; ctx.fillRect(0, 0, 256, 384);
  rr(ctx, 7, 7, 242, 370, 26); ctx.fillStyle = '#191c27'; ctx.fill();
  rr(ctx, 7, 7, 242, 370, 26); ctx.strokeStyle = '#2c3242'; ctx.lineWidth = 2; ctx.stroke();
  ctx.save(); ctx.translate(128, 192); ctx.rotate(-0.5);
  ctx.beginPath(); ctx.ellipse(0, 0, 90, 148, 0, 0, Math.PI * 2);
  ctx.fillStyle = '#d93a30'; ctx.fill();
  ctx.strokeStyle = '#7e1d17'; ctx.lineWidth = 5; ctx.stroke();
  ctx.beginPath(); ctx.ellipse(0, 0, 66, 110, 0, 0, Math.PI * 2);
  ctx.fillStyle = '#f3c73e'; ctx.fill();
  ctx.font = `italic 900 58px ${FONT}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillStyle = '#17181c';
  ctx.fillText('UNO', 0, 4);
  ctx.restore();
}

const faceCache = new Map<number, THREE.CanvasTexture>();
export function getFaceTexture(id: number): THREE.CanvasTexture {
  let t = faceCache.get(id);
  if (!t) {
    const cv = document.createElement('canvas'); cv.width = 256; cv.height = 384;
    const ctx = cv.getContext('2d')!;
    if (id >= 52) drawWildFace(ctx, id === 53);
    else drawColorFace(ctx, Math.floor(id / 13), id % 13);
    t = new THREE.CanvasTexture(cv);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    faceCache.set(id, t);
  }
  return t;
}

let _backTex: THREE.CanvasTexture | null = null;
export function getBackTexture(): THREE.CanvasTexture {
  if (!_backTex) {
    const cv = document.createElement('canvas'); cv.width = 256; cv.height = 384;
    drawBack(cv.getContext('2d')!);
    _backTex = new THREE.CanvasTexture(cv);
    _backTex.colorSpace = THREE.SRGBColorSpace;
    _backTex.anisotropy = 4;
  }
  return _backTex;
}