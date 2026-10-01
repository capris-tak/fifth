// ホシ (Hoshi) — the little star creature. Pure function of t.
// Hoshi.draw(ctx, t, o) — world space, (o.x, o.y) = centre of the star body. Diameter ≈ 110 px at scale 1.
// o = {x, y, scale=1, rot=0, bright:0..1, flicker:0..1, expr, mouth, lookX, lookY, squash:0..1, glow=1,
//      // extras (all optional):
//      wave:0..1      (override arm-wave amount; default from expr),
//      tears:0..1     (override tear amount; default 1 for 'sad'),
//      blink:bool     (auto blink, default true),
//      seed:number    (desync idle motion between multiple Hoshis),
//      halo:0..1      (multiplier for halo/rays only, default 1 — e.g. 0 for a tiny far-away star),
//      tilt:rad       (extra body lean, added to idle sway)}
(function(){
const TAU = Math.PI*2;
const DIM  = {core:'#b4bcdc', body:'#9aa3c7', rim:'#6c7399', line:'#4c5278'};
const BRT  = {core:'#fffbe6', body:'#ffe27a', rim:'#ffb84d', line:'#d98a2b'};
const WARM = '#ffd27a';

// colour of the body at normalised radius p (0 centre .. 1 rim) for brightness b.
// Light fills from the core outward: inner radii warm up first.
function bodyColor(p, b){
  const dim = p<.5 ? U.mixHex(DIM.core, DIM.body, p/.5) : U.mixHex(DIM.body, DIM.rim, (p-.5)/.5);
  const brt = p<.45 ? U.mixHex(BRT.core, BRT.body, p/.45) : U.mixHex(BRT.body, BRT.rim, (p-.45)/.55);
  const front = b*1.45 - .15;                    // radius the warm light has reached
  let k = U.clamp((front - p)/.35 + .5);
  k = U.smooth(k) * U.clamp(b*3);               // tiny b -> no fill
  // half-filled zones look like warm lamp light (slightly more orange) before becoming star-gold
  const mid = U.mixHex(dim, WARM, .55);
  return k<.5 ? U.mixHex(dim, mid, k*2) : U.mixHex(mid, brt, (k-.5)*2 * U.clamp(b*1.3));
}

// 4-point twinkle sparkle path
function sparkle(ctx, x, y, r){
  const q = r*.22;
  ctx.moveTo(x, y-r);
  ctx.quadraticCurveTo(x+q, y-q, x+r, y);
  ctx.quadraticCurveTo(x+q, y+q, x, y+r);
  ctx.quadraticCurveTo(x-q, y+q, x-r, y);
  ctx.quadraticCurveTo(x-q, y-q, x, y-r);
}

// expression rig parameters
const EXPR = {
  hurt:      {tuftA:-.22, tuftL:.94, armA:.30, armL:.92, wave:0, eye:'squint', lid:0, tilt:0,  blush:.55, look:[0,0]},
  sad:       {tuftA:.30, tuftL:.88, armA:.42, armL:.9,  wave:0, eye:'open', lid:.34, lidTilt:.55, blush:.4, look:[0,.25], tears:1},
  hope:      {tuftA:.04, tuftL:1.04,armA:-.08,armL:.98, wave:0, eye:'open', lid:0, sparkle:1, blush:.6, look:[0,-.35], big:1.08},
  surprised: {tuftA:0,   tuftL:1.12,armA:-.42,armL:1.04,wave:0, eye:'open', lid:0, blush:.35, look:[0,0], big:1.16, small:1},
  joy:       {tuftA:0,   tuftL:1.06,armA:-.36,armL:1.04,wave:1, eye:'happy', lid:0, blush:1, look:[0,0]},
  gentle:    {tuftA:.12, tuftL:1.0, armA:.08, armL:.97, wave:.25,eye:'open', lid:.12, lidTilt:-.1, lower:.38, blush:.75, look:[0,.1]},
};

function draw(ctx, t, o){
  o = o||{};
  const E = EXPR[o.expr] || EXPR.gentle;
  const sc = o.scale==null?1:o.scale, glow = o.glow==null?1:o.glow, haloK = o.halo==null?1:o.halo;
  const flick = U.clamp(o.flicker||0), sq = U.clamp(o.squash||0);
  const seed = o.seed||0, ts = t + seed*7.31;
  const lookX = U.clamp(o.lookX!=null?o.lookX:E.look[0], -1, 1), lookY = U.clamp(o.lookY!=null?o.lookY:E.look[1], -1, 1);
  // sputter: when dim, brightness occasionally hiccups up a little
  const sput = flick * Math.pow(U.clamp(U.noise(ts*6.5, 11)*1.8 - .55), 2) * (1-U.clamp(o.bright||0));
  const b = U.clamp((o.bright||0) + sput*.35);
  const R = 55;

  ctx.save();
  ctx.translate(o.x||0, o.y||0);
  ctx.rotate(o.rot||0);
  ctx.scale(sc, sc);

  // ---------- glow behind (additive) ----------
  const g = b*glow*haloK;
  ctx.globalCompositeOperation = 'lighter';
  if (g > .01){
    const pulse = 1 + .06*Math.sin(ts*2.3) + .03*Math.sin(ts*5.1);
    // big soft outer halo
    let Rh = R*(1.6 + 3.6*g)*pulse;
    let gr = ctx.createRadialGradient(0,0,R*.3,0,0,Rh);
    gr.addColorStop(0, U.rgba('#fff2a8', .55*g));
    gr.addColorStop(.25, U.rgba('#ffd27a', .28*g));
    gr.addColorStop(.6, U.rgba('#ffb347', .08*g));
    gr.addColorStop(1, 'rgba(255,140,60,0)');
    ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(0,0,Rh,0,TAU); ctx.fill();
    // soft rotating rays: two counter-rotating layers
    for (let layer=0; layer<2; layer++){
      const n = layer? 9 : 14, rot = (layer? -1:1)*ts*(layer? .13:.21) + layer*.3;
      const Lr = R*(layer? 3.4:2.6)*(.5+.7*g)*pulse;
      ctx.beginPath();
      for (let i=0;i<n;i++){
        const a = rot + i*TAU/n + .12*Math.sin(ts*.7+i*1.7);
        const len = Lr*(.65 + .35*U.hash(i*7.7+layer*31) + .2*Math.sin(ts*1.9+i*2.3));
        const hw = (layer? .07:.045) * (.7+.6*U.hash(i*3.1+layer));
        ctx.moveTo(Math.cos(a-hw)*R*.5, Math.sin(a-hw)*R*.5);
        ctx.lineTo(Math.cos(a)*len, Math.sin(a)*len);
        ctx.lineTo(Math.cos(a+hw)*R*.5, Math.sin(a+hw)*R*.5);
        ctx.closePath();
      }
      gr = ctx.createRadialGradient(0,0,R*.4,0,0,Lr*1.2);
      gr.addColorStop(0, U.rgba('#fff6c8', (layer? .22:.32)*g));
      gr.addColorStop(.5, U.rgba('#ffd27a', (layer? .08:.12)*g));
      gr.addColorStop(1, 'rgba(255,190,90,0)');
      ctx.fillStyle = gr; ctx.fill();
    }
    // tight inner glow
    gr = ctx.createRadialGradient(0,0,0,0,0,R*1.55);
    gr.addColorStop(0, U.rgba('#fffbe6', .5*g));
    gr.addColorStop(.6, U.rgba('#ffe27a', .2*g));
    gr.addColorStop(1, 'rgba(255,220,120,0)');
    ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(0,0,R*1.55,0,TAU); ctx.fill();
  }
  // faint cold aura when dim (so it still reads as a "star")
  const cold = (1-b)*glow*(.10 + .25*sput);
  if (cold > .01){
    const gr = ctx.createRadialGradient(0,0,R*.5,0,0,R*2);
    gr.addColorStop(0, sput>.05 ? U.rgba('#e8e2b8', cold) : U.rgba('#8f9ad0', cold));
    gr.addColorStop(1, 'rgba(120,130,200,0)');
    ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(0,0,R*2,0,TAU); ctx.fill();
  }
  ctx.globalCompositeOperation = 'source-over';

  // ---------- body transform: jelly squash & idle wobble ----------
  ctx.save();
  const breathe = Math.sin(ts*2.2);
  const jel = .028*Math.sin(ts*3.3) + .012*Math.sin(ts*7.1);
  const sx = (1 + .26*sq + jel) * (1 + .012*breathe);
  const sy = (1 - .3*sq - jel) * (1 + .018*breathe);
  const footY = R*.72;
  ctx.translate(0, footY); ctx.rotate((o.tilt||0) + .035*Math.sin(ts*1.1) * (o.expr==='joy'?2:1)); ctx.scale(sx, sy); ctx.translate(0, -footY);
  const bob = (o.expr==='joy' ? -Math.abs(Math.sin(ts*4.2))*3 : 0);
  ctx.translate(0, bob);

  // ---------- star shape (5 rounded mochi points) ----------
  const waveAmt = o.wave!=null ? o.wave : E.wave;
  const wv = Math.sin(ts*9.5);
  const tips = [];
  const dt = -Math.PI/2;
  const tuftSway = .07*Math.sin(ts*1.7) + .04*Math.sin(ts*3.9) + (E.tuftA>.2 ? .06*Math.sin(ts*.9):0);
  // [angle, length, width]
  tips.push([dt + E.tuftA + tuftSway, E.tuftL*(1+.02*Math.sin(ts*2.6)), .25]);
  const armSwing = .05*Math.sin(ts*2.2+.5);
  tips.push([dt + TAU/5 + E.armA + armSwing - waveAmt*(.35 + .32*wv), E.armL + waveAmt*.06, .22]); // right arm (screen right)
  tips.push([dt + 2*TAU/5 - .04 + .03*Math.sin(ts*1.3), .92, .24]);   // right foot
  tips.push([dt + 3*TAU/5 + .04 - .03*Math.sin(ts*1.3), .92, .24]);   // left foot
  tips.push([dt + 4*TAU/5 - E.armA - armSwing + waveAmt*.12*Math.sin(ts*9.5+1.2), E.armL, .22]); // left arm
  const dl = .42, kk = R*.3;
  function starPath(){
    ctx.beginPath();
    for (let i=0;i<5;i++){
      const [a, L, w0] = tips[i];
      const w = w0*R, D = L*R - w;
      const cx = Math.cos(a)*D, cy = Math.sin(a)*D;
      const s = a - Math.PI/2 - dl, e = a + Math.PI/2 + dl;
      const sxp = cx + Math.cos(s)*w, syp = cy + Math.sin(s)*w;
      if (i===0) ctx.moveTo(sxp, syp);
      else {
        // bezier from previous arc end, with tangents continuing smoothly
        const [pa, pL, pw0] = tips[i-1];
        const pw = pw0*R, pD = pL*R - pw, pcx = Math.cos(pa)*pD, pcy = Math.sin(pa)*pD, pe = pa + Math.PI/2 + dl;
        const ex = pcx + Math.cos(pe)*pw, ey = pcy + Math.sin(pe)*pw;
        ctx.bezierCurveTo(ex - Math.sin(pe)*kk, ey + Math.cos(pe)*kk, sxp + Math.sin(s)*kk, syp - Math.cos(s)*kk, sxp, syp);
      }
      ctx.arc(cx, cy, w, s, e, false);
    }
    // close: last tip -> first tip
    const [pa, pL, pw0] = tips[4];
    const pw = pw0*R, pD = pL*R - pw, pcx = Math.cos(pa)*pD, pcy = Math.sin(pa)*pD, pe = pa + Math.PI/2 + dl;
    const ex = pcx + Math.cos(pe)*pw, ey = pcy + Math.sin(pe)*pw;
    const [a, L, w0] = tips[0];
    const w = w0*R, D = L*R - w, s = a - Math.PI/2 - dl;
    const sxp = Math.cos(a)*D + Math.cos(s)*w, syp = Math.sin(a)*D + Math.sin(s)*w;
    ctx.bezierCurveTo(ex - Math.sin(pe)*kk, ey + Math.cos(pe)*kk, sxp + Math.sin(s)*kk, syp - Math.cos(s)*kk, sxp, syp);
    ctx.closePath();
  }

  // body fill: radial gradient whose warm zone grows from the core
  starPath();
  {
    const gr = ctx.createRadialGradient(0, R*.02, 0, 0, R*.02, R*1.02);
    for (let i=0;i<=6;i++){ const p=i/6; gr.addColorStop(p, bodyColor(p, b)); }
    ctx.fillStyle = gr; ctx.fill();
    ctx.lineWidth = 2.6; ctx.lineJoin = 'round';
    ctx.strokeStyle = U.mixHex(DIM.line, BRT.line, U.smooth(b));
    ctx.globalAlpha = .85 - .35*b; ctx.stroke(); ctx.globalAlpha = 1;
  }
  // volume shading inside the body (clipped)
  ctx.save();
  starPath(); ctx.clip();
  {
    // underside shade (cool when dim, peach when bright)
    let gr = ctx.createLinearGradient(0, -R*.2, 0, R);
    gr.addColorStop(0, 'rgba(0,0,0,0)');
    gr.addColorStop(1, b<.5 ? U.rgba('#3a3f7d', .38*(1-b*1.6)+.0) : U.rgba('#ff8a3d', .28*(b-.5)*2));
    ctx.fillStyle = gr; ctx.fillRect(-R*1.3, -R*1.3, R*2.6, R*2.6);
    // top-left glossy sheen
    gr = ctx.createRadialGradient(-R*.3, -R*.45, 0, -R*.3, -R*.45, R*.7);
    gr.addColorStop(0, `rgba(255,255,255,${.35 + .2*b})`);
    gr.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = gr; ctx.fillRect(-R*1.3, -R*1.3, R*2.6, R*2.6);
    // gloss highlight on the head tuft
    const [a0, L0, w0] = tips[0];
    const hx = Math.cos(a0)*(L0*R - w0*R) - w0*R*.35, hy = Math.sin(a0)*(L0*R - w0*R) - w0*R*.15;
    ctx.fillStyle = `rgba(255,255,255,${.45 + .3*b})`;
    ctx.beginPath(); ctx.ellipse(hx, hy, w0*R*.28, w0*R*.16, a0 + Math.PI/2 - .5, 0, TAU); ctx.fill();
    // inner core glow (light filling from the core outward)
    if (b > .02){
      ctx.globalCompositeOperation = 'lighter';
      const cr = R*(.35 + .75*b);
      gr = ctx.createRadialGradient(0, R*.05, 0, 0, R*.05, cr);
      const pul = .85 + .15*Math.sin(ts*3.1);
      gr.addColorStop(0, U.rgba('#fff3c4', .55*b*pul));
      gr.addColorStop(.5, U.rgba('#ffb347', .22*b*pul));
      gr.addColorStop(1, 'rgba(255,160,60,0)');
      ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(0, R*.05, cr, 0, TAU); ctx.fill();
      ctx.globalCompositeOperation = 'source-over';
    }
    // rim light (bright: inner warm rim)
    if (b > .3){
      ctx.lineWidth = 7; ctx.strokeStyle = U.rgba('#fff6d0', .35*(b-.3)/.7);
      ctx.globalCompositeOperation = 'lighter';
      starPath(); ctx.stroke();
      ctx.globalCompositeOperation = 'source-over';
    }
  }
  ctx.restore();

  // ---------- face ----------
  const fx = lookX*R*.07, fy = lookY*R*.05 + R*.02;
  const lidCol = bodyColor(.33, b);
  const lineCol = b>.5 ? '#5a2a1e' : '#2a2448';
  ctx.save();
  ctx.translate(fx, fy);
  // blush
  {
    const bl = E.blush * (.55 + .45*b);
    for (const sd of [-1,1]){
      const gr = ctx.createRadialGradient(sd*R*.45, R*.2, 0, sd*R*.45, R*.2, R*.17);
      gr.addColorStop(0, U.rgba(b>.4 ? '#ff7f8a' : '#d68aa8', .75*bl));
      gr.addColorStop(1, U.rgba(b>.4 ? '#ff7f8a' : '#d68aa8', 0));
      ctx.fillStyle = gr;
      ctx.beginPath(); ctx.ellipse(sd*R*.45, R*.2, R*.17, R*.11, 0, 0, TAU); ctx.fill();
      if (E.blush > .7){ // tiny blush hatch lines
        ctx.strokeStyle = U.rgba('#ff5f78', .5*bl); ctx.lineWidth = 1.3; ctx.lineCap='round';
        ctx.beginPath();
        for (let k=-1;k<=1;k++){ const bx = sd*R*.45 + k*R*.06; ctx.moveTo(bx+R*.025, R*.16); ctx.lineTo(bx-R*.025, R*.24); }
        ctx.stroke();
      }
    }
  }
  // eyes
  const big = E.big||1;
  const erx = R*.155*big, ery = R*.2*big;
  // auto blink
  let blink = 0;
  if (o.blink !== false && E.eye==='open'){
    const per = 3.4, slot = Math.floor((ts+.37)/per), ph = (ts+.37) - slot*per - U.hash(slot*9.13+seed)*1.6;
    if (ph>0 && ph<.16) blink = Math.sin(ph/.16*Math.PI);
    if (U.hash(slot*4.7+seed) < .3){ const ph2 = ph-.26; if (ph2>0 && ph2<.14) blink = Math.max(blink, Math.sin(ph2/.14*Math.PI)); }
  }
  for (const sd of [-1,1]){
    const ex = sd*R*.29, ey = -R*.04;
    drawEye(ctx, ex, ey, sd, erx, ery, E, b, blink, lookX, lookY, lidCol, lineCol, ts, R);
  }
  // tears (sad)
  const tearAmt = o.tears!=null ? o.tears : (E.tears||0);
  if (tearAmt > 0 && E.eye==='open'){
    for (const sd of [-1,1]){
      const ex = sd*R*.29, ey = -R*.04;
      // pooled tear along lower lid
      ctx.fillStyle = U.rgba('#cfe8ff', .55*tearAmt);
      ctx.beginPath(); ctx.ellipse(ex, ey + ery*.86, erx*.95, ery*.2, 0, 0, Math.PI); ctx.fill();
      ctx.fillStyle = U.rgba('#ffffff', .8*tearAmt);
      ctx.beginPath(); ctx.arc(ex + sd*erx*.4, ey + ery*.92, R*.025, 0, TAU); ctx.fill();
      // falling drop
      const per = 2.6, off = sd>0 ? 0 : 1.3, ph = ((ts+off) % per)/per;
      if (ph < .7){
        const k = ph/.7;
        const dx = ex + sd*erx*.75, dy = ey + ery*.95 + U.easeIn(k)*R*.45;
        const dr = R*(.03 + .025*U.clamp(k*3));
        ctx.globalAlpha = tearAmt * (1 - U.clamp((k-.75)/.25));
        ctx.fillStyle = '#bfe0ff';
        ctx.beginPath(); ctx.moveTo(dx, dy - dr*2.2);
        ctx.quadraticCurveTo(dx + dr*1.1, dy - dr*.2, dx, dy + dr);
        ctx.quadraticCurveTo(dx - dr*1.1, dy - dr*.2, dx, dy - dr*2.2); ctx.fill();
        ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(dx - dr*.3, dy - dr*.1, dr*.3, 0, TAU); ctx.fill();
        ctx.globalAlpha = 1;
      }
    }
  }
  // mouth
  drawMouth(ctx, 0, R*.24, o.mouth||'x', o.expr||'gentle', R, b, lineCol, ts);
  // hurt: little sweat drop
  if (o.expr==='hurt'){
    const sx2 = R*.55, sy2 = -R*.42 + Math.sin(ts*3)*1.5;
    ctx.fillStyle = 'rgba(200,230,255,.85)';
    ctx.beginPath(); ctx.moveTo(sx2, sy2 - R*.1);
    ctx.quadraticCurveTo(sx2 + R*.07, sy2 + R*.01, sx2, sy2 + R*.05);
    ctx.quadraticCurveTo(sx2 - R*.07, sy2 + R*.01, sx2, sy2 - R*.1); ctx.fill();
  }
  ctx.restore(); // face

  ctx.restore(); // body transform

  // ---------- front sparkles & sputter sparks (additive) ----------
  ctx.globalCompositeOperation = 'lighter';
  if (g > .05){
    const n = 7;
    for (let i=0;i<n;i++){
      const dir = i%2 ? 1 : -1;
      const a = ts*(.55 + .25*U.hash(i*5.3))*dir + i*TAU/n;
      const rr = R*(1.35 + .45*U.hash(i*2.9) + .15*Math.sin(ts*1.3+i)) * (.7 + .3*g);
      const px = Math.cos(a)*rr, py = Math.sin(a)*rr*.75 - R*.05;
      const tw = Math.pow(.5 + .5*Math.sin(ts*(3+U.hash(i)*3) + i*2.1), 2);
      const r = R*(.05 + .07*tw) * g;
      ctx.fillStyle = U.rgba('#fffbe6', .9*g);
      ctx.beginPath(); sparkle(ctx, px, py, r*1.8); ctx.fill();
      const gr = ctx.createRadialGradient(px,py,0,px,py,r*2.2);
      gr.addColorStop(0, U.rgba('#ffe9a0', .5*g*tw)); gr.addColorStop(1, 'rgba(255,220,120,0)');
      ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(px,py,r*2.2,0,TAU); ctx.fill();
    }
  }
  if (flick > .01){
    // weak sputtering sparks from the tips; each spark lives ~0.55 s
    const rate = 6, now = Math.floor(ts*rate);
    for (let j=0;j<5;j++){
      const slot = now - j;
      if (U.hash(slot*13.17 + 2.1 + seed) > .5*flick + .1) continue;
      const birth = slot/rate + U.hash(slot*3.3+.7)/rate, age = ts - birth, life = .45 + .25*U.hash(slot*1.9);
      if (age < 0 || age > life) continue;
      const k = age/life;
      const ti = Math.floor(U.hash(slot*7.9+.3)*5);
      const ta = -Math.PI/2 + ti*TAU/5 + (U.hash(slot*2.2)-.5)*.9;
      const d = R*(.85 + k*.7), px = Math.cos(ta)*d, py = Math.sin(ta)*d + k*k*R*.5;
      const a = (1-k) * (.6 + .4*Math.sin(age*60)) * flick * (1-b);
      const r = R*.04*(1-k*.5);
      ctx.fillStyle = U.rgba('#fff2b0', a);
      ctx.beginPath(); sparkle(ctx, px, py, r*2.4); ctx.fill();
      ctx.fillStyle = U.rgba('#ffd27a', a*.35);
      ctx.beginPath(); ctx.arc(px, py, r*3, 0, TAU); ctx.fill();
    }
  }
  ctx.globalCompositeOperation = 'source-over';
  ctx.restore();
}

function drawEye(ctx, ex, ey, sd, rx, ry, E, b, blink, lookX, lookY, lidCol, lineCol, ts, R){
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  if (E.eye === 'happy'){ // ^ ^
    ctx.strokeStyle = lineCol; ctx.lineWidth = R*.06;
    ctx.beginPath(); ctx.arc(ex, ey + ry*.45, rx*1.0, Math.PI + .45, -.45); ctx.stroke();
    return;
  }
  if (E.eye === 'squint'){ // > <
    ctx.strokeStyle = lineCol; ctx.lineWidth = R*.055;
    const w = rx*1.05, h = ry*.62, wob = Math.sin(ts*14)*R*.008;
    ctx.beginPath();
    ctx.moveTo(ex - sd*w, ey - h + wob);
    ctx.quadraticCurveTo(ex - sd*w*.1, ey - h*.35, ex + sd*w*.85, ey);
    ctx.quadraticCurveTo(ex - sd*w*.1, ey + h*.35, ex - sd*w, ey + h - wob);
    ctx.stroke();
    return;
  }
  const lid = Math.max(E.lid||0, blink);
  if (lid > .96){ // fully closed: soft curve
    ctx.strokeStyle = lineCol; ctx.lineWidth = R*.05;
    ctx.beginPath(); ctx.arc(ex, ey - ry*.15, rx, .5, Math.PI - .5); ctx.stroke();
    return;
  }
  ctx.save();
  ctx.beginPath(); ctx.ellipse(ex, ey, rx, ry, 0, 0, TAU);
  // glossy eye: deep plum-navy, glowing warmer at the bottom
  let gr = ctx.createLinearGradient(0, ey - ry, 0, ey + ry);
  gr.addColorStop(0, '#120c2a');
  gr.addColorStop(.55, b>.5 ? '#2c1430' : '#1e1a44');
  gr.addColorStop(1, U.mixHex('#3d4a9a', '#a3501c', b));
  ctx.fillStyle = gr; ctx.fill();
  ctx.clip();
  const lx = lookX*rx*.25, ly = lookY*ry*.2;
  // iris glow crescent
  const ic = U.mixHex('#7d93e6', '#ffb347', b);
  gr = ctx.createRadialGradient(ex+lx, ey+ly+ry*.55, 0, ex+lx, ey+ly+ry*.55, rx*1.0);
  gr.addColorStop(0, U.rgba(ic, .85)); gr.addColorStop(1, U.rgba(ic, 0));
  ctx.fillStyle = gr; ctx.fillRect(ex - rx, ey - ry, rx*2, ry*2);
  // pupil
  ctx.fillStyle = 'rgba(8,4,20,.75)';
  ctx.beginPath(); ctx.ellipse(ex+lx, ey+ly+ry*.05, rx*(E.small? .38:.5), ry*(E.small? .38:.52), 0, 0, TAU); ctx.fill();
  // catch-lights (shift slightly opposite to look for gloss)
  const tw = .9 + .1*Math.sin(ts*4 + sd);
  ctx.fillStyle = '#ffffff';
  ctx.beginPath(); ctx.ellipse(ex + lx*.4 - rx*.32, ey + ly*.4 - ry*.38, rx*.42*tw, ry*.34*tw, -.4, 0, TAU); ctx.fill();
  ctx.beginPath(); ctx.arc(ex + lx*.4 + rx*.38, ey + ly*.4 + ry*.36, rx*.17, 0, TAU); ctx.fill();
  if (E.sparkle || b > .8){ // twinkle star in the eye
    const s2 = (E.sparkle? 1 : .6) * (.75 + .25*Math.sin(ts*5 + sd*2));
    ctx.globalAlpha = .95;
    ctx.beginPath(); sparkle(ctx, ex + lx*.4 + rx*.3, ey + ly*.4 - ry*.05, rx*.42*s2); ctx.fill();
    ctx.globalAlpha = 1;
  }
  ctx.fillStyle = 'rgba(255,255,255,.35)';
  ctx.beginPath(); ctx.arc(ex - rx*.45, ey + ry*.5, rx*.09, 0, TAU); ctx.fill();
  // eyelid (sad / gentle / blink)
  let lidY = 0;
  if (lid > .01){
    const tilt = (E.lidTilt||0) * (1 - blink);   // + => inner corner raised (sad)
    lidY = ey - ry + 2*ry*lid;
    const yi = lidY - tilt*ry*.5, yo = lidY + tilt*ry*.5; // inner (toward centre) / outer
    const xi = ex - sd*rx*1.2, xo = ex + sd*rx*1.2;
    ctx.fillStyle = lidCol;
    ctx.beginPath();
    ctx.moveTo(xi, ey - ry*1.3); ctx.lineTo(xo, ey - ry*1.3);
    ctx.lineTo(xo, yo); ctx.quadraticCurveTo(ex, (yi+yo)/2 + ry*.18*(1-tilt*.5), xi, yi);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = lineCol; ctx.lineWidth = R*.035;
    ctx.beginPath(); ctx.moveTo(xo, yo); ctx.quadraticCurveTo(ex, (yi+yo)/2 + ry*.18*(1-tilt*.5), xi, yi); ctx.stroke();
  }
  // smiling lower lid (cheeks push up)
  if (E.lower){
    const ly0 = ey + ry - 2*ry*E.lower;
    ctx.fillStyle = lidCol;
    ctx.beginPath(); ctx.moveTo(ex - rx*1.3, ey + ry*1.3);
    ctx.lineTo(ex - rx*1.3, ly0 + ry*.35); ctx.quadraticCurveTo(ex, ly0 - ry*.45, ex + rx*1.3, ly0 + ry*.35);
    ctx.lineTo(ex + rx*1.3, ey + ry*1.3); ctx.closePath(); ctx.fill();
  }
  ctx.restore();
  // upper lash line + cute outer lash flick
  if (lid < .15){
    ctx.strokeStyle = lineCol; ctx.lineWidth = R*.038;
    ctx.beginPath(); ctx.ellipse(ex, ey, rx, ry, 0, Math.PI + .35, -.35); ctx.stroke();
    ctx.lineWidth = R*.03;
    const ax = ex + sd*rx*Math.cos(.35), ay = ey - ry*Math.sin(.35);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax + sd*R*.06, ay - R*.05); ctx.stroke();
  }
}

function drawMouth(ctx, mx, my, vis, expr, R, b, lineCol, ts){
  const inside = '#5a1e34', tongue = '#ff8a96';
  ctx.save();
  ctx.translate(mx, my);
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.strokeStyle = lineCol; ctx.lineWidth = R*.035;
  const s = R;
  const openShape = (w, h, top, flatTop) => {   // filled mouth: w=half width, h=depth below 0, top=height above 0
    ctx.beginPath();
    ctx.moveTo(-w, 0);
    if (flatTop) ctx.quadraticCurveTo(0, -top, w, 0); else ctx.bezierCurveTo(-w*.6, -top*1.35, w*.6, -top*1.35, w, 0);
    ctx.bezierCurveTo(w*.95, h*1.25, -w*.95, h*1.25, -w, 0);
    ctx.closePath();
    ctx.fillStyle = inside; ctx.fill();
    ctx.save(); ctx.clip();
    ctx.fillStyle = tongue; ctx.beginPath(); ctx.ellipse(0, h*.95, w*.7, h*.5, 0, 0, TAU); ctx.fill();
    ctx.restore();
    ctx.stroke();
  };
  if (vis === 'x'){
    // expression-default mouth
    if (expr === 'joy'){ openShape(s*.13, s*.15, s*.01, true); }
    else if (expr === 'surprised'){ ctx.beginPath(); ctx.ellipse(0, s*.03, s*.05, s*.065, 0, 0, TAU); ctx.fillStyle = inside; ctx.fill(); ctx.stroke(); }
    else if (expr === 'sad'){ ctx.beginPath(); ctx.moveTo(-s*.08, s*.04); ctx.quadraticCurveTo(0, -s*.04, s*.08, s*.04); ctx.stroke(); }
    else if (expr === 'hurt'){
      ctx.beginPath(); ctx.moveTo(-s*.11, s*.02);
      for (let i=1;i<=4;i++) ctx.lineTo(-s*.11 + i*s*.055, s*.02 + (i%2? -1:1)*s*.03);
      ctx.stroke();
    }
    else if (expr === 'hope'){ openShape(s*.07, s*.07, s*.0, true); }
    else { // gentle: soft cat-like ω smile
      ctx.beginPath();
      ctx.moveTo(-s*.11, -s*.01); ctx.quadraticCurveTo(-s*.055, s*.07, 0, s*.0);
      ctx.quadraticCurveTo(s*.055, s*.07, s*.11, -s*.01); ctx.stroke();
    }
  } else {
    switch (vis){
      case 'a': openShape(s*.1, s*.17, s*.02, false); break;              // wide open
      case 'i': // wide flat grin with teeth
        ctx.beginPath(); ctx.moveTo(-s*.13, 0); ctx.quadraticCurveTo(0, -s*.03, s*.13, 0);
        ctx.quadraticCurveTo(0, s*.07, -s*.13, 0); ctx.closePath();
        ctx.fillStyle = inside; ctx.fill();
        ctx.save(); ctx.clip(); ctx.fillStyle = '#fff'; ctx.fillRect(-s*.13, -s*.03, s*.26, s*.025); ctx.restore();
        ctx.stroke(); break;
      case 'u': // small pucker
        ctx.beginPath(); ctx.ellipse(0, s*.02, s*.035, s*.035, 0, 0, TAU); ctx.fillStyle = inside; ctx.fill(); ctx.stroke(); break;
      case 'e': openShape(s*.11, s*.08, s*.02, true); break;
      case 'o': ctx.beginPath(); ctx.ellipse(0, s*.04, s*.065, s*.09, 0, 0, TAU); ctx.fillStyle = inside; ctx.fill();
        ctx.save(); ctx.clip(); ctx.fillStyle = tongue; ctx.beginPath(); ctx.ellipse(0, s*.12, s*.05, s*.04, 0, 0, TAU); ctx.fill(); ctx.restore();
        ctx.stroke(); break;
      case 'n': ctx.beginPath(); ctx.moveTo(-s*.07, 0); ctx.quadraticCurveTo(0, s*.035, s*.07, 0); ctx.stroke(); break;
      case 'c': default: openShape(s*.06, s*.05, s*.005, true); break;
    }
  }
  ctx.restore();
}

window.Hoshi = { draw, EXPRS: Object.keys(EXPR), bodyColor };
})();
