// Mina — the lighthouse-keeper girl. window.Mina.draw(ctx, t, o)
// Pure function of t. Local frame: origin = point between feet, x = facing direction, y down, ~300 px tall at scale 1.
// Skeleton: pose params (numeric, blendable) -> joints (hip, chest, neck, head, shoulders, elbows, wrists, knees, ankles)
// -> limbs & clothes drawn around them. Cel shading = shape minus shape shifted toward the key light (moon, upper-left).
(function(){
const PI = Math.PI, sin = Math.sin, cos = Math.cos;
const C = {
  skin:{b:'#ffe0cf', s:'#f2b8a8', l:'#c97f78'},
  hair:{b:'#3a2430', s:'#2a1822', l:'#1c0f17', h:'#6b4256'},
  coat:{b:'#2b3a6b', s:'#1f2a52', l:'#121a3c', h:'#3c4f8a'},
  scarf:{b:'#e2474f', s:'#b5303d', l:'#7a1a28', h:'#ff7a78'},
  boot:{b:'#5a3a2e', s:'#432a20', l:'#26150f', h:'#7a5240'},
  tights:{b:'#5a3550', s:'#43263d', l:'#24121f'},
  button:'#e6c06e', mouth:'#7a2836', tongue:'#e8737c', lip:'#9c4048',
  eyeDark:'#2a1830', eyeMid:'#5e3560', eyeLow:'#a8708f', white:'#fffaf6', lidShade:'#e2d4ea', lash:'#24121c',
  cheek:'#ff9a9a'
};
const LW = 1.7;
const HS = .87;            // head scale           // outline width (local units)
// ---- light state (set per draw) ----
let LX = -0.6, LY = -0.8; // direction TOWARD key light in current drawing frame
let RIM = null;            // {x,y (dir toward light, current frame), col, k}

// ---------------- helpers ----------------
function part(c, fn, m, sh = 5, rimW = 3.2){
  c.save();
  c.beginPath(); fn(c); c.fillStyle = m.b; c.fill();
  c.clip();
  if (sh){
    c.beginPath(); c.rect(-3000,-3000,6000,6000);
    c.translate(LX*sh, LY*sh); fn(c); c.translate(-LX*sh, -LY*sh);
    c.fillStyle = m.s; c.fill('evenodd');
  }
  if (RIM && rimW){
    c.beginPath(); c.rect(-3000,-3000,6000,6000);
    const dx = -RIM.x*rimW, dy = -RIM.y*rimW;
    c.translate(dx, dy); fn(c); c.translate(-dx, -dy);
    c.fillStyle = RIM.col; c.globalAlpha = RIM.k*.85; c.fill('evenodd');
  }
  c.restore();
  c.beginPath(); fn(c); c.strokeStyle = m.l; c.lineWidth = LW; c.stroke();
}
function poly(c, pts){ c.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) c.lineTo(pts[i][0], pts[i][1]); }
// tapered limb drawn as round-capped strokes: outline pass, shade pass, lit pass, rim pass
function limb(c, pts, ws, m){
  c.lineCap = 'round'; c.lineJoin = 'round';
  const seg = (dx, dy, wk, wa) => { for (let i = 0; i < pts.length-1; i++){ c.beginPath(); c.moveTo(pts[i][0]+dx, pts[i][1]+dy); c.lineTo(pts[i+1][0]+dx, pts[i+1][1]+dy); c.lineWidth = ws[i]*wk + wa; c.stroke(); } };
  c.strokeStyle = m.l; seg(0, 0, 1, LW*2);
  c.strokeStyle = m.s; seg(0, 0, 1, 0);
  // lit side: per segment, shift a narrower base-colour stroke along the normal toward the key light
  c.strokeStyle = m.b;
  for (let i = 0; i < pts.length-1; i++){
    const a = pts[i], b = pts[i+1], dl = Math.hypot(b[0]-a[0], b[1]-a[1]) || 1;
    let nx = -(b[1]-a[1])/dl, ny = (b[0]-a[0])/dl; const d = nx*LX + ny*LY; if (d < 0){ nx = -nx; ny = -ny; }
    const k = .1 + .12*Math.abs(d), w = ws[i];
    c.beginPath(); c.moveTo(a[0] + nx*w*k, a[1] + ny*w*k); c.lineTo(b[0] + nx*w*k, b[1] + ny*w*k); c.lineWidth = w*(1 - 2*k); c.stroke();
  }
  if (RIM){
    c.save(); c.globalAlpha = RIM.k*.85; c.strokeStyle = RIM.col; c.lineCap = 'butt';
    for (let i = 0; i < pts.length-1; i++){
      const a = pts[i], b = pts[i+1], dl = Math.hypot(b[0]-a[0], b[1]-a[1]) || 1;
      const nx = -(b[1]-a[1])/dl, ny = (b[0]-a[0])/dl, d = nx*RIM.x + ny*RIM.y;
      const sg = d < 0 ? -1 : 1, ad = Math.abs(d); if (ad < .15) continue;
      const w = ws[i], off = w/2 - .7, tx = (b[0]-a[0])/dl, ty = (b[1]-a[1])/dl, sh = Math.min(w*.3, dl*.2);
      c.globalAlpha = RIM.k*.85*Math.min(1, (ad - .15)*2.5);
      c.lineWidth = 2.4; c.beginPath();
      c.moveTo(a[0] + sg*nx*off + tx*sh, a[1] + sg*ny*off + ty*sh); c.lineTo(b[0] + sg*nx*off - tx*sh, b[1] + sg*ny*off - ty*sh); c.stroke();
    }
    c.restore();
  }
}
const lerpA = (a, b, k) => a + (b-a)*k;

// ---------------- poses ----------------
// angles: from straight down, + toward facing direction. Arms are torso-relative, legs are local.
// a?1 = upper arm, a?2 = elbow bend (forward). l?1 = thigh, l?2 = knee bend (backward).
const BASE = {hipX:0, hipY:0, lean:0, head:0, aN1:-.02, aN2:.22, aF1:-.08, aF2:.3, fo:0, lN1:.03, lN2:.02, lF1:-.05, lF2:.02,
  air:0, eyeY:0, eyeX:0, run:0, shrug:0, sway:0, hN:'relax', hF:'relax'};
const POSES = {
  stand(t){ return {}; },
  run(t, pt, o){
    const p = t*2*PI*(o.runRate||1.6);
    const s = sin(p), cc = cos(p);
    const kn = ph => .35 + 1.25*Math.pow(Math.max(0, cos(ph+.5)), 1.4);
    return {lean:.24, head:-.16, run:1,
      lN1:.12 + .72*s, lN2:kn(p), lF1:.12 - .72*s, lF2:kn(p+PI),
      aN1:-.95*s + .05, aN2:1.35 + .35*Math.max(0, s), aF1:.95*s + .05, aF2:1.35 + .35*Math.max(0, -s),
      air:5 + 7*Math.abs(cos(p)), hipY:-2, sway:.05*cc, hN:'fist', hF:'fist'};
  },
  surprised(t, pt){
    const j = pt == null ? 0 : sin(PI*U.clamp(pt*2.2));
    return {lean:-.13, head:-.08, aN1:.55, aN2:2.05, aF1:.75, aF2:1.85, lN1:.18 + .2*j, lN2:.2 + .5*j, lF1:-.08, lF2:.1 + .3*j,
      air:16*j, shrug:3, hN:'open', hF:'open', eyeY:-.1};
  },
  crouch(t){
    return {hipY:50, lean:.3, head:.12, hipX:-6,
      lN1:1.45, lN2:1.55, lF1:.35, lF2:1.95,
      aN1:.65, aN2:.9, aF1:.95, aF2:.55, eyeY:.7, eyeX:.3, hN:'relax', hF:'open'};
  },
  reach(t){
    const b = sin(t*2.1)*.04;
    return {lean:.16, head:.02, aN1:1.5 + b, aN2:.08, aF1:-.35, aF2:.45, lN1:.32, lN2:.18, lF1:-.25, lF2:.05,
      eyeY:.2, eyeX:.5, hN:'open', hF:'relax'};
  },
  point(t){
    return {lean:-.04, head:-.2, aF1:2.45, aF2:.06, aN1:-.18, aN2:.55, lN1:.26, lN2:.03, lF1:-.2, lF2:.02, fo:1,
      eyeY:-.6, eyeX:.6, hF:'point', hN:'fist'};
  },
  wave(t){
    const w = t*2*PI*1.7;
    return {lean:-.04 + .03*sin(w*.5), head:-.12 + .04*sin(w*.5), aF1:2.5 + .1*sin(w - .6), aF2:.4 + .5*sin(w), fo:1,
      aN1:-.05, aN2:.25, lN1:.06, lN2:.02, lF1:-.05, lF2:.02, sway:.02*sin(w*.5), eyeY:-.3, hF:'open', hN:'relax'};
  },
  clasp(t){
    return {lean:.04, head:-.1, aN1:.35, aN2:1.9, aF1:-.3, aF2:2.4, lN1:.0, lN2:.06, lF1:.02, lF2:.08, shrug:1.5, fo:1,
      eyeY:-.2, hN:'fist', hF:'fist'};
  },
  lookup(t){
    return {lean:-.12, head:-.42, aN1:-.08, aN2:.15, aF1:-.18, aF2:.2, lN1:.02, lN2:.02, lF1:-.06, lF2:.02, eyeY:-.85, eyeX:.2};
  }
};
function poseParams(name, t, pt, o){
  const f = POSES[name] || POSES.stand;
  return Object.assign({}, BASE, f(t, pt, o));
}

// ---------------- expressions ----------------
// bY brow raise, bA inner-brow tilt (+ = worried), lid upper-lid drop, lb lower lid raise, es eye scale, ir iris scale,
// sm mouth smile, rest = closed-mouth style, hap = ^^ eyes, bl = blush, gl = extra eye gloss
const EXPR = {
  neutral:   {bY:0,  bA:0,    lid:.06, lb:0,   es:1,    ir:1,   sm:.3,  rest:'line',  hap:0, bl:.55, gl:0},
  happy:     {bY:1.5,bA:.05,  lid:.04, lb:.32, es:1,    ir:1,   sm:.9,  rest:'smile', hap:0, bl:.8,  gl:.2},
  surprised: {bY:5,  bA:.15,  lid:0,   lb:0,   es:1.1,  ir:.74, sm:0,   rest:'o',     hap:0, bl:.6,  gl:0},
  worried:   {bY:2.5,bA:.6,   lid:.18, lb:.05, es:1,    ir:.95, sm:-.45,rest:'wobble',hap:0, bl:.6,  gl:.4},
  determined:{bY:-2, bA:-.5,  lid:.26, lb:.08, es:1,    ir:.95, sm:.4,  rest:'line',  hap:0, bl:.6,  gl:.1},
  wistful:   {bY:1,  bA:.38,  lid:.36, lb:.14, es:1,    ir:1,   sm:.15, rest:'line',  hap:0, bl:.7,  gl:.9},
  joy:       {bY:3,  bA:.08,  lid:0,   lb:0,   es:1,    ir:1,   sm:1,   rest:'grin',  hap:1, bl:1,   gl:0}
};
function exprParams(o){
  const a = EXPR[o.expr] || EXPR.neutral;
  if (!o.expr2 || !EXPR[o.expr2] || !o.exprMix) return a;
  const b = EXPR[o.expr2], k = U.clamp(o.exprMix), r = {};
  for (const key in a) r[key] = typeof a[key] === 'number' ? lerpA(a[key], b[key], k) : (k < .5 ? a[key] : b[key]);
  return r;
}

// deterministic auto blink 0..1 (1 = closed)
function autoBlink(t, seed){
  const P = 3.3, k = Math.floor(t/P); let b = 0;
  for (let j = k-1; j <= k; j++){
    const bt = j*P + .3 + U.hash(j*7.13 + seed)*2.2;
    let d = t - bt; if (d >= 0 && d < .17) b = Math.max(b, sin(PI*d/.17));
    if (U.hash(j*3.71 + seed) > .72){ d -= .26; if (d >= 0 && d < .15) b = Math.max(b, sin(PI*d/.15)); }
  }
  return Math.min(1, b*1.25);
}

// ---------------- skeleton ----------------
const L_UA = 38, L_FA = 35, L_TH = 52, L_SH = 50, SOLE = 9;
function skeleton(P, t){
  const br = sin(t*2*PI/3.4);                 // breathing
  const lean = P.lean + .012*U.noise(t*.6, 3) + P.sway;
  const H = [P.hipX, -111 + P.hipY];
  const right = [cos(lean), sin(lean)], up = [sin(lean), -cos(lean)];
  const T = (u, v) => [H[0] + right[0]*u + up[0]*v, H[1] + right[1]*u + up[1]*v];
  const dirT = a => [sin(a - lean), cos(a - lean)];
  const chest = .8*br;
  const sv = 60 + chest + P.shrug;
  const S = {N: T(-16, sv), F: T(11, sv)};
  const arm = (Sh, a1, a2) => { const d1 = dirT(a1), E = [Sh[0] + d1[0]*L_UA, Sh[1] + d1[1]*L_UA];
    const d2 = dirT(a1 + a2), W = [E[0] + d2[0]*L_FA, E[1] + d2[1]*L_FA]; return {S:Sh, E, W, ang:Math.atan2(d2[1], d2[0])}; };
  const leg = (hx, a1, a2) => { const Hp = [H[0] + hx, H[1]]; const K = [Hp[0] + sin(a1)*L_TH, Hp[1] + cos(a1)*L_TH];
    const ph = a1 - a2, A = [K[0] + sin(ph)*L_SH, K[1] + cos(ph)*L_SH]; return {H:Hp, K, A, ph}; };
  const sk = {lean, H, T, right, up, br,
    armN: arm(S.N, P.aN1, P.aN2), armF: arm(S.F, P.aF1, P.aF2),
    legN: leg(-6, P.lN1, P.lN2), legF: leg(6, P.lF1, P.lF2)};
  // grounding: lowest sole / knee point touches y=0, then lift by air
  let maxY = -1e9;
  for (const lg of [sk.legN, sk.legF]){
    const r = -lg.ph, cr = cos(r), sr = sin(r);
    for (const [x, y] of [[-7, SOLE], [14, SOLE]]) maxY = Math.max(maxY, lg.A[1] + x*sr + y*cr);
    maxY = Math.max(maxY, lg.K[1] + 7);
  }
  sk.dy = -maxY - P.air;
  sk.N = T(0, 72 + chest + P.shrug*.6);
  return sk;
}

// ---------------- parts ----------------
function scarfTails(c, sk, t, P, o){
  const wind = o.wind == null ? 1 : o.wind;
  const run = P.run;
  const K = sk.T(-15, 66 + .8*sk.br);
  const tails = [{n:10, len:12.5, w:14, ph:0, off:.0, col:C.scarf},
                 {n:7, len:12, w:12.5, ph:1.7, off:.18, col:{b:C.scarf.s, s:'#962534', l:C.scarf.l}}];
  for (let k = tails.length-1; k >= 0; k--){
    const tl = tails[k];
    const lift = .35 + .42*wind + .75*run + .08*U.noise(t*.7, 11 + k);
    const om = 4.2 + 3.5*run + .8*wind;
    let x = K[0] + (k ? 4 : 0), y = K[1] + (k ? 2 : 0);
    const pts = [[x, y]], ws = [];
    for (let i = 0; i < tl.n; i++){
      const f = i/(tl.n-1);
      let a = -.12 - tl.off - sk.lean*.6 - lift*(.35 + .75*f)
        + (.18 + .3*wind + .2*run)*Math.pow(f, .7)*sin(om*t - i*.9 + tl.ph)
        + .18*f*U.noise(t*1.4 + i*.25, 5 + k);
      a = Math.max(a, -1.95);
      x += sin(a)*tl.len; y += cos(a)*tl.len;
      pts.push([x, y]);
    }
    for (let i = 0; i < pts.length; i++){
      const f = i/(pts.length-1);
      ws.push(tl.w*(.9 + .1*f)*(.62 + .38*Math.abs(cos(om*.5*t - i*.45 + tl.ph))));
    }
    // ribbon outline from centreline
    const L = [], R = [];
    for (let i = 0; i < pts.length; i++){
      const a = pts[Math.max(0, i-1)], b = pts[Math.min(pts.length-1, i+1)];
      let nx = -(b[1] - a[1]), ny = b[0] - a[0]; const d = Math.hypot(nx, ny) || 1; nx /= d; ny /= d;
      const hw = ws[i]/2; L.push([pts[i][0] + nx*hw, pts[i][1] + ny*hw]); R.push([pts[i][0] - nx*hw, pts[i][1] - ny*hw]);
    }
    const fn = cc => {
      cc.moveTo(L[0][0], L[0][1]);
      for (let i = 1; i < L.length-1; i++) cc.quadraticCurveTo(L[i][0], L[i][1], (L[i][0] + L[i+1][0])/2, (L[i][1] + L[i+1][1])/2);
      cc.lineTo(L[L.length-1][0], L[L.length-1][1]);
      cc.lineTo(R[R.length-1][0], R[R.length-1][1]);
      for (let i = R.length-2; i > 0; i--) cc.quadraticCurveTo(R[i][0], R[i][1], (R[i][0] + R[i-1][0])/2, (R[i][1] + R[i-1][1])/2);
      cc.lineTo(R[0][0], R[0][1]); cc.closePath();
    };
    part(c, fn, tl.col, 3, 2.5);
    // end stripe + fringe
    const e = pts.length-1, A = pts[e-1], B = pts[e];
    const dx = B[0] - A[0], dy = B[1] - A[1], dl = Math.hypot(dx, dy) || 1, ux = dx/dl, uy = dy/dl;
    c.strokeStyle = k ? '#e9d8c8' : '#fff1e2'; c.globalAlpha = .85; c.lineWidth = 2.2;
    const m1 = [B[0] - ux*7, B[1] - uy*7], hw = ws[e]/2 - 1;
    c.beginPath(); c.moveTo(m1[0] - uy*hw, m1[1] + ux*hw); c.lineTo(m1[0] + uy*hw, m1[1] - ux*hw); c.stroke();
    c.globalAlpha = 1;
    c.strokeStyle = tl.col.b; c.lineWidth = 2; c.lineCap = 'round';
    for (let j = 0; j < 4; j++){
      const q = (j/3 - .5)*(ws[e] - 3), fl = 5 + 1.5*sin(t*9 + j*1.7 + k);
      const sx = B[0] - uy*q, sy = B[1] + ux*q;
      c.beginPath(); c.moveTo(sx, sy); c.lineTo(sx + ux*fl + uy*1.2*sin(t*7 + j), sy + uy*fl - ux*1.2*sin(t*7 + j)); c.stroke();
    }
  }
}

function drawLeg(c, lg, t){
  limb(c, [lg.H, lg.K, lg.A], [19, 15], C.tights);
  c.save(); c.translate(lg.A[0], lg.A[1]); c.rotate(-lg.ph);
  part(c, cc => {
    cc.moveTo(-6.5, -17); cc.lineTo(-7.5, 4); cc.quadraticCurveTo(-8, SOLE, -3.5, SOLE); cc.lineTo(12, SOLE);
    cc.quadraticCurveTo(17.5, SOLE, 16, 3.5); cc.quadraticCurveTo(14, -.5, 6.5, -2); cc.lineTo(6.5, -17); cc.closePath();
  }, C.boot, 2.5, 2);
  // sole + cuff
  c.fillStyle = '#2e1c16'; c.beginPath(); c.moveTo(-8, SOLE-2); c.lineTo(15, SOLE-2); c.quadraticCurveTo(16.5, SOLE, 12, SOLE+.5); c.lineTo(-4, SOLE+.5); c.quadraticCurveTo(-8, SOLE+.5, -8, SOLE-2); c.fill();
  part(c, cc => { cc.moveTo(-8.5, -19); cc.lineTo(8.5, -19); cc.lineTo(8, -12.5); cc.quadraticCurveTo(0, -11, -8, -12.5); cc.closePath(); }, {b:C.boot.h, s:C.boot.b, l:C.boot.l}, 1.5, 1.5);
  c.restore();
}

function drawHand(c, W, ang, kind, far){
  const m = far ? {b:C.skin.s, s:'#e3a497', l:C.skin.l} : C.skin;
  c.save(); c.translate(W[0], W[1]); c.rotate(ang);
  c.lineCap = 'round';
  const finger = (x0, y0, a, len) => {
    const x1 = x0 + cos(a)*len, y1 = y0 + sin(a)*len;
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1);
    c.strokeStyle = m.l; c.lineWidth = 3.3 + LW*1.6; c.stroke(); c.strokeStyle = m.b; c.lineWidth = 3.3; c.stroke();
  };
  if (kind === 'open'){
    finger(5, -3.5, -1.1, 4.8);                    // thumb
    for (let i = 0; i < 4; i++) finger(7.5, -2.6 + i*1.75, -.3 + i*.2, 6 - Math.abs(i - 1.2)*.8);
    part(c, cc => cc.ellipse(5, 0, 5.6, 5.3, 0, 0, 2*PI), m, 1.5, 1.5);
  } else if (kind === 'point'){
    part(c, cc => cc.ellipse(5.5, .5, 6.3, 5.8, 0, 0, 2*PI), m, 1.5, 1.5);
    finger(9, -2.5, -.05, 9);
    c.strokeStyle = m.l; c.lineWidth = 1; c.beginPath(); c.moveTo(9, 1); c.lineTo(11, 1.5); c.moveTo(8.5, 3.5); c.lineTo(10.5, 4); c.stroke();
  } else if (kind === 'fist'){
    part(c, cc => cc.ellipse(5.5, 0, 6.6, 6, 0, 0, 2*PI), m, 1.5, 1.5);
    c.strokeStyle = m.l; c.lineWidth = 1; c.beginPath();
    c.moveTo(9, -2.5); c.lineTo(11.3, -2); c.moveTo(9.3, .8); c.lineTo(11.8, 1.2); c.moveTo(8.8, 3.8); c.lineTo(10.9, 4.3); c.stroke();
  } else { // relaxed mitten
    part(c, cc => { cc.ellipse(6, .5, 7, 5.6, .1, 0, 2*PI); }, m, 1.5, 1.5);
    part(c, cc => { cc.ellipse(5, -4.2, 3.6, 2.3, -.5, 0, 2*PI); }, m, 0, 0);
  }
  c.restore();
}

function drawArm(c, a, hand, far){
  const m = far ? {b:C.coat.s, s:'#18214a', l:C.coat.l} : C.coat;
  drawHand(c, a.W, a.ang, hand, far);
  const d = [a.W[0] - a.E[0], a.W[1] - a.E[1]], dl = Math.hypot(d[0], d[1]) || 1;
  const Wc = [a.W[0] - d[0]/dl*2, a.W[1] - d[1]/dl*2];
  limb(c, [a.S, a.E, Wc], [16.5, 14.5], m);
  // cuff line
  const nx = -d[1]/dl, ny = d[0]/dl, cx = Wc[0] - d[0]/dl*4, cy = Wc[1] - d[1]/dl*4;
  c.strokeStyle = m.l; c.lineWidth = 1.2; c.beginPath(); c.moveTo(cx + nx*6.5, cy + ny*6.5); c.lineTo(cx - nx*6.5, cy - ny*6.5); c.stroke();
}

function drawCoat(c, sk, t, P){
  const T = sk.T, run = P.run, br = .8*sk.br;
  const fl = run*(2.5*sin(t*2*PI*1.6*2)), bk = run*4;
  const sv = 63 + br + P.shrug;
  const pts = {
    nf: T(8, 71 + br), sf: T(19, sv + 1), af: T(21.5, sv - 10), wf: T(18.5, 14), hf: T(27 + fl, -27),
    hb: T(-27 - bk - fl, -25 + bk*.8), wb: T(-19, 14), ab: T(-21.5, sv - 10), sb: T(-18, sv + 2), nb: T(-8, 71 + br)
  };
  const fn = cc => {
    cc.moveTo(pts.nb[0], pts.nb[1]);
    cc.lineTo(pts.nf[0], pts.nf[1]);
    cc.quadraticCurveTo(pts.sf[0], pts.sf[1], pts.af[0], pts.af[1]);
    cc.quadraticCurveTo(pts.wf[0], pts.wf[1] - 12, pts.wf[0], pts.wf[1]);
    cc.lineTo(pts.hf[0], pts.hf[1]);
    const mid = T(0, -31 + (fl - bk)*.2);
    cc.quadraticCurveTo(mid[0], mid[1], pts.hb[0], pts.hb[1]);
    cc.lineTo(pts.wb[0], pts.wb[1]);
    cc.quadraticCurveTo(pts.wb[0], pts.wb[1] - 0, pts.ab[0], pts.ab[1]);
    cc.quadraticCurveTo(pts.sb[0], pts.sb[1], pts.nb[0], pts.nb[1]);
    cc.closePath();
  };
  part(c, fn, C.coat, 6, 3.5);
  // details: front overlap edge, buttons, pocket, hem band
  c.save(); c.beginPath(); fn(c); c.clip();
  c.strokeStyle = C.coat.l; c.lineWidth = 1.3; c.lineCap = 'round';
  let p1 = T(17, 50), p2 = T(19.5 + fl*.9, -27);
  c.beginPath(); c.moveTo(p1[0], p1[1]); c.quadraticCurveTo(T(17, 10)[0], T(17, 10)[1], p2[0], p2[1]); c.stroke();
  // lapel
  const l0 = T(-4, 68), l1 = T(10, 44), l2 = T(19, 64);
  c.beginPath(); c.moveTo(l0[0], l0[1]); c.lineTo(l1[0], l1[1]); c.lineTo(l2[0], l2[1]); c.stroke();
  // hem band
  const h0 = T(-26 - bk, -18), h1 = T(26 + fl, -20), hm = T(0, -24);
  c.globalAlpha = .7; c.beginPath(); c.moveTo(h0[0], h0[1]); c.quadraticCurveTo(hm[0], hm[1], h1[0], h1[1]); c.stroke();
  // pocket flap
  const q0 = T(-16, 4), q1 = T(-3, 4.5), q2 = T(-4, -1), q3 = T(-15, -1.5);
  c.globalAlpha = 1; c.fillStyle = C.coat.s; c.beginPath(); poly(c, [q0, q1, q2, q3]); c.closePath(); c.fill(); c.stroke();
  // buttons (double-breasted)
  for (const [u, v] of [[4, 36], [14, 37], [4, 22], [14, 23], [4.5, 8], [14.5, 9]]){
    const b = T(u, v + br*.5);
    c.fillStyle = C.coat.l; c.beginPath(); c.arc(b[0] + .5, b[1] + .6, 2.6, 0, 2*PI); c.fill();
    c.fillStyle = C.button; c.beginPath(); c.arc(b[0], b[1], 2.3, 0, 2*PI); c.fill();
    c.fillStyle = '#fff3c4'; c.beginPath(); c.arc(b[0] - .7, b[1] - .8, .8, 0, 2*PI); c.fill();
  }
  c.restore();
}

function scarfWrap(c, sk, t){
  const T = sk.T, b = .8*sk.br;
  const P = (u, v) => T(u, v + b);
  const fn = cc => {
    let p = P(-19, 75); cc.moveTo(p[0], p[1]);
    let q1 = P(-8, 84), q2 = P(12, 84), q3 = P(20, 77); cc.bezierCurveTo(q1[0], q1[1], q2[0], q2[1], q3[0], q3[1]);
    q1 = P(25, 70); q2 = P(22, 60); q3 = P(10, 59); cc.bezierCurveTo(q1[0], q1[1], q2[0], q2[1], q3[0], q3[1]);
    q1 = P(-2, 57); q2 = P(-15, 58); q3 = P(-21, 64); cc.bezierCurveTo(q1[0], q1[1], q2[0], q2[1], q3[0], q3[1]);
    q1 = P(-24, 69); cc.quadraticCurveTo(q1[0], q1[1], p[0], p[1]); cc.closePath();
  };
  part(c, fn, C.scarf, 4, 3);
  // fold lines
  c.strokeStyle = C.scarf.s; c.lineWidth = 1.6; c.lineCap = 'round';
  let a = P(-16, 69), m = P(2, 74), e = P(19, 70);
  c.beginPath(); c.moveTo(a[0], a[1]); c.quadraticCurveTo(m[0], m[1], e[0], e[1]); c.stroke();
  a = P(-12, 62); m = P(4, 64); e = P(16, 62);
  c.globalAlpha = .7; c.beginPath(); c.moveTo(a[0], a[1]); c.quadraticCurveTo(m[0], m[1], e[0], e[1]); c.stroke(); c.globalAlpha = 1;
  // knot at the back side where the tails come out
  const k = P(-17, 66);
  part(c, cc => cc.ellipse(k[0], k[1], 6.5, 7.5, sk.lean - .3, 0, 2*PI), C.scarf, 2.5, 2);
}

// ---------------- head ----------------
function drawEye(c, cx, cy, hw, hh, E, blink, lx, ly, outer, t, far){
  const open = U.clamp((1 - blink)*(1 - E.lid*.85));
  const hap = E.hap;
  const yb = cy + hh - E.lb*hh*.55;
  const yc = cy + hh*.25;                         // corner height
  const xo = cx + outer*hw, xi = cx - outer*hw;   // outer / inner corner x
  const ytop = yb - (2*hh)*E.es*Math.max(open, 0) ;
  if (open > .14 && hap < .5){
    const lidFn = cc => {
      cc.moveTo(xi, yc + 1);
      cc.bezierCurveTo(xi + outer*hw*.05, ytop - 1, xo - outer*hw*.25, ytop - 2, xo, yc - hh*.25 + (1 - open)*hh*.6);
      cc.bezierCurveTo(xo - outer*hw*.05, yb + 1, xi + outer*hw*.4, yb + 2, xi, yc + 1);
    };
    c.save();
    c.beginPath(); lidFn(c); c.fillStyle = C.white; c.fill(); c.clip();
    // lid shadow
    c.fillStyle = C.lidShade; c.beginPath(); c.ellipse(cx, ytop - hh*.15, hw*1.4, hh*.55, 0, 0, 2*PI); c.fill();
    // iris
    const ix = cx + lx*hw*.32, iy = cy + 1.5 + ly*hh*.22;
    const rx = hw*.8*E.ir, ry = hh*.86*E.ir;
    const g = c.createLinearGradient(0, iy - ry, 0, iy + ry);
    g.addColorStop(0, C.eyeDark); g.addColorStop(.5, C.eyeMid); g.addColorStop(1, C.eyeLow);
    c.fillStyle = g; c.beginPath(); c.ellipse(ix, iy, rx, ry, 0, 0, 2*PI); c.fill();
    c.strokeStyle = C.eyeDark; c.lineWidth = 1.1; c.stroke();
    // lower iris glow
    c.fillStyle = 'rgba(255,200,225,0.35)'; c.beginPath(); c.ellipse(ix, iy + ry*.5, rx*.6, ry*.3, 0, 0, 2*PI); c.fill();
    // pupil
    c.fillStyle = '#12070f'; c.beginPath(); c.ellipse(ix, iy - ry*.05, rx*.45, ry*.5, 0, 0, 2*PI); c.fill();
    // catch-lights
    const gl = 1 + .25*E.gl*sin(t*3);
    c.fillStyle = '#ffffff';
    c.beginPath(); c.ellipse(ix - rx*.32, iy - ry*.38, rx*.36*gl, ry*.27*gl, -.4, 0, 2*PI); c.fill();
    c.beginPath(); c.ellipse(ix + rx*.35, iy + ry*.38, rx*.15, ry*.12, 0, 0, 2*PI); c.fill();
    if (E.gl > .3){ c.globalAlpha = .55*E.gl; c.beginPath(); c.ellipse(ix + rx*.1, iy + ry*.62, rx*.5, ry*.12, 0, 0, 2*PI); c.fill(); c.globalAlpha = 1; }
    c.restore();
    // upper lash line (thick, with wing at outer corner)
    c.fillStyle = C.lash; c.beginPath();
    c.moveTo(xi - outer*.5, yc + 1.5);
    c.bezierCurveTo(xi + outer*hw*.05, ytop - 1.5, xo - outer*hw*.25, ytop - 2.5, xo + outer*2.8, yc - hh*.45 + (1 - open)*hh*.6);
    c.lineTo(xo + outer*1.2, yc - hh*.15 + (1 - open)*hh*.6);
    c.bezierCurveTo(xo - outer*hw*.25, ytop + 1.6, xi + outer*hw*.1, ytop + 1.8, xi + outer*.6, yc + 2);
    c.closePath(); c.fill();
    // lashes flick
    c.strokeStyle = C.lash; c.lineWidth = 1.3; c.lineCap = 'round';
    const lyy = yc - hh*.35 + (1 - open)*hh*.6;
    c.beginPath(); c.moveTo(xo, lyy); c.lineTo(xo + outer*4, lyy - 2.5); c.moveTo(xo - outer*2, lyy - 2.4); c.lineTo(xo + outer*1.5, lyy - 5.5); c.stroke();
    // lower lash
    c.strokeStyle = C.skin.l; c.lineWidth = 1.1;
    c.beginPath(); c.moveTo(cx + outer*hw*.05, yb + 1.2); c.quadraticCurveTo(xo - outer*hw*.3, yb + 1, xo - outer*.3, yc + hh*.15); c.stroke();
  } else {
    // closed eye: blink = gentle downward curve, joy = ^ arc
    const k = Math.max(hap, 0);
    const yy = yc + (1 - k)*hh*.15;
    const ctrl = yy + (1 - k)*hh*.55 - k*hh*.95;
    c.strokeStyle = C.lash; c.lineWidth = 2.6; c.lineCap = 'round';
    c.beginPath(); c.moveTo(xi, yy + k*2); c.quadraticCurveTo(cx, ctrl, xo + outer*1.5, yy - (1 - k)*1.5 + k*2); c.stroke();
    if (k < .5){ c.lineWidth = 1.2; c.beginPath(); c.moveTo(xo - outer*1, yy + 1); c.lineTo(xo + outer*3, yy + 3); c.stroke(); }
  }
}

function mouthShape(v, E){
  // w width, h open height, rd roundness, teeth
  switch (v){
    case 'a': return {w:11.5, h:10, rd:.15, te:.25};
    case 'i': return {w:12.5, h:4.2, rd:0, te:1};
    case 'u': return {w:5.5, h:5.2, rd:1, te:0};
    case 'e': return {w:11.5, h:6.5, rd:.05, te:.7};
    case 'o': return {w:8, h:9.5, rd:.85, te:0};
    case 'c': return {w:8, h:3.2, rd:.2, te:.5};
    case 'n': return {w:7, h:0, rd:0, te:0, press:1};
    default:
      if (E.rest === 'o') return {w:6, h:6.5, rd:1, te:0};
      if (E.rest === 'grin') return {w:13, h:7.5, rd:0, te:.3, grin:1};
      if (E.rest === 'smile') return {w:10, h:0, rd:0, te:0};
      if (E.rest === 'wobble') return {w:8, h:0, rd:0, te:0, wob:1};
      return {w:7.5, h:0, rd:0, te:0};
  }
}

function drawMouth(c, mx, my, v, E, t){
  const M = Object.assign({}, mouthShape(v, E)); M.w *= 1.25; M.h *= 1.3;
  const sm = E.sm;
  c.lineCap = 'round'; c.lineJoin = 'round';
  if (M.h < 1){
    const w = M.w/2, cy = -sm*2.2, mid = sm*2.6;
    c.strokeStyle = C.lip; c.lineWidth = M.press ? 2 : 1.6;
    c.beginPath(); c.moveTo(mx - w, my + cy);
    if (M.wob){ c.bezierCurveTo(mx - w*.4, my - 1.5, mx + w*.2, my + 1.8, mx + w, my + cy + .5); }
    else c.quadraticCurveTo(mx, my + mid, mx + w, my + cy);
    c.stroke();
    return;
  }
  const w = M.w/2, h = M.h;
  const lift = sm*(M.grin ? 3 : 1.6);
  const fn = cc => {
    if (M.grin){
      cc.moveTo(mx - w, my - lift); cc.quadraticCurveTo(mx, my - lift + 1.5, mx + w, my - lift);
      cc.bezierCurveTo(mx + w*.8, my + h*.9, mx - w*.8, my + h*.9, mx - w, my - lift); return;
    }
    const top = my - h*.45, bot = my + h*.55;
    const rd = M.rd;
    cc.moveTo(mx - w, my - lift*.6);
    cc.bezierCurveTo(mx - w, lerpA(my - lift*.6, top, .5 + rd*.5) - (1-rd)*1, mx - w*.5*(1 - rd*.1), top, mx, top + (1 - rd)*.6);
    cc.bezierCurveTo(mx + w*.5, top, mx + w, lerpA(my - lift*.6, top, .5 + rd*.5) - (1-rd)*1, mx + w, my - lift*.6);
    cc.bezierCurveTo(mx + w*(1 - rd*.05), lerpA(my, bot, .7 + rd*.3), mx + w*.45, bot, mx, bot);
    cc.bezierCurveTo(mx - w*.45, bot, mx - w*(1 - rd*.05), lerpA(my, bot, .7 + rd*.3), mx - w, my - lift*.6);
  };
  c.save();
  c.beginPath(); fn(c); c.fillStyle = C.mouth; c.fill(); c.clip();
  c.fillStyle = C.tongue; c.beginPath(); c.ellipse(mx + .5, my + h*.62, w*.75, h*.38 + 1, 0, 0, 2*PI); c.fill();
  if (M.te > 0){ c.fillStyle = '#fffaf4'; c.globalAlpha = Math.min(1, M.te + .2); c.fillRect(mx - w, my - h*.6 - lift, 2*w, h*.32*M.te + 1.4 + lift*.5); c.globalAlpha = 1; }
  c.restore();
  c.beginPath(); fn(c); c.strokeStyle = C.lip; c.lineWidth = 1.3; c.stroke();
}

function drawHead(c, sk, t, P, o, E, mouth){
  const talking = mouth && mouth !== 'x';
  const nod = (talking ? .025*sin(t*8.3) + .015*sin(t*13.1) : 0) + .015*U.noise(t*.45, 9);
  const ha = sk.lean*.55 + P.head + nod;
  const run = P.run;
  c.save(); c.translate(sk.N[0], sk.N[1]); c.rotate(ha); c.scale(HS, HS); c.translate(0, -41);
  // light direction into head frame
  const lx0 = LX, ly0 = LY, rim0 = RIM, ca = cos(-ha), sa = sin(-ha);
  LX = lx0*ca - ly0*sa; LY = lx0*sa + ly0*ca;
  if (RIM) RIM = Object.assign({}, RIM, {x:RIM.x*ca - RIM.y*sa, y:RIM.x*sa + RIM.y*ca});
  const sw = 2.2*U.noise(t*.9, 21) + 1.6*sin(t*1.7) - run*(3 + 2*sin(t*2*PI*3.2)) - ha*8;   // hair sway
  const bo = run*1.6*sin(t*2*PI*3.2);
  // --- back hair mass
  part(c, cc => {
    cc.moveTo(4, -58);
    cc.bezierCurveTo(32, -58, 53, -36, 52, -6);
    cc.bezierCurveTo(52, 12, 50, 26, 46 + sw*.5, 35 + bo);
    cc.quadraticCurveTo(34, 40, 18, 36);
    cc.lineTo(-20, 36);
    cc.quadraticCurveTo(-40, 42, -52 + sw, 38 + bo);
    cc.quadraticCurveTo(-56 + sw*.6, 30, -55, 22);
    cc.bezierCurveTo(-60, 4, -58, -18, -54, -30);
    cc.bezierCurveTo(-46, -50, -24, -58, 4, -58);
    cc.closePath();
  }, {b:C.hair.s, s:'#1f1119', l:C.hair.l}, 4, 3);
  // --- face
  part(c, cc => {
    cc.moveTo(-40, -14);
    cc.bezierCurveTo(-42, 10, -34, 28, -16, 38);
    cc.bezierCurveTo(-6, 44, 6, 47, 13, 45);
    cc.bezierCurveTo(28, 41, 38, 27, 39, 9);
    cc.bezierCurveTo(40, -8, 38, -24, 30, -34);
    cc.bezierCurveTo(10, -50, -30, -46, -40, -14);
    cc.closePath();
  }, C.skin, 4, 3);
  // ear hint under hair is hidden; cheeks
  const bl = E.bl;
  for (const [x, y, r] of [[-21, 25, 9], [29, 25, 7]]){
    const g = c.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, U.rgba(C.cheek, .55*bl)); g.addColorStop(1, U.rgba(C.cheek, 0));
    c.fillStyle = g; c.beginPath(); c.ellipse(x, y, r*1.3, r*.8, 0, 0, 2*PI); c.fill();
  }
  if (bl > .75){
    c.strokeStyle = U.rgba('#e86a72', (bl - .75)*2.4); c.lineWidth = 1; c.beginPath();
    for (let i = 0; i < 3; i++){ c.moveTo(-26 + i*4, 27); c.lineTo(-24 + i*4, 23); }
    c.stroke();
  }
  // eyes
  const blink = (o.blink === false) ? 0 : (typeof o.blink === 'number' ? o.blink : autoBlink(t, o.blinkSeed || 0));
  const lx = U.clamp((o.lookX == null ? .25 : o.lookX) + P.eyeX, -1, 1), ly = U.clamp((o.lookY || 0) + P.eyeY, -1, 1);
  drawEye(c, -12, 9.5, 10, 12.6, E, blink, lx, ly, -1, t, false);
  drawEye(c, 22.5, 9.5, 8, 12.1, E, blink, lx, ly, 1, t, true);
  // nose & mouth
  c.strokeStyle = C.skin.l; c.lineWidth = 1.2; c.lineCap = 'round';
  c.beginPath(); c.moveTo(12.5, 19.5); c.lineTo(13.5, 21.5); c.stroke();
  drawMouth(c, 8.5, 31, mouth || 'x', E, t);
  // --- front hair (cap, bangs, side locks)
  const capFn = cc => {
    cc.moveTo(-55, 2);
    cc.bezierCurveTo(-58, -38, -30, -61, 4, -60);
    cc.bezierCurveTo(34, -59, 54, -38, 51, -6);
    // front side lock
    cc.bezierCurveTo(53, 8, 51, 24, 45 + sw*.4, 38 + bo);
    cc.bezierCurveTo(41, 26, 39, 12, 36.5, -1);
    // bangs (front -> back)
    cc.quadraticCurveTo(35, -9, 30, -14);
    cc.quadraticCurveTo(31, -6, 29, 0);
    cc.quadraticCurveTo(24, -9, 17, -17);
    cc.quadraticCurveTo(19, -7, 15, 2);
    cc.quadraticCurveTo(9, -9, 2, -16);
    cc.quadraticCurveTo(2, -6, -3, 1);
    cc.quadraticCurveTo(-7, -9, -12, -15);
    cc.quadraticCurveTo(-13, -5, -19, 0);
    cc.quadraticCurveTo(-22, -8, -27, -12);
    cc.quadraticCurveTo(-29, -4, -33, 1);
    // near side lock in front of the cheek
    cc.bezierCurveTo(-36, 12, -36, 26, -31 + sw*.5, 41 + bo);
    cc.bezierCurveTo(-41, 34, -46, 22, -47, 8);
    cc.quadraticCurveTo(-52, 6, -55, 2);
    cc.closePath();
  };
  part(c, capFn, C.hair, 4, 3);
  // hair highlights + strand lines (clipped to cap)
  c.save(); c.beginPath(); capFn(c); c.clip();
  c.fillStyle = C.hair.h;
  const ring = (a0, a1, w) => {
    c.beginPath();
    const n = 10;
    for (let i = 0; i <= n; i++){ const a = lerpA(a0, a1, i/n), r = 43 + w*sin(PI*i/n); c.lineTo(-2 + cos(a)*r*1.05, -14 + sin(a)*r*.82); }
    for (let i = n; i >= 0; i--){ const a = lerpA(a0, a1, i/n), r = 43 - w*.6*sin(PI*i/n); c.lineTo(-2 + cos(a)*r*1.05, -14 + sin(a)*r*.82); }
    c.fill();
  };
  ring(3.55, 3.95, 3.2); ring(4.02, 4.45, 3.6); ring(4.52, 4.85, 2.8); ring(4.93, 5.15, 2);
  c.strokeStyle = C.hair.l; c.lineWidth = 1.1; c.globalAlpha = .55;
  c.beginPath();
  c.moveTo(-6, -46); c.quadraticCurveTo(-12, -30, -12, -15);
  c.moveTo(10, -46); c.quadraticCurveTo(14, -30, 17, -17);
  c.moveTo(-26, -40); c.quadraticCurveTo(-34, -24, -33, 1);
  c.moveTo(30, -38); c.quadraticCurveTo(38, -20, 40, 10);
  c.stroke(); c.globalAlpha = 1;
  c.restore();
  // loose strands + ahoge
  const aw = 3*sin(t*2.3) + 2*U.noise(t*1.1, 31) - run*4;
  const lock = (x0, y0, cx, cy, x1, y1, w, m) => part(c, cc => {
    const dx = x1 - x0, dy = y1 - y0, d = Math.hypot(dx, dy) || 1, nx = -dy/d*w, ny = dx/d*w;
    cc.moveTo(x0 + nx, y0 + ny); cc.quadraticCurveTo(cx + nx*.6, cy + ny*.6, x1, y1);
    cc.quadraticCurveTo(cx - nx*.6, cy - ny*.6, x0 - nx, y0 - ny); cc.closePath();
  }, m || C.hair, 0, 1.5);
  lock(-4, -55, 2 + aw*.2, -82, 16 + aw, -70, 2.6);                 // ahoge
  lock(-50, 14, -57 + sw*.6, 28, -61 + sw*1.3, 40 + bo, 2.6, {b:C.hair.s, l:C.hair.l});
  lock(47, 12, 53 + sw*.4, 26, 52 + sw*.8, 39 + bo, 2.2, {b:C.hair.s, l:C.hair.l});
  c.strokeStyle = C.hair.l; c.lineCap = 'round'; c.globalAlpha = .7;
  c.lineWidth = 1.1; c.beginPath(); c.moveTo(4, -14); c.quadraticCurveTo(7 + sw*.3, -4, 5 + sw*.4, 4); c.stroke(); c.globalAlpha = 1;
  // brows (drawn over bangs, slightly transparent)
  c.globalAlpha = .85; c.strokeStyle = C.hair.l; c.lineCap = 'round';
  const by = -9.5 - E.bY, ba = E.bA*8;
  c.lineWidth = 2.2;
  c.beginPath(); c.moveTo(-21, by + 2 + ba*.2); c.quadraticCurveTo(-12, by - 2.5 + ba*.1, -3, by + 1 - ba); c.stroke();
  c.lineWidth = 1.9;
  c.beginPath(); c.moveTo(16, by + 1 - ba); c.quadraticCurveTo(22, by - 2.5 + ba*.1, 28, by + 1.5 + ba*.2); c.stroke();
  c.globalAlpha = 1;
  c.restore();
  LX = lx0; LY = ly0; RIM = rim0;
}

// ---------------- main ----------------
function draw(ctx, t, o){
  o = o || {};
  const s = o.scale || 1, f = o.facing < 0 ? -1 : 1;
  let P = poseParams(o.pose || 'stand', t, o.poseT, o);
  if (o.pose2 && o.mix > 0){
    const Q = poseParams(o.pose2, t, o.poseT2 != null ? o.poseT2 : o.poseT, o), k = U.clamp(o.mix), R = {};
    for (const key in P) R[key] = typeof P[key] === 'number' ? lerpA(P[key], Q[key], k) : (k < .5 ? P[key] : Q[key]);
    P = R;
  }
  const E = exprParams(o);
  const sk = skeleton(P, t);
  ctx.save();
  ctx.translate(o.x || 0, o.y || 0);
  ctx.scale(s*f, s);
  ctx.translate(0, sk.dy);
  // lights: key light from upper-left in world
  LX = -.55*f; LY = -.83;
  RIM = null;
  if (o.rim && (o.rim.strength == null || o.rim.strength > 0)){
    let dx = (o.rim.x - (o.x || 0))*f/s, dy = (o.rim.y - (o.y || 0))/s - sk.dy + 150;
    const d = Math.hypot(dx, dy) || 1;
    RIM = {x:dx/d, y:dy/d, col:o.rim.color || '#ffd27a', k:U.clamp(o.rim.strength == null ? 1 : o.rim.strength)};
  }
  // contact shadow (stays on ground)
  ctx.save(); ctx.translate(0, -sk.dy);
  const sh = U.clamp(1 - (sk.dy < 0 ? 0 : 0) - P.air/60);
  ctx.fillStyle = 'rgba(10,8,30,' + (.28*sh) + ')';
  ctx.beginPath(); ctx.ellipse(P.hipX*.5 + 2, 0, 38 + P.hipY*.2, 6, 0, 0, 2*PI); ctx.fill();
  ctx.restore();
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  const mouth = o.mouth || 'x';
  scarfTails(ctx, sk, t, P, o);
  if (P.fo < .5) drawArm(ctx, sk.armF, P.hF, true);
  drawLeg(ctx, sk.legF, t);
  drawLeg(ctx, sk.legN, t);
  // neck
  limb(ctx, [sk.T(0, 62), [sk.N[0], sk.N[1] - 4]], [12], C.skin);
  drawCoat(ctx, sk, t, P);
  scarfWrap(ctx, sk, t);
  if (P.fo >= .5) drawArm(ctx, sk.armF, P.hF, true);
  drawHead(ctx, sk, t, P, o, E, mouth);
  drawArm(ctx, sk.armN, P.hN, false);
  ctx.restore();
  RIM = null;
}

window.Mina = {draw, POSES:Object.keys(POSES), EXPRS:Object.keys(EXPR),
  // joint positions in world space (e.g. to place props in her hand): returns {hand, handFar, head, chest}
  joints(t, o){
    o = o || {}; const s = o.scale || 1, f = o.facing < 0 ? -1 : 1;
    let P = poseParams(o.pose || 'stand', t, o.poseT, o);
    if (o.pose2 && o.mix > 0){ const Q = poseParams(o.pose2, t, o.poseT, o), k = U.clamp(o.mix), R = {};
      for (const key in P) R[key] = typeof P[key] === 'number' ? lerpA(P[key], Q[key], k) : P[key]; P = R; }
    const sk = skeleton(P, t);
    const W = p => [(o.x || 0) + p[0]*s*f, (o.y || 0) + (p[1] + sk.dy)*s];
    return {hand:W(sk.armN.W), handFar:W(sk.armF.W), head:W([sk.N[0], sk.N[1] - 41*HS]), chest:W(sk.T(0, 45))};
  }
};
})();
