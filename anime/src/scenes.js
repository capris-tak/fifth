// Director: camera, staging, cuts. Pure function of t.
(function(){
const {lerp, inv, clamp, easeInOut, easeOut, easeIn, smooth} = U;
const LH = () => Object.assign({x:260, lampX:260, lampY:-860, galleryY:-790, galleryHalfW:110, doorX:260, doorY:0}, World.LH||{});
const HX = -140;                                   // Hoshi crash site
const gy = x => World.groundY(x);
const MS = 0.5, HS = 0.5;                          // character scales on the ground
const hoshiRest = () => [HX, gy(HX) - 26];

// camera keyframe helper: keys = [[t, {x,y,zoom}], ...] eased between
function camPath(t, keys, ease=easeInOut){
  if (t <= keys[0][0]) return {...keys[0][1]};
  for (let i=0;i<keys.length-1;i++){
    const [t0,a]=keys[i],[t1,b]=keys[i+1];
    if (t<=t1){const k=ease(inv(t0,t1,t));return {x:lerp(a.x,b.x,k),y:lerp(a.y,b.y,k),zoom:lerp(a.zoom,b.zoom,k)};}
  }
  return {...keys[keys.length-1][1]};
}
function shake(cam, t, t0, amp, dur){
  const k = inv(t0, t0+dur, t); if (k<=0||k>=1) return cam;
  const a = amp*(1-k)*(1-k);
  return {...cam, x:cam.x+U.noise(t*40,3)*a, y:cam.y+U.noise(t*40,7)*a};
}

// Hoshi path during ascent to gallery (37.9–39.0) and to the sky (48.6–51.5)
function hoshiGallery(t){ const L=LH(); return [L.x-135 + Math.sin(t*1.7)*4, L.galleryY-95 + Math.sin(t*2.3)*5]; }
function hoshiSkyPos(){ const L=LH(); return [L.x+430, L.lampY-640]; }
function hoshiPos(t){
  const [rx,ry]=hoshiRest();
  if (t < 35.0) return [rx, ry];
  if (t < 37.9) { const k=easeInOut(inv(35,37.6,t)); return [rx, ry - 22*k + Math.sin(t*9)*1.5*k]; }
  if (t < 39.0) { // swoop up to the gallery along a curve
    const k = easeInOut(inv(37.9, 39.0, t)); const [gx,gy2]=hoshiGallery(t);
    const sx=rx, sy=ry-22; const cx=sx-160, cy=(sy+gy2)/2;
    const x=(1-k)*(1-k)*sx+2*(1-k)*k*cx+k*k*gx, y=(1-k)*(1-k)*sy+2*(1-k)*k*cy+k*k*gy2; return [x,y];
  }
  if (t < 48.6) return hoshiGallery(t);
  const k = easeInOut(inv(48.6, 51.5, t)); const [gx,gy2]=hoshiGallery(48.6); const [sx,sy]=hoshiSkyPos();
  const cx = gx + 60, cy = sy + 60;
  return [(1-k)*(1-k)*gx+2*(1-k)*k*cx+k*k*sx, (1-k)*(1-k)*gy2+2*(1-k)*k*cy+k*k*sy + (k>=1?Math.sin(t*2)*4:0)];
}
function hoshiBright(t){
  if (t < 12.5) return 1;
  if (t < 13.8) return lerp(0.9, 0.12, easeOut(inv(12.5, 13.8, t)));
  if (t < 35.0) return 0.12 + 0.05*Math.sin(t*3.1);
  if (t < 37.6) return lerp(0.15, 0.7, easeIn(inv(35, 37.6, t)));
  return 1;
}

// Mina state machine → {x,y,scale,facing,pose,expr,lookX,lookY, alpha}
function minaState(t){
  const L=LH(); const mouth = TL.mouth('mina', t);
  if (t < 12.9) return null;                                       // inside
  if (t < 13.8) { const k=inv(13.0,13.8,t); const x=lerp(L.doorX, 20, easeOut(k));
    return {x, y:gy(x), facing:-1, pose:k<1&&t>13.0?'run':'stand', expr:'surprised', mouth, alpha:clamp((t-12.95)/0.15)}; }
  if (t < 17.0) return {x:20, y:gy(20), facing:-1, pose:'surprised', poseT:inv(13.8,14.4,t), expr:'surprised', mouth, lookX:-0.6, lookY:0.4};
  if (t < 28.6) {
    const talkingM = TL.talking('mina', t);
    return {x:-88, y:gy(-88), facing:-1, pose: (t>22.6&&t<25.2)?'reach':'crouch', pose2:'crouch', mix:0,
            expr: talkingM?'worried':(t>25.4?'worried':'worried'), mouth, lookX:-0.8, lookY:0.5};
  }
  if (t < 30.8) { const k=inv(28.6,29.3,t);
    return {x:-88, y:gy(-88), facing: t<29.9?-1:1, pose: t<29.0?'crouch':'point', pose2:'crouch', mix: 1-k, expr:'determined', mouth, lookX: t<29.9?-0.5:0.6, lookY:-0.6}; }
  if (t < 31.9) { const k=inv(30.8,31.7,t); const x=lerp(-88, L.doorX, easeIn(k)*0.6+k*0.4);
    return {x, y:gy(x), facing:1, pose:'run', expr:'determined', mouth, alpha:1-inv(31.55,31.75,t)}; }
  // on the gallery from 32.0 on
  const gx = L.x - 40;
  let pose='stand', expr='determined', lookX=-0.6, lookY=0.6, mouthM=mouth;
  if (t < 37.6) { pose='reach'; expr='determined'; lookX=-0.8; lookY=0.8; }
  else if (t < 39.0) { pose='surprised'; expr='joy'; }
  else if (t < 42.6) { pose='clasp'; expr='happy'; lookX=-0.9; lookY=-0.2; }
  else if (t < 45.0) { pose='clasp'; expr='wistful'; lookX=-0.9; lookY=-0.1; }
  else if (t < 48.6) { pose='stand'; expr='happy'; lookX=-0.9; lookY=-0.2; }
  else if (t < 52.0) { pose='wave'; expr='joy'; lookX=-0.4; lookY=-1; }
  else { pose='lookup'; expr='happy'; lookX=0; lookY=-1; }
  return {x:gx, y:L.galleryY, facing:-1, pose, expr, mouth:mouthM, lookX, lookY, scale:0.55};
}
function hoshiState(t){
  if (t < 12.45) return null;
  const [x,y]=hoshiPos(t); const mouth=TL.mouth('hoshi',t);
  let expr='hurt', rot=0, squash=0, flicker=0;
  if (t < 17.0) { expr='hurt'; rot=0.5*(1-inv(15.5,16.8,t)); squash=0.3*(1-inv(12.5,13.2,t)); flicker=1; }
  else if (t < 25.4) { expr = t<22.5?'hurt':'sad'; flicker=0.8; }
  else if (t < 35.0) { expr = 'hope'; flicker=0.7; }
  else if (t < 37.6) { expr='surprised'; flicker=0.3*(1-inv(35,37,t)); squash=0.15*Math.sin(t*14)*inv(35,37.6,t); }
  else if (t < 42.6) expr='joy';
  else if (t < 48.6) expr='gentle';
  else expr='joy';
  const lookX = t>=37.9 && t<48.6 ? 0.8 : (t<28.6 && t>17 ? 0.7 : 0), lookY = t>=37.9&&t<48.6 ? 0.2 : (t<28.6?-0.4:0);
  return {x, y, rot, squash, bright:hoshiBright(t), flicker, expr, mouth, lookX, lookY, scale: HS};
}

// ---------------- shot list ----------------
function camera(t){
  const L=LH(); const [hx,hy]=hoshiRest();
  if (t < 10.5) return camPath(t, [[0,{x:-40,y:-2300,zoom:0.62}],[1.0,{x:-40,y:-2300,zoom:0.62}],[10.5,{x:60,y:-280,zoom:0.74}]]);
  if (t < 17.0) { let c = camPath(t, [[10.5,{x:60,y:-300,zoom:0.76}],[12.6,{x:30,y:-280,zoom:0.8}],[13.0,{x:30,y:-280,zoom:0.8}],[17.0,{x:-40,y:-150,zoom:1.7}]]);
    return shake(c, t, 12.5, 14, 0.9); }
  if (t < 22.75) return camPath(t, [[17,{x:-112,y:-62,zoom:4.1}],[22.75,{x:-110,y:-60,zoom:4.35}]]);          // two-shot
  if (t < 25.2) return camPath(t, [[22.75,{x:-80,y:-82,zoom:5.3}],[25.2,{x:-80,y:-84,zoom:5.55}]]);          // on Mina
  if (t < 28.6) return camPath(t, [[25.2,{x:hx+12,y:hy-14,zoom:6.2}],[28.6,{x:hx+12,y:hy-14,zoom:6.7}]]);    // on Hoshi
  if (t < 30.8) return camPath(t, [[28.6,{x:-96,y:-84,zoom:4.6}],[30.8,{x:-96,y:-92,zoom:4.8}]]);           // Mina determined
  if (t < 32.0) return camPath(t, [[30.8,{x:60,y:-200,zoom:1.45}],[32.0,{x:120,y:-230,zoom:1.5}]]);           // run to door
  if (t < 33.5) return camPath(t, [[32.0,{x:L.x-20,y:L.galleryY-60,zoom:2.5}],[33.5,{x:L.x-30,y:L.galleryY-55,zoom:2.7}]]); // lamp room
  if (t < 35.0) return camPath(t, [[33.5,{x:40,y:-470,zoom:0.78}],[35.0,{x:20,y:-430,zoom:0.8}]]);            // beam sweep wide
  if (t < 37.6) return camPath(t, [[35.0,{x:hx+8,y:hy-30,zoom:3.0}],[37.6,{x:hx+8,y:hy-34,zoom:3.9}]], easeIn); // charging
  if (t < 39.0) return shake(camPath(t, [[37.6,{x:40,y:-430,zoom:0.8}],[39.0,{x:60,y:-500,zoom:0.86}]]), t, 37.6, 18, 1.1); // burst wide
  if (t < 42.6) return camPath(t, [[39.0,{x:L.x-85,y:L.galleryY-88,zoom:2.6}],[42.6,{x:L.x-85,y:L.galleryY-88,zoom:2.75}]]);   // gallery two-shot
  if (t < 45.0) return camPath(t, [[42.6,{x:L.x-45,y:L.galleryY-95,zoom:4.4}],[45.0,{x:L.x-45,y:L.galleryY-97,zoom:4.6}]]);       // Mina close
  if (t < 48.6) return camPath(t, [[45.0,{x:L.x-128,y:L.galleryY-96,zoom:5.0}],[48.6,{x:L.x-128,y:L.galleryY-96,zoom:5.3}]]);   // Hoshi close
  if (t < 51.5) return camPath(t, [[48.6,{x:L.x-85,y:L.galleryY-100,zoom:2.5}],[51.5,{x:L.x+200,y:L.lampY-480,zoom:0.95}]]);       // follow ascent
  return camPath(t, [[51.5,{x:L.x+200,y:L.lampY-480,zoom:0.95}],[60,{x:L.x+170,y:L.lampY-360,zoom:0.68}]], t=>easeOut(t));            // final pull back
}

function lampState(t){
  const L=LH();
  // default: lamp turning slowly, beam sweeping the horizon (angle near 0 or π, intensity ~ facing)
  const spin = t*0.9;
  let o = {lampOn:1, lampAngle:spin, windowsLit:1, doorOpen:0};
  o.doorOpen = t<12.9?0 : t<17?easeOut(inv(12.9,13.3,t)) : t<30.8?1 : t<31.9?1 : t<32.2?1-inv(31.9,32.2,t):0;
  let beam = {angle: Math.cos(spin)>0 ? 0.05 : Math.PI-0.05, length: 2600, width: 110, intensity: 0.55*Math.pow(Math.abs(Math.cos(spin)),3)};
  if (t >= 32.0 && t < 33.5) { o.lampOn = 1 + 0.6*inv(32.4,33.4,t); beam.intensity *= 1-inv(32,32.4,t); }
  if (t >= 33.5 && t < 37.9) {
    const target = Math.atan2(hoshiRest()[1]-L.lampY, hoshiRest()[0]-L.lampX);
    const k = easeInOut(inv(33.5, 34.8, t));
    beam = {angle: lerp(Math.PI-0.05, target, k), length: Math.hypot(hoshiRest()[0]-L.lampX, hoshiRest()[1]-L.lampY)+80,
            width: 140, intensity: 0.95*inv(33.5,33.9,t)};
    o.lampOn = 1.6;
  }
  if (t >= 37.6 && t < 39.5) beam.intensity *= 1 - inv(37.6, 38.2, t);
  if (t >= 37.6 && t < 39.5) beam.intensity = Math.max(beam.intensity, 0);
  return {o, beam};
}

function drawWorld(ctx, t, cam, ms, hs){
  const L=LH();
  World.sky(ctx, cam, t, {aurora: 0.45 + 0.15*inv(37.6,40,t) - 0.2*inv(51,54,t), starBoost: inv(50,56,t)});
  World.cloudsBack(ctx, cam, t);
  U.camApply(ctx, cam);
  const bloom = easeOut(inv(37.6, 39.5, t));
  World.island(ctx, t, {bloom});
  const {o, beam} = lampState(t);
  World.lighthouse(ctx, t, {...o, noRail: !!(ms && ms.y < -300)});
  // Mina on the gallery → between tower and front rail
  if (ms && ms.y < -300) { drawMina(ctx, t, ms, hs); if (World.galleryRail) World.galleryRail(ctx, t, o); }
  if (beam.intensity > 0.002) World.beam(ctx, t, beam);
  if (t>=33.6 && t<38.4) { const [hx,hy]=hoshiRest(); World.lightPool(ctx, hx, gy(hx), 230, 0.9*inv(33.8,34.8,t)*(1-inv(37.8,38.4,t)), '#ffd27a'); }
  if (hs && hs.bright>0.5 && t<39.2) World.lightPool(ctx, hs.x, gy(hs.x), 260*hs.bright, 0.6*hs.bright*(1-inv(38.2,39.2,t)), '#fff2a8');
  if (o.doorOpen>0.01) World.lightPool(ctx, L.doorX, gy(L.doorX), 160, 0.5*o.doorOpen, '#ffb347');
  // effects behind characters
  FX.motes(ctx, t, {x:-700, y:-500, w:1400, h:520, count:40, seed:5, color:'#fff2a8'});
  FX.shootingStar(ctx, t, {x0:-1500, y0:-1500, x1:HX, y1:gy(HX)-20, t0:10.8, t1:12.45});
  FX.impact(ctx, t, {x:HX, y:gy(HX)-10, t0:12.5, scale:0.8, groundY:gy(HX)});
  const [hx,hy]=hoshiRest();
  FX.absorb(ctx, t, {x:hx, y:hy, t0:35.0, t1:37.6, radius:260, scale:0.6, squash:0.45});
  if (hs && t>=37.9 && t<52) FX.trail(ctx, t, {path:hoshiPos, t0: t<48?37.9:48.6, t1: t<48?39.0:51.5});
  if (hs) Hoshi.draw(ctx, t, hs);
  if (ms && ms.y >= -300) drawMina(ctx, t, ms, hs);
  FX.burst(ctx, t, {x:hx, y:hy-22, t0:37.6, groundY:gy(hx)});
  World.cloudsFront(ctx, cam, t);
}
function drawMina(ctx, t, ms, hs){
  const L=LH();
  let rim = null;
  if (hs && hs.bright>0.4) rim = {x:hs.x, y:hs.y, color:'#ffd27a', strength:0.8*hs.bright};
  else if (ms.y < -300) rim = {x:L.lampX, y:L.lampY, color:'#ffd27a', strength:0.7};
  else if (t>12.9 && t<17) rim = {x:L.doorX, y:-60, color:'#ffb347', strength:0.5};
  const a = ms.alpha===undefined?1:ms.alpha; if (a<=0) return;
  ctx.save(); ctx.globalAlpha = a; Mina.draw(ctx, t, {scale:MS, ...ms, rim}); ctx.restore();
}

// constellation (screen-space) — lighthouse shape around Hoshi's final position
function constellation(ctx, t, cam){
  if (t < 52.3) return;
  const [sx,sy]=hoshiSkyPos(); const S=1;
  const P = [[0,-260],[-70,-150],[70,-150],[-55,-70],[55,-70],[-90,190],[90,190],[-130,200],[130,200],[0,-170],[-170,-260],[170,-260]]
    .map(([dx,dy])=>[sx+dx*S*1.5, sy+dy*S*1.5+150]);
  const lines=[[0,1],[0,2],[1,2],[1,3],[2,4],[3,4],[3,5],[4,6],[5,6],[5,7],[6,8],[9,10],[9,11]];
  ctx.save(); ctx.setTransform(1,0,0,1,0,0);
  FX.constellation(ctx, t, {points:P, lines, t0:52.5, t1:56.0, cam});
  ctx.restore();
}

window.Scenes = { render(ctx, t){
  const cam = camera(t);
  const ms = minaState(t), hs = hoshiState(t);
  drawWorld(ctx, t, cam, ms, hs);
  constellation(ctx, t, cam);
  ctx.setTransform(1,0,0,1,0,0);
  // flash on impact and burst
  const fl = Math.max(0.22*(1-inv(12.5,12.75,t))*(t>=12.5?1:0), 0.45*(1-inv(37.6,37.95,t))*(t>=37.6?1:0));
  if (fl>0.001){ ctx.save(); ctx.globalCompositeOperation='lighter'; ctx.fillStyle=`rgba(255,240,200,${fl})`; ctx.fillRect(0,0,U.W,U.H); ctx.restore(); }
  FX.bloom(ctx, 0.38 + 0.2*inv(37.6,38,t)*(1-inv(39,41,t)));
  FX.vignetteGrain(ctx, t, 0.38, 0.035);
  FX.title(ctx, t, {t0:56.5, x:U.W*0.3, y:U.H*0.46, size:104});
  FX.credits(ctx, t, {t0:57.5});
  const fade = Math.max(1-inv(0,1.5,t), inv(59.2,60,t));
  if (fade>0) FX.fade(ctx, fade);
}, camera, hoshiPos };
})();
