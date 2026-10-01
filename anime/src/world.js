// World: sky, cloud sea, floating island, lighthouse, beam. Pure functions of t; static art cached in U.buf.
(function(){
const W = {};
const TAU = Math.PI*2;
const SP = 0.15;           // sky (stars/moon/aurora) parallax
const HP = 0.3, HORIZON_Y = 40; // horizon = far cloud layer
const MOON = {x:-640, y:-430, r:70};
const STAR_X0=-1150, STAR_Y0=-1000, STAR_W=2300, STAR_H=1600;

// ---------------------------------------------------------------- ground
const rawG = x => -10*Math.sin(x*0.0045+0.6) - 7*Math.sin(x*0.011+2.1) + 4*Math.sin(x*0.023);
const G260 = rawG(260);
W.groundY = x => {
  let y = rawG(x);
  const d = Math.abs(x-260);
  if (d < 190) { const k = U.smooth(U.clamp((190-d)/70)); y = U.lerp(y, G260, k); }
  const ax = Math.abs(x);
  if (ax > 640) { const k = (ax-640)/90; y += k*k*24; }
  return y;
};

// ---------------------------------------------------------------- constants
const LH = W.LH = {
  x:260, baseY:W.groundY(260), galleryY:-790, galleryHalfW:104, galleryDepth:13,
  lampX:260, lampY:-866, doorX:260, doorY:W.groundY(260)+2, doorW:95, doorH:175,
  towerTop:-776, glassTop:-912, glassBot:-820, glassHW:48, railH:44
};
W.LAMP = {x:LH.lampX, y:LH.lampY};

// ---------------------------------------------------------------- helpers
function glowSprite(color){
  const name='w_glow_'+color; const c=U.buf(name,64,64);
  if(c._ok) return c; c._ok=1;
  const g=c.getContext('2d'); const gr=g.createRadialGradient(32,32,0,32,32,32);
  gr.addColorStop(0,U.rgba(color,1)); gr.addColorStop(0.18,U.rgba(color,0.45));
  gr.addColorStop(0.45,U.rgba(color,0.1)); gr.addColorStop(1,U.rgba(color,0));
  g.fillStyle=gr; g.fillRect(0,0,64,64); return c;
}
function spikeSprite(){
  const c=U.buf('w_spike',128,128); if(c._ok) return c; c._ok=1;
  const g=c.getContext('2d');
  const line=(len,th,a,rot)=>{g.save();g.translate(64,64);g.rotate(rot);
    const gr=g.createLinearGradient(-len,0,len,0);
    gr.addColorStop(0,'rgba(255,255,255,0)');gr.addColorStop(0.5,`rgba(255,255,255,${a})`);gr.addColorStop(1,'rgba(255,255,255,0)');
    g.fillStyle=gr; g.fillRect(-len,-th/2,len*2,th); g.restore();};
  line(64,2.2,1,0); line(64,2.2,1,Math.PI/2); line(30,1.4,0.5,Math.PI/4); line(30,1.4,0.5,-Math.PI/4);
  return c;
}
function skyTf(cam){ const z=1+(cam.zoom-1)*SP; return {z, ox:960-cam.x*SP*z, oy:540-cam.y*SP*z}; }
function horizonY(cam){ const z=1+(cam.zoom-1)*HP; return 540+(HORIZON_Y-cam.y*HP)*z; }
W.horizonY = horizonY;
// draw a buffer whose pixel (0,0) maps to (sx,sy) at scale k, clipped to the screen
function blitClip(ctx,buf,sx,sy,k){
  const x0=Math.max(0,(0-sx)/k), y0=Math.max(0,(0-sy)/k);
  const x1=Math.min(buf.width,(1920-sx)/k), y1=Math.min(buf.height,(1080-sy)/k);
  if(x1<=x0||y1<=y0) return;
  ctx.drawImage(buf,x0,y0,x1-x0,y1-y0,sx+x0*k,sy+y0*k,(x1-x0)*k,(y1-y0)*k);
}
function gauss(r){ return (r()+r()+r()+r()-2)/2; }

// ---------------------------------------------------------------- sky caches
const MW_A=[-1150,450], MW_B=[1150,-950];
const MW_L=Math.hypot(MW_B[0]-MW_A[0],MW_B[1]-MW_A[1]);
const MW_D=[(MW_B[0]-MW_A[0])/MW_L,(MW_B[1]-MW_A[1])/MW_L], MW_N=[-MW_D[1],MW_D[0]];
function mwPoint(s,off){ return [MW_A[0]+MW_D[0]*s*MW_L+MW_N[0]*off, MW_A[1]+MW_D[1]*s*MW_L+MW_N[1]*off]; }

function milkyBuf(){
  const c=U.buf('w_milky',STAR_W/2,STAR_H/2); if(c._ok) return c; c._ok=1;
  const tmp=document.createElement('canvas'); tmp.width=c.width; tmp.height=c.height;
  const g=tmp.getContext('2d'); const r=U.rng(4242);
  g.translate(-STAR_X0/2,-STAR_Y0/2); g.scale(0.5,0.5);
  g.globalCompositeOperation='lighter';
  const cols=['#9aa0e0','#c7a6e6','#8fb6e8','#e6c2d6','#b0a8ff'];
  for(let i=0;i<1100;i++){
    const s=r(), wide=130*(1+0.6*U.noise(s*7,3));
    const [x,y]=mwPoint(s, gauss(r)*wide);
    const rad=30+r()*110, a=(0.018+r()*0.04)*(0.5+0.5*Math.sin(s*Math.PI));
    const gr=g.createRadialGradient(x,y,0,x,y,rad);
    gr.addColorStop(0,U.rgba(cols[(r()*cols.length)|0],a)); gr.addColorStop(1,U.rgba('#000000',0));
    g.fillStyle=gr; g.fillRect(x-rad,y-rad,rad*2,rad*2);
  }
  g.globalCompositeOperation='destination-out';
  for(let i=0;i<260;i++){
    const s=r(); const [x,y]=mwPoint(s, 25*U.noise(s*9,5)+gauss(r)*30);
    const rad=18+r()*50; const gr=g.createRadialGradient(x,y,0,x,y,rad);
    gr.addColorStop(0,'rgba(0,0,0,0.35)'); gr.addColorStop(1,'rgba(0,0,0,0)');
    g.fillStyle=gr; g.fillRect(x-rad,y-rad,rad*2,rad*2);
  }
  const cg=c.getContext('2d'); cg.filter='blur(5px)'; cg.drawImage(tmp,0,0); cg.filter='none';
  return c;
}
let SL=null;
function starList(){ // static faint stars grouped by colour/alpha for cheap batched fillRect
  if(SL) return SL; const r=U.rng(777); const groups={};
  const cols=['#ffffff','#e6eeff','#cfdcff','#fff0dc','#ffe0f0'];
  const add=(x,y,rad,a,col)=>{ const ab=Math.min(4,Math.round(a*5)); const k=col+ab; (groups[k]||(groups[k]={style:U.rgba(col,Math.max(0.12,ab/5)),pts:[]})).pts.push(x,y,rad); };
  for(let i=0;i<3400;i++){ const x=STAR_X0+r()*STAR_W, y=STAR_Y0+r()*STAR_H; add(x,y,0.35+Math.pow(r(),4)*0.9,0.15+r()*0.55,cols[(r()*5)|0]); }
  for(let i=0;i<2600;i++){ const s=r(); const [x,y]=mwPoint(s,gauss(r)*95); add(x,y,0.3+Math.pow(r(),5)*0.8,0.12+r()*0.5,cols[(r()*5)|0]); }
  SL=Object.values(groups); return SL;
}
let TW=null;
function twinkleStars(){
  if(TW) return TW; TW=[]; const r=U.rng(9001);
  const cols=['#ffffff','#ffffff','#cfe0ff','#ffe6c0','#ffd0e8','#c8fff4','#dcd0ff'];
  for(let i=0;i<760;i++){
    let x,y; if(i<600){ x=STAR_X0+r()*STAR_W; y=STAR_Y0+r()*STAR_H; } else { [x,y]=mwPoint(r(),gauss(r)*80); }
    const big=Math.pow(r(),3);
    TW.push({x,y,rad:0.7+big*1.9,a:0.45+r()*0.55,col:cols[(r()*cols.length)|0],f:0.8+r()*3.2,ph:r()*TAU,glint:0});
  }
  for(let i=0;i<18;i++){
    TW.push({x:STAR_X0+100+r()*(STAR_W-200), y:STAR_Y0+80+r()*(STAR_H*0.75), rad:1.8+r()*1.4, a:0.9, col:cols[(r()*cols.length)|0], f:0.6+r()*1.4, ph:r()*TAU, glint:14+r()*26});
  }
  return TW;
}
function moonSprite(){
  const R=MOON.r, S=1.5, HR=R*8, sz=Math.ceil(HR*2*S); const c=U.buf('w_moon',sz,sz); if(c._ok) return c; c._ok=1;
  const g=c.getContext('2d'); g.translate(sz/2,sz/2); g.scale(S,S);
  let gr=g.createRadialGradient(0,0,R*0.9,0,0,HR);
  gr.addColorStop(0,'rgba(170,165,255,0.16)'); gr.addColorStop(0.4,'rgba(120,110,220,0.05)'); gr.addColorStop(1,'rgba(100,90,200,0)');
  g.fillStyle=gr; g.fillRect(-HR,-HR,HR*2,HR*2);
  g.globalCompositeOperation='lighter';
  gr=g.createRadialGradient(0,0,R*0.95,0,0,R*5);
  gr.addColorStop(0,'rgba(255,248,225,0.42)'); gr.addColorStop(0.35,'rgba(230,220,255,0.12)'); gr.addColorStop(1,'rgba(200,190,255,0)');
  g.fillStyle=gr; g.fillRect(-R*5,-R*5,R*10,R*10);
  gr=g.createRadialGradient(0,0,R*3.1,0,0,R*4.1);
  gr.addColorStop(0,'rgba(200,210,255,0)'); gr.addColorStop(0.5,'rgba(210,200,255,0.06)'); gr.addColorStop(1,'rgba(200,210,255,0)');
  g.fillStyle=gr; g.fillRect(-R*4.2,-R*4.2,R*8.4,R*8.4);
  g.globalCompositeOperation='source-over';
  gr=g.createRadialGradient(-R*0.25,-R*0.25,0,0,0,R);
  gr.addColorStop(0,'#fffef6'); gr.addColorStop(0.6,'#f6f1e2'); gr.addColorStop(1,'#d8d2ea');
  g.fillStyle=gr; g.beginPath(); g.arc(0,0,R,0,TAU); g.fill();
  g.save(); g.beginPath(); g.arc(0,0,R,0,TAU); g.clip(); const r=U.rng(31);
  const maria=[[-12,-10,15],[8,-16,10],[14,4,13],[-4,12,9],[-18,8,7]].map(m=>m.map(v=>v*R/44));
  g.filter='blur(4px)';
  for(const [x,y,s] of maria){ g.fillStyle='rgba(175,165,200,0.16)'; g.beginPath(); g.ellipse(x,y,s,s*0.8,r()*3,0,TAU); g.fill(); }
  g.filter='none';
  for(let i=0;i<14;i++){ const x=(r()-0.5)*R*1.6,y=(r()-0.5)*R*1.6,s=1+r()*3; g.strokeStyle='rgba(150,140,180,0.08)'; g.lineWidth=0.6; g.beginPath(); g.arc(x,y,s,0,TAU); g.stroke(); }
  gr=g.createRadialGradient(0,0,R*0.7,0,0,R); gr.addColorStop(0,'rgba(120,110,170,0)'); gr.addColorStop(1,'rgba(120,110,170,0.3)');
  g.fillStyle=gr; g.fillRect(-R,-R,R*2,R*2); g.restore();
  return c;
}
function auroraStrip(k){
  const c=U.buf('w_astrip'+k,1,128); if(c._ok) return c; c._ok=1;
  const g=c.getContext('2d'); const gr=g.createLinearGradient(0,128,0,0);
  const A=k===1?['#b48cff','#7f8cff','#ff9ad6']:['#5ef2c9','#3fb8e0','#b48cff'];
  gr.addColorStop(0,U.rgba(A[0],0)); gr.addColorStop(0.05,U.rgba(A[0],1)); gr.addColorStop(0.12,U.rgba(A[0],0.85));
  gr.addColorStop(0.45,U.rgba(A[1],0.45)); gr.addColorStop(0.75,U.rgba(A[2],0.2)); gr.addColorStop(1,U.rgba(A[2],0));
  g.fillStyle=gr; g.fillRect(0,0,1,128); return c;
}
const CURTAINS=[
  {cy:-300,h:480,seed:11,k:0,amp:1.0},
  {cy:-120,h:340,seed:23,k:0,amp:0.75},
  {cy:-520,h:420,seed:37,k:1,amp:0.6},
];

// ---------------------------------------------------------------- SKY
W.sky = function(ctx, cam, t, o={}){
  const aur = o.aurora===undefined?0.7:o.aurora, boost=o.starBoost||0;
  ctx.save(); ctx.setTransform(1,0,0,1,0,0);
  const {z,ox,oy}=skyTf(cam); const hY=horizonY(cam), hz=1+(cam.zoom-1)*HP;
  // --- half-res base: gradient + milky way + aurora + horizon haze, upscaled once
  const BW2=960, BH2=540; const base=U.buf('w_skybase',BW2,BH2); const b=base.getContext('2d');
  b.setTransform(0.5,0,0,0.5,0,0); b.globalAlpha=1; b.globalCompositeOperation='source-over';
  const top=hY-1500*hz;
  const gr=b.createLinearGradient(0,top,0,hY+30);
  gr.addColorStop(0,'#070a24'); gr.addColorStop(0.45,'#0e1340'); gr.addColorStop(0.68,'#1c1d58');
  gr.addColorStop(0.82,'#2b2366'); gr.addColorStop(0.93,'#5a3b7a'); gr.addColorStop(1,'#8a5a8c');
  b.fillStyle=gr; b.fillRect(0,0,1920,1080);
  b.globalCompositeOperation='lighter'; b.globalAlpha=0.85;
  blitClip(b,milkyBuf(),ox+STAR_X0*z,oy+STAR_Y0*z,z*2);
  b.globalAlpha=1;
  if(aur>0.01) drawAurora(b,t,z,ox,oy,aur);
  b.globalCompositeOperation='source-over';
  ctx.imageSmoothingEnabled=false; ctx.drawImage(base,0,0,1920,1080); ctx.imageSmoothingEnabled=true;
  // --- static stars (batched)
  ctx.globalCompositeOperation='lighter';
  const ylim=hY+10; const sa=0.92+boost*0.08; ctx.globalAlpha=sa;
  for(const G of starList()){ ctx.fillStyle=G.style; const P=G.pts;
    for(let i=0;i<P.length;i+=3){ const x=ox+P[i]*z, y=oy+P[i+1]*z; if(x<0||x>1920||y<0||y>ylim) continue; const rr=P[i+2]*z; ctx.fillRect(x-rr,y-rr,rr*2,rr*2); } }
  // --- twinkling stars
  const stars=twinkleStars(), spike=spikeSprite(); const hole=o.meteorHole;
  for(const s of stars){
    const x=ox+s.x*z, y=oy+s.y*z;
    if(x<-40||x>1960||y<-40||y>hY+20) continue;
    if(hole){ const dx=x-hole.x, dy=y-hole.y; if(dx*dx+dy*dy<hole.r*hole.r) continue; }
    let a=s.a*(0.55+0.45*Math.sin(t*s.f+s.ph))*(0.85+0.15*Math.sin(t*s.f*2.3+s.ph*1.7));
    a=Math.min(1,a*(1+boost*0.7));
    const rad=s.rad*z*(1+boost*0.25);
    if(s.glint){
      const gs=rad*7; ctx.globalAlpha=a; ctx.drawImage(glowSprite(s.col),x-gs,y-gs,gs*2,gs*2);
      const L=s.glint*(0.75+0.35*Math.sin(t*s.f*1.3+s.ph))*z*(1+boost*0.5);
      ctx.globalAlpha=a*0.85; ctx.drawImage(spike,x-L,y-L,L*2,L*2);
      ctx.fillStyle='#ffffff'; ctx.globalAlpha=1; ctx.beginPath(); ctx.arc(x,y,rad*0.7,0,TAU); ctx.fill();
    } else if(rad<1.25){
      ctx.globalAlpha=a; ctx.fillStyle=s.col; ctx.fillRect(x-rad*0.8,y-rad*0.8,rad*1.6,rad*1.6);
    } else {
      const gs=rad*4.5; ctx.globalAlpha=a*0.9; ctx.drawImage(glowSprite(s.col),x-gs,y-gs,gs*2,gs*2);
    }
  }
  ctx.globalAlpha=1; ctx.globalCompositeOperation='source-over';
  // --- horizon haze (fades stars into the glow); only the band above the horizon
  const hz0=Math.max(0,hY-380*hz);
  if(hz0<1080){ const hg=ctx.createLinearGradient(0,hY-380*hz,0,hY+10);
    hg.addColorStop(0,'rgba(138,90,140,0)'); hg.addColorStop(0.6,'rgba(110,72,130,0.35)'); hg.addColorStop(1,'rgba(150,100,150,0.75)');
    ctx.fillStyle=hg; ctx.fillRect(0,hz0,1920,Math.min(1080,hY+10)-hz0); }
  // --- moon (halo baked into sprite)
  const mx=ox+MOON.x*z, my=oy+MOON.y*z;
  const ms=moonSprite(); const msz=ms.width/3*z*(1+0.015*Math.sin(t*0.7)); ctx.drawImage(ms,mx-msz,my-msz,msz*2,msz*2);
  ctx.restore();
};

function drawAurora(ctx,t,z,ox,oy,aur){
  const AW=480, AH=270; const ab=U.buf('w_aur',AW,AH), ab2=U.buf('w_aur2',AW,AH);
  const g=ab.getContext('2d'); g.setTransform(1,0,0,1,0,0); g.globalCompositeOperation='source-over'; g.globalAlpha=1; g.clearRect(0,0,AW,AH);
  g.globalCompositeOperation='lighter';
  const baseF=(C,X)=>C.cy+170*U.fbm(X*0.0008+t*0.018,C.seed)+60*Math.sin(X*0.0038+t*0.28+C.seed)+16*Math.sin(X*0.013-t*0.6+C.seed);
  for(const C of CURTAINS){
    const strip=auroraStrip(C.k);
    let prev=baseF(C,(-4-ox)/z);
    for(let c=0;c<AW;c+=2){
      const sx=c*4+4, X=(sx-ox)/z;
      const base=baseF(C,X); const slope=Math.abs(base-prev)/(8/z); prev=base;
      let e=U.fbm(X*0.0008-t*0.02,C.seed+5)*1.1+0.62; e=Math.pow(U.smooth(U.clamp(e)),1.4);
      if(e<0.02) continue;
      const ray=0.6+0.28*U.noise(X*0.035+t*0.4,C.seed+9)+0.2*U.noise(X*0.11-t*0.8,C.seed+13);
      const a=e*U.clamp(ray)*C.amp*(1+Math.min(1.2,slope*1.5)); if(a<0.02) continue;
      const hh=C.h*(0.5+0.7*(U.noise(X*0.0035+t*0.05,C.seed+3)+1)/2)*(0.55+0.45*e);
      const by=(oy+base*z)/4, ty=(oy+(base-hh)*z)/4;
      if(by<0||ty>AH) continue;
      g.globalAlpha=Math.min(1,a*0.9); g.drawImage(strip,0,0,1,128,c,ty,2.1,by-ty);
    }
  }
  const g2=ab2.getContext('2d'); g2.setTransform(1,0,0,1,0,0); g2.globalAlpha=1; g2.globalCompositeOperation='source-over';
  g2.clearRect(0,0,AW,AH); g2.filter='blur(4px)'; g2.drawImage(ab,0,0);
  ctx.globalCompositeOperation='lighter';
  g2.filter='none'; g2.globalCompositeOperation='lighter'; g2.globalAlpha=0.6; g2.drawImage(ab,0,0); g2.globalAlpha=1; g2.globalCompositeOperation='source-over';
  ctx.globalAlpha=aur; ctx.drawImage(ab2,0,0,1920,1080);
  ctx.globalAlpha=1;
}

// ---------------------------------------------------------------- CLOUDS
const HAZE='#8a5a8c';
const LAYERS={
  far:  {p:0.30,y:40, drift:5, w:2048,h:400,pad:120,rows:5,gap:26,rMin:22,rMax:55, seed:101, haze:0.5, light:0.6},
  mid:  {p:0.55,y:110,drift:9, w:2200,h:520,pad:180,rows:5,gap:36,rMin:34,rMax:85, seed:202, haze:0.3, light:0.75},
  near: {p:0.80,y:190,drift:14,w:2400,h:600,pad:230,rows:4,gap:48,rMin:48,rMax:115,seed:303, haze:0.14, light:0.85},
  front:{p:1.00,y:350,drift:20,w:2600,h:700,pad:280,rows:5,gap:56,rMin:60,rMax:140,seed:404, haze:0.0, light:0.9},
};
function cloudStrip(name,L){
  const c=U.buf('w_cl_'+name,L.w,L.h); if(c._ok) return c; c._ok=1;
  const g=c.getContext('2d'); const r=U.rng(L.seed);
  const hz=k=>U.mixHex(k,HAZE,L.haze);
  const TOP=hz('#d4daf6'), MID=hz('#7f86c4'), SH=hz('#3a3f7d'), DEEP=hz('#1d2156'), RIM=hz('#f4f2ff');
  const circ=(x,y,rad)=>{ for(const dx of [-L.w,0,L.w]){ const X=x+dx; if(X+rad<0||X-rad>L.w) continue; g.moveTo(X+rad,y); g.arc(X,y,rad,0,TAU);} };
  for(let row=0;row<L.rows;row++){
    const ry=L.pad+row*L.gap; const sc=1-row*0.1; const puffs=[]; let x=r()*40;
    while(x<L.w){ const R=U.lerp(L.rMin,L.rMax,Math.pow(r(),1.5))*sc;
      const dome=R*(0.15+0.45*(U.noise(x*0.003,L.seed+row)+1)/2);
      puffs.push([x,ry-dome,R]);
      const n=2+((r()*3)|0);
      for(let k=0;k<n;k++){ const a=-Math.PI*(0.15+0.7*r()); const rr=R*(0.3+0.25*r()); puffs.push([x+Math.cos(a)*R*0.72,ry-dome+Math.sin(a)*R*0.62,rr]); }
      x+=R*(0.9+0.7*r()); }
    puffs.sort((a,b)=>(a[1]+a[2]*0.3)-(b[1]+b[2]*0.3));
    const lt=L.light*Math.max(0.3,1-row*0.19);
    // rim silhouette
    g.beginPath(); for(const [px,py,R] of puffs) circ(px-R*0.03,py-R*0.05,R); g.fillStyle=U.rgba(RIM,0.85*lt); g.fill();
    // body silhouette with vertical shading
    const top=ry-L.rMax*1.4;
    const bg=g.createLinearGradient(0,top,0,ry+L.gap*1.2);
    bg.addColorStop(0,U.mixHex(MID,TOP,0.35*lt)); bg.addColorStop(0.55,MID); bg.addColorStop(0.85,SH); bg.addColorStop(1,DEEP);
    g.fillStyle=bg; g.beginPath(); for(const [px,py,R] of puffs) circ(px,py,R*0.985); g.fill();
    g.fillRect(0,ry,L.w,L.h-ry);
    // soft lobes: shadow pocket lower-right, moonlit cap upper-left
    for(const [px,py,R] of puffs){ for(const dx of [-L.w,0,L.w]){ const X=px+dx; if(X+R<0||X-R>L.w) continue;
      let gr=g.createRadialGradient(X+R*0.35,py+R*0.45,0,X+R*0.35,py+R*0.45,R*0.95);
      gr.addColorStop(0,U.rgba(SH,0.5)); gr.addColorStop(1,U.rgba(SH,0));
      g.fillStyle=gr; g.beginPath(); g.arc(X,py,R*0.98,0,TAU); g.fill();
      gr=g.createRadialGradient(X-R*0.3,py-R*0.42,0,X-R*0.3,py-R*0.42,R*0.85);
      gr.addColorStop(0,U.rgba(TOP,0.85*lt)); gr.addColorStop(0.45,U.rgba(TOP,0.35*lt)); gr.addColorStop(1,U.rgba(TOP,0));
      g.fillStyle=gr; g.beginPath(); g.arc(X,py,R*0.98,0,TAU); g.fill(); } }
    // occlusion below row
    const og=g.createLinearGradient(0,ry,0,ry+L.gap*1.5);
    og.addColorStop(0,U.rgba(DEEP,0)); og.addColorStop(1,U.rgba(DEEP,0.6));
    g.fillStyle=og; g.fillRect(0,ry,L.w,L.gap*1.5);
  }
  const fg=g.createLinearGradient(0,L.pad+L.rows*L.gap,0,L.h);
  fg.addColorStop(0,U.rgba(DEEP,0)); fg.addColorStop(1,U.rgba(DEEP,1));
  g.fillStyle=fg; g.fillRect(0,L.pad+L.rows*L.gap,L.w,L.h);
  return c;
}
function drawCloudLayer(ctx,cam,t,name){
  const L=LAYERS[name]; const buf=cloudStrip(name,L);
  const z=1+(cam.zoom-1)*L.p;
  const sy=540+(L.y-cam.y*L.p)*z-L.pad*z;
  if(sy>1080) return null;
  const tw=L.w*z, th=L.h*z;
  let x0=960+(-cam.x*L.p - t*L.drift - 3000)*z; x0=((x0%tw)+tw)%tw-tw; x0=Math.floor(x0);
  const yA=Math.max(0,-sy/z);
  for(let x=x0;x<1920;x+=tw){
    if(yA<L.h) ctx.drawImage(buf,0,yA,L.w,L.h-yA,x,sy+yA*z,tw+1,th-yA*z);
  }
  const by=sy+th; const deep=U.mixHex('#1d2156',HAZE,L.haze);
  if(by<1080){ ctx.fillStyle=deep; ctx.fillRect(0,Math.max(0,by-1),1920,1080-Math.max(0,by-1)); }
  return {top:sy+L.pad*z, z};
}
W.cloudsBack = function(ctx,cam,t){
  ctx.save(); ctx.setTransform(1,0,0,1,0,0);
  const {z,ox}=skyTf(cam); const mx=ox+MOON.x*z;
  const far=drawCloudLayer(ctx,cam,t,'far');
  if(far){ // moon reflection sheen on far cloud tops
    ctx.globalCompositeOperation='lighter';
    ctx.save(); ctx.translate(mx,far.top); ctx.scale(1,0.22);
    const gr=ctx.createRadialGradient(0,0,0,0,0,700); gr.addColorStop(0,'rgba(220,215,255,0.22)'); gr.addColorStop(1,'rgba(200,200,255,0)');
    ctx.fillStyle=gr; ctx.fillRect(-700,-700,1400,1400); ctx.restore();
    ctx.globalCompositeOperation='source-over';
  }
  for(const n of ['mid','near']){
    const r=drawCloudLayer(ctx,cam,t,n); if(!r) continue;
  }
  ctx.restore();
};
W.cloudsFront = function(ctx,cam,t){
  ctx.save(); ctx.setTransform(1,0,0,1,0,0);
  drawCloudLayer(ctx,cam,t,'front');
  ctx.restore();
};

// ---------------------------------------------------------------- ISLAND
const IB={x0:-800,y0:-70,w:1600,h:660,S:2};
const islTop = x => W.groundY(x);
function islBot(x){
  const u=(x-10)/735; const base=Math.max(0,1-u*u);
  let y=30+400*Math.pow(base,1.4)+90*Math.exp(-Math.pow((x-30)/95,2));
  y+=50*Math.exp(-Math.pow((x+330)/70,2))+38*Math.exp(-Math.pow((x-390)/60,2))+25*Math.exp(-Math.pow((x+560)/45,2));
  y+=16*U.noise(x*0.03,7)+7*U.noise(x*0.11,8)+3*U.noise(x*0.4,9);
  return Math.max(islTop(x)+20,y);
}
function islandPath(g){
  g.beginPath(); g.moveTo(-735,islTop(-735)+28);
  for(let x=-732;x<=732;x+=4) g.lineTo(x,islTop(x)-2);
  for(let x=735;x>=-735;x-=4) g.lineTo(x,islBot(x));
  g.closePath();
}
function islandBuf(){
  const c=U.buf('w_island',IB.w*IB.S,IB.h*IB.S); if(c._ok) return c; c._ok=1;
  const g=c.getContext('2d'); g.scale(IB.S,IB.S); g.translate(-IB.x0,-IB.y0); islandArt(g); return c;
}
// hi-res band of the island top (grass edge, path, fence, flowers' ground) for extreme close-ups
const IH={x0:-780,y0:-80,w:1560,h:190,S:6};
function islandHiBuf(){
  const c=U.buf('w_islandHi',IH.w*IH.S,IH.h*IH.S); if(c._ok) return c; c._ok=1;
  const g=c.getContext('2d'); g.scale(IH.S,IH.S); g.translate(-IH.x0,-IH.y0); islandArt(g); return c;
}
function islandArt(g){
  const r=U.rng(555);
  // rock
  islandPath(g);
  let gr=g.createLinearGradient(-700,0,500,520);
  gr.addColorStop(0,'#4f4880'); gr.addColorStop(0.35,'#3b3560'); gr.addColorStop(0.7,'#2a2548'); gr.addColorStop(1,'#1d1938');
  g.fillStyle=gr; g.fill();
  g.save(); islandPath(g); g.clip();
  // strata
  for(let k=0;k<15;k++){
    const y0=30+k*34+r()*8;
    g.beginPath();
    for(let x=-740;x<=740;x+=10){ const y=y0+12*U.noise(x*0.008,k+20)+4*U.noise(x*0.05,k+40)+x*0.02; x===-740?g.moveTo(x,y):g.lineTo(x,y); }
    g.strokeStyle='rgba(14,10,32,0.5)'; g.lineWidth=2.2; g.stroke();
    g.save(); g.translate(0,3); g.strokeStyle='rgba(150,140,210,0.12)'; g.lineWidth=1.5; g.stroke(); g.restore();
    if(k%2){ g.lineTo(740,y0+60); g.lineTo(-740,y0+60); g.closePath(); g.fillStyle='rgba(20,16,44,0.12)'; g.fill(); }
  }
  // facets
  for(let i=0;i<70;i++){
    const x=-700+r()*1400, y=islTop(x)+40+r()*(islBot(x)-islTop(x)-40); const s=14+r()*36;
    g.beginPath(); g.moveTo(x,y); g.lineTo(x+s*(0.6+r()*0.6),y+s*(0.2+r()*0.4)); g.lineTo(x+s*(0.1+r()*0.4),y+s*(0.8+r()*0.6)); g.closePath();
    g.fillStyle=x<0?'rgba(150,140,220,0.08)':'rgba(10,8,30,0.14)'; g.fill();
    g.strokeStyle='rgba(12,10,28,0.25)'; g.lineWidth=1; g.beginPath(); g.moveTo(x,y); g.lineTo(x+s*(0.1+r()*0.4),y+s*(0.8+r()*0.6)); g.stroke();
  }
  // speckles & pebbles embedded in rock
  for(let i=0;i<420;i++){ const x=-720+r()*1440, y=islTop(x)+30+r()*(islBot(x)-islTop(x)-30); const s=0.8+Math.pow(r(),3)*5;
    g.fillStyle=r()<0.5?'rgba(160,150,230,0.14)':'rgba(10,8,28,0.25)'; g.beginPath(); g.ellipse(x,y,s*1.3,s,r()*3,0,TAU); g.fill(); }
  for(let i=0;i<40;i++){ const x=-680+r()*1360, y=islTop(x)+50+r()*(islBot(x)-islTop(x)-60); const s=4+r()*7;
    g.fillStyle='#2a2548'; g.beginPath(); g.ellipse(x,y+1.5,s*1.2,s*0.8,0,0,TAU); g.fill();
    g.fillStyle='#5a5288'; g.beginPath(); g.ellipse(x-0.8,y,s*1.1,s*0.7,0,0,TAU); g.fill();
    g.fillStyle='rgba(200,195,255,0.3)'; g.beginPath(); g.ellipse(x-s*0.4,y-s*0.3,s*0.4,s*0.2,0,0,TAU); g.fill(); }
  // cracks
  g.strokeStyle='rgba(12,9,26,0.45)'; g.lineWidth=1.4;
  for(let i=0;i<22;i++){ let x=-650+r()*1300, y=islTop(x)+50+r()*120; g.beginPath(); g.moveTo(x,y);
    for(let j=0;j<6;j++){ x+=(r()-0.5)*26; y+=10+r()*22; g.lineTo(x,y);} g.stroke(); }
  // shading: right side shade, bottom bounce light from moonlit clouds, left moon rim
  gr=g.createLinearGradient(-600,0,700,0); gr.addColorStop(0,'rgba(10,8,30,0)'); gr.addColorStop(1,'rgba(10,8,30,0.4)');
  g.fillStyle=gr; g.fillRect(-800,-80,1600,700);
  gr=g.createLinearGradient(0,180,0,560); gr.addColorStop(0,'rgba(140,150,230,0)'); gr.addColorStop(1,'rgba(140,150,230,0.28)');
  g.fillStyle=gr; g.fillRect(-800,180,1600,400);
  gr=g.createLinearGradient(0,0,0,90); gr.addColorStop(0,'rgba(10,8,25,0.55)'); gr.addColorStop(1,'rgba(10,8,25,0)');
  g.fillStyle=gr; g.fillRect(-800,-30,1600,130);
  g.restore();
  // rim light on the lower-left silhouette edge (cloud bounce)
  g.save(); g.beginPath(); for(let x=-735;x<=200;x+=4){ const y=islBot(x); x===-735?g.moveTo(x,y):g.lineTo(x,y);}
  g.strokeStyle='rgba(190,195,255,0.35)'; g.lineWidth=2.5; g.stroke(); g.restore();
  // moss patches hanging under the grass
  for(let i=0;i<34;i++){ const x=-700+r()*1400, y=islTop(x)+24+r()*16; g.fillStyle=U.rgba(r()<0.5?'#2f6b5a':'#24574a',0.35+r()*0.25);
    g.beginPath(); g.ellipse(x,y,5+r()*12,2+r()*3,0,0,TAU); g.fill(); }
  // soil band
  g.beginPath(); g.moveTo(-735,islTop(-735)+28);
  for(let x=-732;x<=732;x+=4) g.lineTo(x,islTop(x));
  for(let x=732;x>=-732;x-=4) g.lineTo(x,islTop(x)+26+6*U.noise(x*0.05,3));
  g.closePath(); g.fillStyle='#2b2034'; g.fill();
  // grass cap with drips
  g.beginPath(); g.moveTo(-740,islTop(-740)+30);
  for(let x=-738;x<=738;x+=3) g.lineTo(x,islTop(x)-3);
  for(let x=738;x>=-738;x-=3){ const drip=Math.max(0,Math.sin(x*0.09+3*U.noise(x*0.02,4)))**3*12; g.lineTo(x,islTop(x)+11+5*U.noise(x*0.07,5)+drip); }
  g.closePath();
  gr=g.createLinearGradient(0,-30,0,30); gr.addColorStop(0,'#5aa77f'); gr.addColorStop(0.35,'#3d8466'); gr.addColorStop(0.7,'#2f6b5a'); gr.addColorStop(1,'#1f4b42');
  g.fillStyle=gr; g.fill();
  g.strokeStyle='rgba(150,220,180,0.35)'; g.lineWidth=1.5; g.beginPath(); for(let x=-730;x<=730;x+=4){const y=islTop(x)-2; x===-730?g.moveTo(x,y):g.lineTo(x,y);} g.stroke();
  // stone path to the door
  for(let i=0;i<9;i++){
    const x=-70+i*34+(r()-0.5)*8; const y=islTop(x)+2; const w=13+r()*5;
    g.fillStyle='#2a2440'; g.beginPath(); g.ellipse(x,y+1.5,w,4.2,0,0,TAU); g.fill();
    g.fillStyle='#8e88b4'; g.beginPath(); g.ellipse(x,y,w,3.6,0,0,TAU); g.fill();
    g.fillStyle='rgba(220,215,255,0.5)'; g.beginPath(); g.ellipse(x-w*0.25,y-1.2,w*0.5,1.2,0,0,TAU); g.fill();
  }
  // boulders
  const boulder=(x,s)=>{ const y=islTop(x)+4; g.fillStyle='#2a2548'; g.beginPath(); g.ellipse(x,y-s*0.45,s,s*0.62,0,Math.PI,0); g.lineTo(x+s,y); g.closePath(); g.fill();
    const bg=g.createRadialGradient(x-s*0.4,y-s*0.7,0,x,y-s*0.3,s); bg.addColorStop(0,'#7d76a8'); bg.addColorStop(1,'#3b3560');
    g.fillStyle=bg; g.beginPath(); g.ellipse(x-1,y-s*0.47,s*0.96,s*0.58,0,Math.PI,0); g.closePath(); g.fill(); };
  boulder(560,34); boulder(600,20); boulder(-300,16); boulder(-680,22);
  // barrel and crate next to the lighthouse
  { const x=420,y=islTop(420)+2; g.fillStyle='#4a3030'; g.fillRect(x-17,y-42,34,42);
    const bg=g.createLinearGradient(x-17,0,x+17,0); bg.addColorStop(0,'#8a5a48'); bg.addColorStop(0.4,'#6b4436'); bg.addColorStop(1,'#3a2626');
    g.fillStyle=bg; g.beginPath(); g.moveTo(x-15,y); g.quadraticCurveTo(x-20,y-21,x-15,y-42); g.lineTo(x+15,y-42); g.quadraticCurveTo(x+20,y-21,x+15,y); g.closePath(); g.fill();
    g.strokeStyle='#2e2a44'; g.lineWidth=2.5; for(const yy of [y-8,y-34]){ g.beginPath(); g.moveTo(x-17,yy); g.lineTo(x+17,yy); g.stroke(); }
    g.fillStyle='#4a3a36'; g.beginPath(); g.ellipse(x,y-42,15,3.5,0,0,TAU); g.fill();
    const cx=452; g.fillStyle='#5a4038'; g.fillRect(cx,y-26,28,26); g.strokeStyle='#3a2a28'; g.lineWidth=2; g.strokeRect(cx+1,y-25,26,24);
    g.beginPath(); g.moveTo(cx+1,y-25); g.lineTo(cx+27,y-1); g.stroke(); g.fillStyle='rgba(200,190,255,0.15)'; g.fillRect(cx,y-26,28,3); }
  // fence
  const fx=[-630,-590,-550,-510,-470,-430];
  const post=(x,lean)=>{ const y=islTop(x)+4; g.save(); g.translate(x,y); g.rotate(lean);
    g.fillStyle='#4a332e'; g.fillRect(-4,-46,8,46); g.fillStyle='#7a5446'; g.fillRect(-4,-46,3.5,46);
    g.fillStyle='#4a332e'; g.beginPath(); g.moveTo(-4,-46); g.lineTo(0,-53); g.lineTo(4,-46); g.fill(); g.restore(); };
  for(let i=0;i<fx.length;i++) post(fx[i],(r()-0.5)*0.12);
  for(const hh of [16,34]){ g.beginPath(); for(let i=0;i<fx.length;i++){ const y=islTop(fx[i])+4-hh; i?g.lineTo(fx[i],y):g.moveTo(fx[i]-6,y);} g.lineTo(fx[fx.length-1]+6,islTop(fx[fx.length-1])+4-hh);
    g.strokeStyle='#3e2a26'; g.lineWidth=6; g.stroke(); g.strokeStyle='#6e4c40'; g.lineWidth=2.5; g.save(); g.translate(0,-1.5); g.stroke(); g.restore(); }
}
let ISL=null;
function islandData(){
  if(ISL) return ISL; const r=U.rng(8080); ISL={blades:[],roots:[],vines:[],flowers:[],rocks:[]};
  for(let i=0;i<760;i++){ const x=-715+r()*1430; ISL.blades.push({x,h:8+Math.pow(r(),1.6)*22,w:1.4+r()*1.6,tone:(r()*3)|0,ph:r()*TAU}); }
  ISL.blades.sort((a,b)=>a.tone-b.tone);
  for(let i=0;i<20;i++){ const x=-560+r()*1100; ISL.roots.push({x,len:50+r()*190,ph:r()*TAU,w:1.5+r()*2.5,curl:(r()-0.5)*40}); }
  const vx=[-722,-700,-668,-630,-410,-200,150,470,640,676,706,724];
  for(const x of vx) ISL.vines.push({x:x+(r()-0.5)*10,len:60+r()*170,ph:r()*TAU,leaf:r()});
  const fxs=[-600,-575,-520,-410,-372,-330,-262,-228,-195,-90,105,132,388,462,500,520,615,655];
  const cols=['#ffb3c7','#c9b3ff','#fff4f0','#ffe08a','#9fe7ff'];
  for(const x of fxs) ISL.flowers.push({x:x+(r()-0.5)*8,h:14+r()*14,col:cols[(r()*cols.length)|0],ph:r()*TAU,s:0.8+r()*0.5});
  const rk=[[-850,170,26],[-770,390,14],[860,110,30],[780,330,17],[-430,640,12],[430,610,18],[930,430,10],[-960,40,9]];
  rk.forEach(([x,y,s],i)=>ISL.rocks.push({x,y,s,ph:r()*TAU,i}));
  return ISL;
}
function rockSprite(i,s){
  const sz=Math.ceil(s*2.6*2); const c=U.buf('w_rock'+i,sz,Math.ceil(sz*1.2)); if(c._ok) return c; c._ok=1;
  const g=c.getContext('2d'); g.scale(2,2); g.translate(s*1.3,s*0.9); const r=U.rng(600+i);
  const n=7; g.beginPath(); g.moveTo(-s,0);
  for(let k=0;k<=n;k++){ const a=Math.PI*(k/n); g.lineTo(-Math.cos(a)*s*(0.9+r()*0.2), Math.sin(a)*s*(1.2+r()*0.8)); }
  g.lineTo(s,0); g.quadraticCurveTo(0,-s*0.35,-s,0); g.closePath();
  const gr=g.createLinearGradient(-s,-s*0.3,s,s*2); gr.addColorStop(0,'#5a5290'); gr.addColorStop(0.5,'#3b3560'); gr.addColorStop(1,'#1f1b3a');
  g.fillStyle=gr; g.fill();
  if(s>15){ g.fillStyle='#3d8466'; g.beginPath(); g.ellipse(0,-s*0.12,s*0.95,s*0.22,0,0,TAU); g.fill(); g.fillStyle='#5aa77f'; g.beginPath(); g.ellipse(-s*0.2,-s*0.2,s*0.6,s*0.1,0,0,TAU); g.fill(); }
  else { g.fillStyle='rgba(180,175,240,0.4)'; g.beginPath(); g.ellipse(-s*0.2,-s*0.12,s*0.6,s*0.12,0,0,TAU); g.fill(); }
  return c;
}
function wind(x,t){ return 0.18+0.22*Math.sin(t*1.3+x*0.011)+0.14*U.noise(t*0.6+x*0.004,77); }

W.island = function(ctx,t,o={}){
  const bloom=o.bloom||0; const D=islandData();
  ctx.save();
  // floating rocks
  for(const R of D.rocks){ const sp=rockSprite(R.i,R.s); const y=R.y+9*Math.sin(t*0.55+R.ph), x=R.x+4*Math.sin(t*0.3+R.ph);
    ctx.save(); ctx.translate(x,y); ctx.rotate(0.06*Math.sin(t*0.4+R.ph)); ctx.drawImage(sp,-R.s*1.3,-R.s*0.9,sp.width/2,sp.height/2); ctx.restore(); }
  // roots (behind body bottom edge, hanging below)
  ctx.lineCap='round';
  for(const R of D.roots){ const y0=islBot(R.x)-8; const sw=Math.sin(t*0.7+R.ph)*0.06+wind(R.x,t)*0.04;
    const ex=R.x+R.curl+Math.sin(sw)*R.len, ey=y0+R.len*Math.cos(sw);
    ctx.strokeStyle='#241d38'; ctx.lineWidth=R.w; ctx.beginPath(); ctx.moveTo(R.x,y0);
    ctx.bezierCurveTo(R.x+R.curl*0.2,y0+R.len*0.4,ex-R.curl*0.5,ey-R.len*0.3,ex,ey); ctx.stroke();
    ctx.strokeStyle='rgba(150,150,230,0.25)'; ctx.lineWidth=R.w*0.4; ctx.stroke(); }
  // body
  const tf=ctx.getTransform(); const zs=Math.hypot(tf.a,tf.b);
  ctx.drawImage(islandBuf(),IB.x0,IB.y0,IB.w,IB.h);
  if(zs>2.4){ // overlay the crisp band, only its visible part
    const inv=tf.inverse(); const c=[[0,0],[1920,0],[0,1080],[1920,1080]].map(([x,y])=>[inv.a*x+inv.c*y+inv.e, inv.b*x+inv.d*y+inv.f]);
    const wx0=Math.max(IH.x0,Math.min(...c.map(p=>p[0]))-2), wx1=Math.min(IH.x0+IH.w,Math.max(...c.map(p=>p[0]))+2);
    const wy0=Math.max(IH.y0,Math.min(...c.map(p=>p[1]))-2), wy1=Math.min(IH.y0+IH.h,Math.max(...c.map(p=>p[1]))+2);
    if(wx1>wx0&&wy1>wy0){ const hb=islandHiBuf(), S=IH.S;
      ctx.drawImage(hb,(wx0-IH.x0)*S,(wy0-IH.y0)*S,(wx1-wx0)*S,(wy1-wy0)*S,wx0,wy0,wx1-wx0,wy1-wy0); } }
  // vines over the rock face
  for(const V of D.vines){ const y0=islTop(V.x)+10; const sw=0.05*Math.sin(t*0.9+V.ph)+wind(V.x,t)*0.05;
    const pts=[]; for(let k=0;k<=10;k++){ const s=k/10; pts.push([V.x+Math.sin(sw*s*s*3)*V.len*s*0.4+6*Math.sin(s*5+V.ph),y0+V.len*s]); }
    ctx.strokeStyle='#244d40'; ctx.lineWidth=2; ctx.beginPath(); pts.forEach((p,k)=>k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])); ctx.stroke();
    for(let k=1;k<=10;k++){ const p=pts[k], side=k%2?1:-1; ctx.fillStyle=k%3?'#3d8466':'#2f6b5a';
      ctx.beginPath(); ctx.ellipse(p[0]+side*4,p[1],5,2.6,side*0.6+sw,0,TAU); ctx.fill(); }
    if(V.leaf>0.6){ const p=pts[10]; ctx.fillStyle=U.rgba('#ffd0e8',0.8); ctx.beginPath(); ctx.arc(p[0],p[1]+3,2.8,0,TAU); ctx.fill(); }
  }
  // grass blades
  const tones=['#22524a','#2f6b5a','#4f9a74'];
  let cur=-1;
  for(let i=0;i<D.blades.length;i++){ const B=D.blades[i];
    if(B.tone!==cur){ if(cur>=0) ctx.fill(); cur=B.tone; ctx.fillStyle=tones[cur]; ctx.beginPath(); }
    const gy=islTop(B.x)+2; const L=wind(B.x,t)+0.08*Math.sin(t*2.7+B.ph);
    const tx=B.x+B.h*Math.sin(L), ty=gy-B.h*Math.cos(L), cx=B.x+B.h*0.45*Math.sin(L*0.5), cy=gy-B.h*0.55;
    ctx.moveTo(B.x-B.w,gy); ctx.quadraticCurveTo(cx-B.w*0.4,cy,tx,ty); ctx.quadraticCurveTo(cx+B.w*0.4,cy,B.x+B.w,gy); }
  if(cur>=0) ctx.fill();
  // flowers
  for(const F of D.flowers){ const gy=islTop(F.x)+2; const L=wind(F.x,t)*0.8+0.05*Math.sin(t*2+F.ph);
    const hx=F.x+F.h*Math.sin(L), hy=gy-F.h*Math.cos(L);
    ctx.strokeStyle='#2f6b5a'; ctx.lineWidth=1.6; ctx.beginPath(); ctx.moveTo(F.x,gy); ctx.quadraticCurveTo(F.x,gy-F.h*0.5,hx,hy); ctx.stroke();
    ctx.fillStyle='#3d8466'; ctx.beginPath(); ctx.ellipse(F.x+3,gy-F.h*0.35,4,1.8,-0.5,0,TAU); ctx.fill();
    const open=0.55+0.45*bloom, pr=3.2*F.s*open;
    if(bloom>0.01){ ctx.globalCompositeOperation='lighter'; const gs=pr*9*bloom*(0.9+0.1*Math.sin(t*3+F.ph));
      ctx.globalAlpha=0.8*bloom; ctx.drawImage(glowSprite(F.col),hx-gs,hy-gs,gs*2,gs*2); ctx.globalAlpha=1; ctx.globalCompositeOperation='source-over'; }
    ctx.fillStyle=bloom>0.01?U.mixHex(F.col,'#ffffff',bloom*0.4):F.col;
    for(let k=0;k<5;k++){ const a=k*TAU/5+L+t*0.1*bloom; ctx.beginPath(); ctx.ellipse(hx+Math.cos(a)*pr*0.9,hy+Math.sin(a)*pr*0.9*open,pr*0.75,pr*0.5,a,0,TAU); ctx.fill(); }
    ctx.fillStyle=bloom>0.01?'#fff6c8':'#ffd27a'; ctx.beginPath(); ctx.arc(hx,hy,pr*0.45,0,TAU); ctx.fill();
  }
  ctx.restore();
};

// ---------------------------------------------------------------- LIGHTHOUSE
const LB={x0:110,y0:-1020,w:300,h:1040,S:3};
function towerHW(y){ const k=(LH.baseY-26-y)/(LH.baseY-26-LH.towerTop); return U.lerp(98,62,k); }
function archPath(g,cx,bottom,w,h){ g.beginPath(); g.moveTo(cx-w/2,bottom); g.lineTo(cx-w/2,bottom-h+w/2); g.arc(cx,bottom-h+w/2,w/2,Math.PI,0); g.lineTo(cx+w/2,bottom); g.closePath(); }
function hShade(g,cx,hw,cols){ const gr=g.createLinearGradient(cx-hw,0,cx+hw,0); cols.forEach((c,i)=>gr.addColorStop(i/(cols.length-1),c)); return gr; }
const WALL=['#a9a3cc','#e9e4f2','#ddd7ec','#b9b3d6','#7d77a6','#5d5888'];
const REDB=['#a8404c','#e0606a','#c8434f','#a83a48','#7e2a3a','#5e2032'];
const WINDOWS=[[246,-232],[274,-450],[250,-650]];
function lighthouseBuf(){
  const c=U.buf('w_lh',LB.w*LB.S,LB.h*LB.S); if(c._ok) return c; c._ok=1;
  const g=c.getContext('2d'); g.scale(LB.S,LB.S); g.translate(-LB.x0,-LB.y0);
  const X=LH.x, B=LH.baseY, TT=LH.towerTop; const r=U.rng(1234);
  // tower body
  const tower=()=>{ g.beginPath(); g.moveTo(X-98,B-26); g.lineTo(X-62,TT); g.lineTo(X+62,TT); g.lineTo(X+98,B-26); g.closePath(); };
  tower(); g.fillStyle=hShade(g,X,98,WALL); g.fill();
  g.save(); tower(); g.clip();
  // red bands
  for(const [y0,y1] of [[-600,-515],[-400,-320]]){ g.fillStyle=hShade(g,X,90,REDB); g.fillRect(X-100,y0,200,y1-y0);
    g.fillStyle='rgba(255,220,230,0.15)'; g.fillRect(X-100,y0,200,2); g.fillStyle='rgba(40,10,30,0.25)'; g.fillRect(X-100,y1-2,200,2); }
  g.fillStyle=hShade(g,X,90,REDB); g.fillRect(X-100,TT,200,22);
  // masonry courses (slightly curved for roundness)
  for(let y=B-40;y>TT;y-=24){ const hw=towerHW(y); g.strokeStyle='rgba(60,50,100,0.1)'; g.lineWidth=1; g.beginPath(); g.moveTo(X-hw,y); g.quadraticCurveTo(X,y+5,X+hw,y); g.stroke(); }
  // weathering streaks
  for(let i=0;i<26;i++){ const x=X-80+r()*160, y=TT+r()*700, l=20+r()*80; g.strokeStyle='rgba(70,60,110,0.035)'; g.lineWidth=4+r()*6; g.beginPath(); g.moveTo(x,y); g.lineTo(x+(r()-0.5)*4,y+l); g.stroke(); }
  // AO at base and under gallery
  let gr=g.createLinearGradient(0,B-100,0,B-26); gr.addColorStop(0,'rgba(30,25,60,0)'); gr.addColorStop(1,'rgba(30,25,60,0.4)'); g.fillStyle=gr; g.fillRect(X-100,B-100,200,80);
  gr=g.createLinearGradient(0,TT,0,TT+60); gr.addColorStop(0,'rgba(30,20,50,0.5)'); gr.addColorStop(1,'rgba(30,20,50,0)'); g.fillStyle=gr; g.fillRect(X-100,TT,200,60);
  g.restore();
  // moonlit left rim
  g.strokeStyle='rgba(240,238,255,0.6)'; g.lineWidth=1.5; g.beginPath(); g.moveTo(X-97,B-28); g.lineTo(X-61,TT); g.stroke();
  // windows (frames)
  for(const [wx,wy] of WINDOWS){ g.fillStyle='#c9c3e0'; archPath(g,wx,wy+4,30,46); g.fill(); g.fillStyle='#2e2a44'; archPath(g,wx,wy,22,38); g.fill();
    g.fillStyle='#8a84b0'; g.fillRect(wx-17,wy,34,5); }
  // plinth
  g.beginPath(); g.moveTo(X-118,B+8); g.lineTo(X-110,B-26); g.lineTo(X+110,B-26); g.lineTo(X+118,B+8); g.closePath();
  g.fillStyle=hShade(g,X,118,['#5e5888','#8a84b4','#6e6898','#4a4470','#36305a']); g.fill();
  g.strokeStyle='rgba(30,25,55,0.5)'; g.lineWidth=1.2;
  g.beginPath(); g.moveTo(X-114,B-9); g.lineTo(X+114,B-9); g.stroke();
  for(let i=0;i<10;i++){ const x=X-108+i*23+(i%2)*11; g.beginPath(); g.moveTo(x,B-26); g.lineTo(x,B-9); g.moveTo(x+12,B-9); g.lineTo(x+12,B+8); g.stroke(); }
  g.fillStyle='rgba(230,225,255,0.35)'; g.fillRect(X-110,B-27,220,2);
  // door arch frame (stone)
  g.fillStyle='#7a74a2'; archPath(g,X,B+2,LH.doorW+16,LH.doorH+10); g.fill();
  g.strokeStyle='rgba(40,34,70,0.6)'; g.lineWidth=1.2;
  for(let k=0;k<9;k++){ const a=Math.PI+k*Math.PI/8; const cy=B-LH.doorH+LH.doorW/2; g.beginPath(); g.moveTo(X+Math.cos(a)*LH.doorW/2,cy+Math.sin(a)*LH.doorW/2); g.lineTo(X+Math.cos(a)*(LH.doorW/2+8),cy+Math.sin(a)*(LH.doorW/2+8)); g.stroke(); }
  g.fillStyle='#14101e'; archPath(g,X,B+2,LH.doorW,LH.doorH); g.fill();
  // step
  g.fillStyle='#6e6898'; g.fillRect(X-60,B+2,120,7); g.fillStyle='rgba(230,225,255,0.4)'; g.fillRect(X-60,B+2,120,1.5);
  // corbels under gallery
  for(let i=-5;i<=5;i++){ const x=X+i*12; g.fillStyle=i<0?'#3e3960':'#2e2a44'; g.beginPath(); g.moveTo(x-4,TT); g.lineTo(x+4,TT); g.lineTo(x+1.5,TT+20); g.lineTo(x-1.5,TT+20); g.closePath(); g.fill(); }
  // lamp-room parapet (behind deck top face)
  const GY=LH.galleryY, GH=LH.galleryHalfW, GD=LH.galleryDepth;
  g.fillStyle=hShade(g,X,56,REDB); g.fillRect(X-56,LH.glassBot,112,GY-LH.glassBot);
  g.fillStyle='rgba(255,220,230,0.25)'; g.fillRect(X-56,LH.glassBot,112,2);
  // glass room base dark
  g.fillStyle='#141535'; g.fillRect(X-LH.glassHW,LH.glassTop,LH.glassHW*2,LH.glassBot-LH.glassTop);
  // back railing (along back edge of the deck ellipse)
  railPath(g,true);
  // deck: side face + top ellipse
  g.fillStyle='#211d36'; g.beginPath(); g.ellipse(X,GY+6,GH,GD,0,0,Math.PI); g.lineTo(X-GH,GY); g.ellipse(X,GY,GH,GD,0,Math.PI,0,true); g.closePath(); g.fill();
  g.fillStyle=hShade(g,X,GH,['#3a3558','#2e2a44','#221e38']); g.beginPath(); g.ellipse(X,GY+6,GH,GD,0,0,Math.PI); g.lineTo(X-GH,GY); g.lineTo(X+GH,GY); g.closePath(); g.fill();
  g.fillStyle=hShade(g,X,GH,['#6a6494','#585282','#3e3962']); g.beginPath(); g.ellipse(X,GY,GH,GD,0,0,TAU); g.fill();
  g.fillStyle='rgba(0,0,10,0.35)'; g.beginPath(); g.ellipse(X,GY-1,58,GD*0.6,0,0,TAU); g.fill();
  // re-draw parapet front so it sits on the deck
  g.fillStyle=hShade(g,X,56,REDB); g.fillRect(X-56,LH.glassBot,112,GY-LH.glassBot-2);
  g.fillStyle='rgba(255,220,230,0.25)'; g.fillRect(X-56,LH.glassBot,112,2);
  g.fillStyle='rgba(30,10,30,0.35)'; g.fillRect(X-56,GY-6,112,4);
  // roof
  const RT=LH.glassTop;
  g.fillStyle='#2e2a44'; g.fillRect(X-56,RT-6,112,8);
  g.beginPath(); g.moveTo(X-60,RT-4); g.quadraticCurveTo(X-52,RT-44,X,RT-62); g.quadraticCurveTo(X+52,RT-44,X+60,RT-4); g.closePath();
  g.fillStyle=hShade(g,X,60,['#8a3c4c','#a84c5c','#7a2e3f','#5a2030','#3e1626']); g.fill();
  g.strokeStyle='rgba(255,200,210,0.25)'; g.lineWidth=1; for(let i=-3;i<=3;i++){ g.beginPath(); g.moveTo(X+i*16,RT-5); g.quadraticCurveTo(X+i*10,RT-40,X,RT-61); g.stroke(); }
  g.fillStyle='#2e2a44'; g.fillRect(X-2,RT-90,4,30); g.beginPath(); g.arc(X,RT-66,8,0,TAU); g.fill();
  g.fillStyle='rgba(220,215,255,0.5)'; g.beginPath(); g.arc(X-2.5,RT-68.5,3,0,TAU); g.fill();
  g.strokeStyle='#2e2a44'; g.lineWidth=1.5; g.beginPath(); g.moveTo(X,RT-92); g.lineTo(X,RT-106); g.stroke();
  return c;
}
// railing along the deck ellipse; back=true draws the far half (y above GY), else near half
function railPath(g,back){
  const X=LH.x, GY=LH.galleryY, GH=LH.galleryHalfW-4, GD=LH.galleryDepth-3, RH=LH.railH;
  const n=22; const pts=[];
  for(let i=0;i<=n;i++){ const a=back?Math.PI+Math.PI*i/n:Math.PI*i/n; pts.push([X+Math.cos(a)*GH, GY+Math.sin(a)*GD]); }
  const col=back?'#26223c':'#2e2a44';
  g.lineCap='round';
  g.strokeStyle=col; g.lineWidth=back?2:2.6;
  for(const [x,y] of pts){ g.beginPath(); g.moveTo(x,y); g.lineTo(x,y-RH); g.stroke(); }
  for(const [h,w] of [[RH,back?3:4.2],[RH*0.5,back?1.6:2.2]]){ g.lineWidth=w; g.beginPath(); pts.forEach(([x,y],i)=>i?g.lineTo(x,y-h):g.moveTo(x,y-h)); g.stroke(); }
  if(!back){ g.strokeStyle='rgba(200,195,255,0.45)'; g.lineWidth=1.2; g.beginPath(); pts.forEach(([x,y],i)=>i?g.lineTo(x,y-RH-1.5):g.moveTo(x,y-RH-1.5)); g.stroke();
    g.strokeStyle='rgba(200,195,255,0.3)'; g.lineWidth=1; for(const [x,y] of pts){ if(x<X){ g.beginPath(); g.moveTo(x-0.8,y-1); g.lineTo(x-0.8,y-RH); g.stroke(); } } }
}
function railBuf(){
  const c=U.buf('w_rail',600,240); if(c._ok) return c; c._ok=1;
  const g=c.getContext('2d'); g.scale(2,2); g.translate(-(LH.x-150),-(LH.galleryY-80)); railPath(g,false); return c;
}
W.galleryRail = function(ctx,t,o={}){
  ctx.save(); railPath(ctx,false);
  ctx.restore();
};

W.lighthouse = function(ctx,t,o={}){
  const doorOpen=o.doorOpen||0, lampOn=o.lampOn===undefined?1:o.lampOn, ang=o.lampAngle||0, win=o.windowsLit===undefined?1:o.windowsLit;
  const X=LH.x, B=LH.baseY;
  ctx.save();
  ctx.drawImage(lighthouseBuf(),LB.x0,LB.y0,LB.w,LB.h);
  // ---- lamp room interior
  const gx0=X-LH.glassHW, gy0=LH.glassTop, gw=LH.glassHW*2, gh=LH.glassBot-LH.glassTop, LY=LH.lampY;
  ctx.save(); ctx.beginPath(); ctx.rect(gx0,gy0,gw,gh); ctx.clip();
  let gr=ctx.createRadialGradient(X,LY,4,X,LY,64);
  gr.addColorStop(0,U.mixHex('#2a2850','#ffd88a',lampOn)); gr.addColorStop(0.45,U.mixHex('#1c1b42','#c86a2c',lampOn*0.85)); gr.addColorStop(1,U.mixHex('#141535','#4a2026',lampOn*0.9));
  ctx.fillStyle=gr; ctx.fillRect(gx0,gy0,gw,gh);
  // pedestal, gear ring & brass frame
  ctx.fillStyle='#2e2430'; ctx.fillRect(X-9,LY+26,18,gh); ctx.fillStyle='#8a6a3a'; ctx.fillRect(X-26,LY+33,52,5);
  ctx.fillStyle='#5a4428'; ctx.fillRect(X-26,LY+38,52,3);
  ctx.fillStyle='#8a6a3a'; ctx.fillRect(X-26,LY-38,52,4);
  // lens carousel: 4 bull's-eye Fresnel panels turning around a vertical axis
  const panels=[]; for(let k=0;k<4;k++){ const a=ang+k*Math.PI/2; panels.push({a,c:Math.cos(a),s:Math.sin(a)}); }
  panels.sort((p,q)=>p.c-q.c);
  for(const P of panels){
    const px=X+26*P.s, pw=Math.max(1.5,19*Math.abs(P.c)), ph=32, front=P.c>0; const br=lampOn*(front?0.35+0.65*P.c:0.18);
    gr=ctx.createRadialGradient(px,LY,0,px,LY,ph);
    gr.addColorStop(0,U.mixHex('#3a3a70','#fffbe8',br)); gr.addColorStop(0.5,U.mixHex('#2a2a5a','#ffcf7a',br*0.9)); gr.addColorStop(1,U.mixHex('#1e1c44','#b0602a',br*0.8));
    ctx.fillStyle=gr; ctx.beginPath(); ctx.ellipse(px,LY,pw,ph,0,0,TAU); ctx.fill();
    for(const k of [0.28,0.5,0.7,0.88]){ ctx.strokeStyle=front?U.rgba('#7a4a1a',0.25+0.25*k):'rgba(90,70,60,0.35)'; ctx.lineWidth=1.2;
      ctx.beginPath(); ctx.ellipse(px,LY,pw*k,ph*k,0,0,TAU); ctx.stroke();
      if(front){ ctx.strokeStyle=U.rgba('#fff4d0',0.35*br); ctx.lineWidth=0.7; ctx.beginPath(); ctx.ellipse(px-0.6,LY-0.8,pw*k,ph*k,0,Math.PI*1.05,Math.PI*1.6); ctx.stroke(); } }
    ctx.strokeStyle='#8a6a3a'; ctx.lineWidth=2; ctx.beginPath(); ctx.ellipse(px,LY,pw,ph,0,0,TAU); ctx.stroke();
  }
  // core bulb + blaze of the facing panel
  ctx.globalCompositeOperation='lighter';
  if(lampOn>0){ const fl=0.92+0.08*Math.sin(t*23)*Math.sin(t*7.1);
    gr=ctx.createRadialGradient(X,LY,0,X,LY,22); gr.addColorStop(0,`rgba(255,255,245,${0.9*lampOn*fl})`); gr.addColorStop(0.35,`rgba(255,230,160,${0.45*lampOn})`); gr.addColorStop(1,'rgba(255,180,80,0)');
    ctx.fillStyle=gr; ctx.fillRect(X-24,LY-24,48,48);
    const P=panels[3]; if(P.c>0){ const px=X+26*P.s; gr=ctx.createRadialGradient(px,LY,0,px,LY,16+10*P.c);
      gr.addColorStop(0,`rgba(255,255,240,${lampOn*P.c*0.9})`); gr.addColorStop(1,'rgba(255,200,120,0)'); ctx.fillStyle=gr; ctx.fillRect(px-28,LY-28,56,56); } }
  ctx.globalCompositeOperation='source-over';
  // glass reflections
  ctx.strokeStyle='rgba(220,230,255,0.18)'; ctx.lineWidth=5; ctx.beginPath(); ctx.moveTo(gx0+8,gy0+gh); ctx.lineTo(gx0+28,gy0); ctx.stroke();
  ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(gx0+20,gy0+gh); ctx.lineTo(gx0+36,gy0); ctx.stroke();
  ctx.restore();
  // mullions
  ctx.fillStyle='#2e2a44';
  for(const k of [-1,-0.5,0,0.5,1]){ const x=X+Math.sin(k*1.2)*LH.glassHW/Math.sin(1.2); ctx.fillRect(x-(Math.abs(k)===1?3:1.6),gy0,Math.abs(k)===1?6:3.2,gh); }
  ctx.fillRect(gx0-3,gy0,gw+6,4); ctx.fillRect(gx0-3,gy0+gh-4,gw+6,4); ctx.fillRect(gx0,gy0+gh*0.2,gw,1.8);
  ctx.fillStyle='rgba(210,205,255,0.35)'; ctx.fillRect(gx0-3,gy0+gh-4,3,4);
  // lamp halo
  if(lampOn>0){ ctx.globalCompositeOperation='lighter';
    const fl=0.95+0.05*Math.sin(t*5.3);
    gr=ctx.createRadialGradient(X,LY,0,X,LY,260*fl); gr.addColorStop(0,`rgba(255,230,160,${0.3*lampOn})`); gr.addColorStop(0.25,`rgba(255,190,100,${0.13*lampOn})`); gr.addColorStop(1,'rgba(255,160,70,0)');
    ctx.fillStyle=gr; ctx.fillRect(X-270,LY-270,540,540);
    gr=ctx.createRadialGradient(X,LY,0,X,LY,700); gr.addColorStop(0,`rgba(255,200,130,${0.12*lampOn})`); gr.addColorStop(1,'rgba(255,160,90,0)');
    ctx.fillStyle=gr; ctx.fillRect(X-700,LY-700,1400,1400);
    // anamorphic streak
    ctx.save(); ctx.translate(X,LY); ctx.scale(1,0.03); gr=ctx.createRadialGradient(0,0,0,0,0,360);
    gr.addColorStop(0,`rgba(255,240,200,${0.3*lampOn})`); gr.addColorStop(1,'rgba(255,200,140,0)'); ctx.fillStyle=gr; ctx.fillRect(-360,-360,720,720); ctx.restore();
    ctx.globalCompositeOperation='source-over'; }
  // ---- windows
  for(let i=0;i<WINDOWS.length;i++){ const [wx,wy]=WINDOWS[i]; const fl=win*(0.88+0.12*U.noise(t*3+i*5,90));
    ctx.fillStyle=U.mixHex('#2e2a44','#ffc46a',fl); archPath(ctx,wx,wy-2,18,34); ctx.fill();
    ctx.fillStyle='#2e2a44'; ctx.fillRect(wx-1,wy-36,2,34); ctx.fillRect(wx-9,wy-18,18,2);
    if(fl>0.02){ ctx.globalCompositeOperation='lighter'; gr=ctx.createRadialGradient(wx,wy-16,0,wx,wy-16,70);
      gr.addColorStop(0,`rgba(255,200,110,${0.35*fl})`); gr.addColorStop(1,'rgba(255,170,80,0)'); ctx.fillStyle=gr; ctx.fillRect(wx-70,wy-86,140,140); ctx.globalCompositeOperation='source-over'; } }
  // ---- door
  const dw=LH.doorW, dh=LH.doorH, dB=B+2;
  ctx.save(); archPath(ctx,X,dB,dw,dh); ctx.clip();
  if(doorOpen>0){ gr=ctx.createRadialGradient(X,dB-20,4,X,dB-40,dh); gr.addColorStop(0,U.rgba('#fff3c4',doorOpen)); gr.addColorStop(0.4,U.rgba('#ffb347',doorOpen*0.95)); gr.addColorStop(1,U.rgba('#a0502e',doorOpen*0.9));
    ctx.fillStyle=gr; ctx.fillRect(X-dw/2,dB-dh,dw,dh);
    // stairs silhouette inside
    ctx.fillStyle=U.rgba('#6a3424',0.5*doorOpen); for(let k=0;k<8;k++) ctx.fillRect(X+6+k*8,dB-16-k*18,dw,6); }
  // leaf: hinged on left, swings inward (narrows)
  const lw=dw*(1-doorOpen*0.86);
  const lg=ctx.createLinearGradient(X-dw/2,0,X-dw/2+lw,0);
  lg.addColorStop(0,U.mixHex('#7a4a34','#3a2018',doorOpen*0.6)); lg.addColorStop(1,U.mixHex('#5a3426','#2a160f',doorOpen*0.6));
  ctx.fillStyle=lg; ctx.fillRect(X-dw/2,dB-dh,lw,dh);
  ctx.strokeStyle='rgba(30,14,10,0.6)'; ctx.lineWidth=1.3;
  for(let k=1;k<6;k++){ const x=X-dw/2+lw*k/6; ctx.beginPath(); ctx.moveTo(x,dB-dh); ctx.lineTo(x,dB); ctx.stroke(); }
  ctx.fillStyle='#2e2a44'; for(const hy of [dB-dh+56,dB-36]) ctx.fillRect(X-dw/2,hy,lw*0.7,5);
  ctx.fillStyle='rgba(255,220,180,0.18)'; ctx.fillRect(X-dw/2,dB-dh,lw*0.15,dh);
  if(doorOpen<0.6){ ctx.fillStyle='#d8b060'; ctx.beginPath(); ctx.arc(X-dw/2+lw*0.85,dB-dh*0.45,3.5,0,TAU); ctx.fill(); }
  ctx.restore();
  if(doorOpen>0){ ctx.globalCompositeOperation='lighter';
    gr=ctx.createRadialGradient(X,dB-dh*0.4,0,X,dB-dh*0.4,200); gr.addColorStop(0,`rgba(255,200,110,${0.4*doorOpen})`); gr.addColorStop(1,'rgba(255,160,80,0)');
    ctx.fillStyle=gr; ctx.fillRect(X-200,dB-dh*0.4-200,400,400);
    // light spilling onto the ground
    ctx.save(); ctx.translate(X,dB+2); ctx.scale(1,0.18); gr=ctx.createRadialGradient(0,0,0,0,0,280);
    gr.addColorStop(0,`rgba(255,210,130,${0.75*doorOpen})`); gr.addColorStop(1,'rgba(255,170,80,0)'); ctx.fillStyle=gr; ctx.fillRect(-280,-280,560,560); ctx.restore();
    ctx.globalCompositeOperation='source-over'; }
  // rail (front) unless caller draws it separately
  if(!o.noRail) W.galleryRail(ctx,t,{lampOn});
  ctx.restore();
};

// ---------------------------------------------------------------- BEAM
function beamSprites(){
  const BW=512,BH=256; const c=U.buf('w_beam',BW,BH);
  if(!c._ok){ c._ok=1; const g=c.getContext('2d'); const im=g.createImageData(BW,BH); const d=im.data;
    for(let x=0;x<BW;x++){ const u=x/BW; const hw=U.lerp(0.06,1,u)*BH/2*0.94; const fall=Math.pow(1-u,0.85)*U.smooth(U.clamp(u*14));
      for(let y=0;y<BH;y++){ const dd=Math.abs(y-BH/2)/hw; if(dd>1.25) continue;
        const edge=1-U.smooth(U.clamp((dd-0.45)/0.8)); const core=Math.exp(-dd*dd*2.5);
        const a=fall*edge*(0.3+0.5*core+0.3*Math.exp(-dd*dd*12)*(1-u*0.6)); const i=(y*BW+x)*4;
        d[i]=255; d[i+1]=Math.round(U.lerp(205,243,core)); d[i+2]=Math.round(U.lerp(120,196,core)); d[i+3]=Math.round(255*U.clamp(a)); } }
    g.putImageData(im,0,0); }
  const s=U.buf('w_beamstreak',512,128);
  if(!s._ok){ s._ok=1; const g=s.getContext('2d'); const im=g.createImageData(512,128); const d=im.data;
    for(let y=0;y<128;y++){ const v=U.clamp(0.5+0.5*U.fbm(y*0.09,3)*1.6); for(let x=0;x<512;x++){ const u=x/512; const hw=U.lerp(0.06,1,u)*64*0.9; const dd=Math.abs(y-64)/hw;
      if(dd>1) continue; const a=v*(1-dd)*Math.pow(1-u,1.2)*U.smooth(U.clamp(u*10)); const i=(y*512+x)*4; d[i]=255; d[i+1]=236; d[i+2]=180; d[i+3]=Math.round(255*a); } }
    g.putImageData(im,0,0); }
  return [c,s];
}
let MOTES=null;
W.beam = function(ctx,t,o={}){
  const I=o.intensity===undefined?1:o.intensity; if(I<=0.001) return;
  const ang=o.angle===undefined?Math.PI/2:o.angle, len=o.length||1400, wid=o.width||380;
  const [bs,ss]=beamSprites();
  ctx.save(); ctx.translate(LH.lampX,LH.lampY); ctx.rotate(ang);
  ctx.globalCompositeOperation='lighter';
  const br=0.94+0.06*Math.sin(t*3.1);
  ctx.globalAlpha=Math.min(1,0.8*I*br); ctx.drawImage(bs,0,-wid/2,len,wid);
  // drifting volumetric streaks
  const sh=Math.sin(t*0.4)*wid*0.05;
  ctx.globalAlpha=0.3*I; ctx.drawImage(ss,0,-wid/2+sh,len,wid);
  // dust motes
  if(!MOTES){ MOTES=[]; const r=U.rng(4711); for(let i=0;i<90;i++) MOTES.push({u:r(),v:r()*2-1,sp:0.01+r()*0.025,ph:r()*TAU,sz:0.8+r()*2.2}); }
  const gs=glowSprite('#fff3c4');
  for(const m of MOTES){ const u=(m.u+t*m.sp)%1; const uu=0.04+u*0.92;
    const hw=U.lerp(0.06,1,uu)*wid/2*0.8; const v=U.clamp(m.v+0.15*Math.sin(t*0.7+m.ph),-1,1);
    const x=uu*len, y=v*hw+6*Math.sin(t*1.3+m.ph*2);
    const a=I*(1-Math.abs(v))*(1-uu)*U.smooth(U.clamp(u*8))*U.smooth(U.clamp((1-u)*6))*(0.6+0.4*Math.sin(t*4+m.ph));
    if(a<0.02) continue; const s=m.sz*(1+uu*1.5)*3;
    ctx.globalAlpha=a; ctx.drawImage(gs,x-s,y-s,s*2,s*2); }
  // source flare
  ctx.globalAlpha=1;
  const gr=ctx.createRadialGradient(0,0,0,0,0,90); gr.addColorStop(0,`rgba(255,250,230,${0.9*I})`); gr.addColorStop(0.3,`rgba(255,220,150,${0.35*I})`); gr.addColorStop(1,'rgba(255,180,90,0)');
  ctx.fillStyle=gr; ctx.fillRect(-90,-90,180,180);
  ctx.restore();
};

// ---------------------------------------------------------------- light pool
W.lightPool = function(ctx,x,y,r,intensity=1,color='#ffd27a'){
  if(intensity<=0) return;
  ctx.save(); ctx.globalCompositeOperation='lighter'; ctx.translate(x,y); ctx.scale(1,0.32);
  const gr=ctx.createRadialGradient(0,0,0,0,0,r);
  gr.addColorStop(0,U.rgba(color,0.7*intensity)); gr.addColorStop(0.4,U.rgba(color,0.3*intensity)); gr.addColorStop(1,U.rgba(color,0));
  ctx.fillStyle=gr; ctx.fillRect(-r,-r,r*2,r*2);
  ctx.restore();
};

window.World = W;
})();
