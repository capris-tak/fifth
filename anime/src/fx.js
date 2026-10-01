// FX — light, sparkles, post-processing. Every effect is a pure function of t.
// World-space effects expect the caller to have applied the camera transform.
// Screen-space ones (constellation, bloom, vignette, grain, fade, title, credits) expect an identity transform.
(function(){
const FX = {};
const TAU = Math.PI*2;

// ---------------------------------------------------------------- palette / sprites (deterministic caches)
// heat ramp: 0 = white-hot -> gold -> amber -> rose -> soft violet
const RAMP = ['#ffffff','#fff8e2','#fff0b8','#ffe28a','#ffd06a','#ffb84a','#ff9a4a','#ff8a5c','#ff7f7a','#f27a96'];
const NR = RAMP.length;
const sprites = {};
let cid=0; function mkCanvas(w,h){const c=document.createElement('canvas');c.width=w;c.height=h;c._id=++cid;return c;}
// soft gaussian-ish glow, colour hex, 128px
function glowSprite(hex){
  const k='g'+hex; if(sprites[k])return sprites[k];
  const S=128,c=mkCanvas(S,S),g=c.getContext('2d');
  const gr=g.createRadialGradient(S/2,S/2,0,S/2,S/2,S/2);
  const st=[[0,1],[.08,.82],[.2,.5],[.35,.26],[.5,.12],[.7,.04],[1,0]];
  for(const [p,a] of st)gr.addColorStop(p,U.rgba(hex,a));
  g.fillStyle=gr;g.fillRect(0,0,S,S);
  return sprites[k]=c;
}
// 4-point cross sparkle (thin rays + tiny core), colour hex, 128px. diag=true -> rotated 45deg
function sparkSprite(hex,diag){
  const k='s'+hex+(diag?'d':''); if(sprites[k])return sprites[k];
  const S=128,c=mkCanvas(S,S),g=c.getContext('2d');
  g.translate(S/2,S/2); if(diag)g.rotate(Math.PI/4);
  g.globalCompositeOperation='lighter';
  const ray=(len,wid,a)=>{
    const gr=g.createRadialGradient(0,0,0,0,0,len);
    gr.addColorStop(0,U.rgba('#ffffff',a));gr.addColorStop(.06,U.rgba(hex,a*.95));
    gr.addColorStop(.45,U.rgba(hex,a*.3));gr.addColorStop(1,U.rgba(hex,0));
    g.fillStyle=gr;
    for(let i=0;i<2;i++){g.save();g.rotate(i*Math.PI/2);g.beginPath();
      g.moveTo(-len,0);g.quadraticCurveTo(0,-wid,len,0);g.quadraticCurveTo(0,wid,-len,0);g.fill();g.restore();}
  };
  ray(S/2,5,1); ray(S/2*.55,10,.5);
  const cg=g.createRadialGradient(0,0,0,0,0,S*.16);
  cg.addColorStop(0,'rgba(255,255,255,1)');cg.addColorStop(.18,U.rgba(hex,.8));cg.addColorStop(1,U.rgba(hex,0));
  g.fillStyle=cg;g.beginPath();g.arc(0,0,S*.16,0,TAU);g.fill();
  return sprites[k]=c;
}
// hard-ish tiny dot (bright pixel core)
function dotSprite(hex){
  const k='d'+hex; if(sprites[k])return sprites[k];
  const S=32,c=mkCanvas(S,S),g=c.getContext('2d');
  const gr=g.createRadialGradient(S/2,S/2,0,S/2,S/2,S/2);
  gr.addColorStop(0,'rgba(255,255,255,1)');gr.addColorStop(.25,U.rgba(hex,.95));gr.addColorStop(.55,U.rgba(hex,.25));gr.addColorStop(1,U.rgba(hex,0));
  g.fillStyle=gr;g.fillRect(0,0,S,S);
  return sprites[k]=c;
}
// blurred radial god-ray texture (white), deterministic per seed. 512px, rays fade to edge.
function rayTex(seed,n,wmin,wmax){
  const k='r'+seed+':'+n; if(sprites[k])return sprites[k];
  const S=512,raw=mkCanvas(S,S),g=raw.getContext('2d'),c=S/2;
  const r=U.rng(seed);
  g.globalCompositeOperation='lighter';
  for(let i=0;i<n;i++){
    const an=(i+r()*.9)/n*TAU, w=wmin+(wmax-wmin)*r()*r(), len=c*(.45+.55*r()), a=.35+.65*r();
    const gr=g.createRadialGradient(c,c,0,c,c,len);
    gr.addColorStop(0,`rgba(255,255,255,${a})`);gr.addColorStop(.3,`rgba(255,255,255,${a*.45})`);gr.addColorStop(1,'rgba(255,255,255,0)');
    g.fillStyle=gr;g.beginPath();g.moveTo(c,c);
    g.lineTo(c+Math.cos(an-w)*len,c+Math.sin(an-w)*len);g.lineTo(c+Math.cos(an+w)*len,c+Math.sin(an+w)*len);g.closePath();g.fill();
  }
  const out=mkCanvas(S,S),o=out.getContext('2d');
  o.filter='blur(2.5px)';o.drawImage(raw,0,0);o.filter='none';
  return sprites[k]=out;
}
// burst god-ray texture: broad amber rays + fine white/gold rays, baked once
function burstRays(seed){
  const k='br'+seed; if(sprites[k])return sprites[k];
  const RA=warmRays(rayTex(seed,48,.006,.035)), RB=tinted(rayTex(seed+13,30,.015,.06),'#ff9a4a');
  const S=768,c=mkCanvas(S,S),g=c.getContext('2d');g.globalCompositeOperation='lighter';
  g.globalAlpha=.55;g.drawImage(RB,0,0,S,S);
  g.globalAlpha=1;const r=S/2*1500/1800;g.drawImage(RA,S/2-r,S/2-r,r*2,r*2);
  return sprites[k]=c;
}
// white-core / gold-edge version of a ray texture
function warmRays(src){
  const k='w'+src._id; if(sprites[k])return sprites[k];
  const S=src.width,c=mkCanvas(S,S),g=c.getContext('2d');
  g.drawImage(src,0,0);g.globalCompositeOperation='source-in';
  const gr=g.createRadialGradient(S/2,S/2,0,S/2,S/2,S/2);gr.addColorStop(0,'#ffffff');gr.addColorStop(.25,'#fff0b8');gr.addColorStop(.6,'#ffc860');gr.addColorStop(1,'#ff9a4a');
  g.fillStyle=gr;g.fillRect(0,0,S,S);
  return sprites[k]=c;
}
function tinted(src,hex){
  const k='t'+src._id+hex; if(sprites[k])return sprites[k];
  const c=mkCanvas(src.width,src.height),g=c.getContext('2d');
  g.drawImage(src,0,0);g.globalCompositeOperation='source-in';g.fillStyle=hex;g.fillRect(0,0,c.width,c.height);
  return sprites[k]=c;
}
// several concentric glows baked into one sprite (exact, since all layers share one alpha envelope)
// layers: [[radiusRel (0..1), hex, alpha], ...]
function bakeGlow(key,layers){
  const k='m'+key; if(sprites[k])return sprites[k];
  const S=256,c=mkCanvas(S,S),g=c.getContext('2d');g.globalCompositeOperation='lighter';
  for(const [rr,hex,a] of layers){g.globalAlpha=a;const r=rr*S/2;g.drawImage(glowSprite(hex),S/2-r,S/2-r,r*2,r*2);}
  return sprites[k]=c;
}
// draw a texture centred at (x,y), radius R, rotation rot
function texAt(ctx,tex,x,y,R,rot,a){ if(a<=.003)return; ctx.save();ctx.translate(x,y);ctx.rotate(rot);ctx.globalAlpha=Math.min(1,a);ctx.drawImage(tex,-R,-R,R*2,R*2);ctx.restore(); }
const rampHex = h=>RAMP[Math.max(0,Math.min(NR-1,Math.round(h*(NR-1))))];
function glow(ctx,x,y,r,hex,a){ if(a<=.003||r<=.2)return; ctx.globalAlpha=Math.min(1,a); ctx.drawImage(glowSprite(hex),x-r,y-r,r*2,r*2); }
function spark(ctx,x,y,r,hex,a,diag){ if(a<=.003||r<=.2)return; ctx.globalAlpha=Math.min(1,a); ctx.drawImage(sparkSprite(hex,diag),x-r,y-r,r*2,r*2); }
function dot(ctx,x,y,r,hex,a){ if(a<=.003||r<=.1)return; ctx.globalAlpha=Math.min(1,a); ctx.drawImage(dotSprite(hex),x-r,y-r,r*2,r*2); }
// rotated sparkle (for a few hero glints)
function sparkRot(ctx,x,y,r,hex,a,rot){ if(a<=.003)return; ctx.save();ctx.translate(x,y);ctx.rotate(rot);ctx.globalAlpha=Math.min(1,a);ctx.drawImage(sparkSprite(hex),-r,-r,r*2,r*2);ctx.restore(); }
FX._glow=glow; FX._spark=spark; FX._dot=dot; FX.RAMP=RAMP;

// closed-form ballistic motion with linear drag k and gravity g (y down)
function ball(x0,y0,vx,vy,k,g,a){
  const e=Math.exp(-k*a), f=(1-e)/k;
  return [x0+vx*f, y0+vy*f+(g/k)*(a-f)];
}
// cached deterministic particle tables
const tables={};
function table(name,n,seed,gen){const k=name+':'+n+':'+seed;if(tables[k])return tables[k];const r=U.rng(seed);const a=[];for(let i=0;i<n;i++)a.push(gen(r,i));return tables[k]=a;}
// twinkle 0..1 deterministic per particle
const twk=(t,ph,sp)=>.55+.45*Math.sin(t*sp+ph);

// soft chromatic ring (ellipse when squash<1): rose outside, gold middle, cyan inside
function chromaRing(ctx,x,y,R,squash,w,a,spread){
  if(a<=.003||R<=0)return; const off=(spread||1)*w*.9;
  const L=[['#ff7a5a',off,.6],['#ffe7a0',0,1],['#6fd8ff',-off,.45]];
  ctx.globalAlpha=1;
  for(const [c,d,m] of L){const r=Math.max(1,R+d);
    ctx.strokeStyle=U.rgba(c,a*m*.22);ctx.lineWidth=w*3;ctx.beginPath();ctx.ellipse(x,y,r,r*squash,0,0,TAU);ctx.stroke();
    ctx.strokeStyle=U.rgba(c,a*m*.6);ctx.lineWidth=w*.8;ctx.beginPath();ctx.ellipse(x,y,r,r*squash,0,0,TAU);ctx.stroke();}
}
// tapered-looking motion streak: faint wide + bright thin core
function streak(ctx,x0,y0,x1,y1,w,hex,a){
  if(a<=.003)return; ctx.globalAlpha=1; ctx.lineCap='round';
  ctx.strokeStyle=U.rgba(hex,a*.3);ctx.lineWidth=w*1.3;ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();
  ctx.strokeStyle=U.rgba('#fffbe6',a*.8);ctx.lineWidth=w*.45;ctx.beginPath();ctx.moveTo((x0+x1)/2,(y0+y1)/2);ctx.lineTo(x1,y1);ctx.stroke();
}

// ---------------------------------------------------------------- shooting star
// o: {x0,y0,x1,y1,t0,t1, arc=0 (perpendicular bulge px), size=1, seed=7, tail=0.45 (s of history), ease=1.5, density=1}
// Head accelerates (ease exponent), tail = sampled path history, sparks shed along the flight. Lingers ~1.5 s after t1 (sparks only).
FX.shootingStar = function(ctx,t,o){
  const t0=o.t0,t1=o.t1; if(t<t0||t>t1+1.6)return;
  const S=o.size||1, seed=o.seed||7, arc=o.arc||0, ep=o.ease||1.5, tailT=o.tail||.45, D=t1-t0;
  const dx=o.x1-o.x0, dy=o.y1-o.y0, L=Math.hypot(dx,dy), nx=-dy/L, ny=dx/L;
  const pos=tt=>{const u=U.clamp((tt-t0)/D), p=Math.pow(u,ep), b=Math.sin(Math.PI*p)*arc;
    return [o.x0+dx*p+nx*b, o.y0+dy*p+ny*b];};
  const live=t<=t1, fadeAfter=live?1:Math.exp(-(t-t1)*10);
  ctx.save(); ctx.globalCompositeOperation='lighter'; ctx.lineCap='round';
  // --- tail
  const N=40, pts=[];
  for(let i=0;i<=N;i++){const tt=Math.min(t,t1)-tailT*i/N - (live?0:(t-t1)); if(tt<t0)break; pts.push(pos(Math.max(t0,tt)));}
  if(pts.length>1&&fadeAfter>.01){
    const passes=[[46,'#ff9a30',.06,1.1],[22,'#ffc040',.14,1.4],[9,'#ffe28a',.35,1.8],[2.6,'#fffbe6',.95,2.4]];
    for(const [w,col,al,pw] of passes){
      for(let i=0;i<pts.length-1;i++){
        const k=i/(pts.length-1), taper=Math.pow(1-k,pw);
        ctx.strokeStyle=U.rgba(col,al*taper*fadeAfter); ctx.lineWidth=Math.max(.4,w*S*(.2+.8*Math.pow(1-k,.7)));
        ctx.beginPath(); ctx.moveTo(pts[i][0],pts[i][1]); ctx.lineTo(pts[i+1][0],pts[i+1][1]); ctx.stroke();
      }
    }
  }
  // --- shed sparks (ballistic, drag + gravity), colour cools from white-hot to gold to rose
  const NS=Math.round(170*(o.density||1));
  const sp=table('ss',NS,seed,r=>({u:Math.pow(r(),.7),vx:(r()-.5)*240,vy:(r()-.5)*200-30,life:.45+r()*1.1,sz:1.5+r()*4.5,ph:r()*TAU,h:r()*.2,dg:r()<.5,tw:10+r()*30}));
  for(const q of sp){
    const te=t0+q.u*D, a=t-te; if(a<0||a>q.life)continue;
    const [hx,hy]=pos(te); const [px,py]=ball(hx,hy,q.vx*S,q.vy*S,2.4,220,a);
    const k=a/q.life, al=Math.pow(1-k,1.5)*(.6+.4*Math.sin(t*q.tw+q.ph));
    const c=rampHex(q.h+k*.8);
    glow(ctx,px,py,q.sz*4.5*S,c,al*.3); dot(ctx,px,py,q.sz*S,c,al);
    if(q.sz>4.6)spark(ctx,px,py,q.sz*4.5*S,c,al*.8,q.dg);
  }
  // --- head
  if(fadeAfter>.01){
    const [hx,hy]=pos(Math.min(t,t1)); const fl=.88+.12*Math.sin(t*53)+.06*Math.sin(t*31);
    const hs=S*U.smooth(U.inv(t0,t0+.25,t))*fadeAfter;
    glow(ctx,hx,hy,300*hs,'#ffa030',.12*fl);
    glow(ctx,hx,hy,170*hs,'#ffb347',.28*fl);
    glow(ctx,hx,hy,70*hs,'#ffe28a',.6);
    glow(ctx,hx,hy,26*hs,'#ffffff',1);
    const ang=Math.atan2(dy,dx);
    sparkRot(ctx,hx,hy,130*hs*fl,'#ffe7a0',.95,ang+Math.PI/4+.12*Math.sin(t*7));
    sparkRot(ctx,hx,hy,50*hs,'#ffffff',.6,ang+t*3);
  }
  ctx.restore();
};

// ---------------------------------------------------------------- impact
// o: {x,y,t0, scale=1, seed=11, groundY=y (particles settle there)}. Active t0 .. t0+5.5
FX.impact = function(ctx,t,o){
  const a=t-o.t0; if(a<0||a>5.5)return;
  const S=o.scale||1, seed=o.seed||11, x=o.x, y=o.y, gy=o.groundY!==undefined?o.groundY:y;
  ctx.save(); ctx.globalCompositeOperation='lighter'; ctx.lineCap='round';
  const fl=Math.exp(-a*8), ag=Math.exp(-a*.8)*U.smooth(U.inv(0,.1,a));
  // flash
  if(fl>.03)texAt(ctx,bakeGlow('iflash',[[1,'#ff9a30',.65],[.42,'#ffd060',.85],[.16,'#ffffff',1]]),x,y-20*S,1000*S*(.7+.3*U.easeOut(U.clamp(a*4))),0,fl);
  // god-ray splash, upward fan (ground hemisphere)
  const rk=Math.exp(-a*2.2)*U.smooth(U.inv(0,.05,a));
  if(rk>.01){ctx.save();ctx.beginPath();ctx.rect(x-3000,y-3000,6000,3000+6*S);ctx.clip();
    texAt(ctx,warmRays(rayTex(seed+3,40,.01,.05)),x,y,(500+500*U.easeOut(U.clamp(a*2)))*S,a*.05,rk);
    ctx.restore();}
  // warm afterglow pool
  ctx.save();ctx.translate(x,gy);ctx.scale(1,.32);texAt(ctx,bakeGlow('ipool',[[1,'#ff9a30',.5],[.4,'#ffe28a',.55]]),0,0,340*S,0,ag);ctx.restore();
  glow(ctx,x,y-14*S,70*S,'#fff3c4',ag*.5);
  // anamorphic flare
  if(fl>.02){ctx.save();ctx.translate(x,y-12*S);ctx.scale(10,.28);glow(ctx,0,0,100*S,'#ffe7a0',fl*.9);ctx.restore();}
  sparkRot(ctx,x,y-12*S,(460*S)*(.4+.6*U.easeOut(U.clamp(a*3)))*(fl+.15*ag),'#fff0b8',fl+ag*.3,.0);
  // ground rings (perspective ellipse) with chromatic edge
  for(let ri=0;ri<2;ri++){
    const ra=a-ri*.2, dur=1.5; if(ra<0||ra>dur)continue;
    const k=U.easeOut(ra/dur), R=(50+680*k)*S*(ri?0.72:1), f=Math.pow(1-ra/dur,1.6);
    chromaRing(ctx,x,gy,R,.2,(ri?2.2:3.5)*S*(1-k*.5),f*.75,1.6);
  }
  // airborne shock dome
  if(a<.7){const k=U.easeOut(a/.7), R=(40+460*k)*S, f=Math.pow(1-a/.7,2);
    ctx.save();ctx.beginPath();ctx.rect(x-3000,y-3000,6000,3000);ctx.clip();chromaRing(ctx,x,y,R,.9,3.5*S,f*.5,1.6);ctx.restore();}
  // splash particles: arc up & fall, settle on ground and glimmer
  const P=table('imp',150,seed,r=>{const an=-Math.PI*(.05+.9*r()), sp=260+r()*820*(r()<.25?1.6:1);
    return {vx:Math.cos(an)*sp,vy:Math.sin(an)*sp,k:1.2+r()*1.4,life:1.6+r()*2.6,sz:1.3+r()*3.6,ph:r()*TAU,h:r()*.2,dg:r()<.5,big:r()<.2,tw:6+r()*10,h2:.12+r()*.25};});
  for(const q of P){
    if(a>q.life)continue; const k=a/q.life;
    let [px,py]=ball(x,y-6*S,q.vx*S,q.vy*S,q.k,1100,a);
    let landed=false; if(py>gy){py=gy;landed=true;}
    const al=Math.pow(1-k,1.2)*(landed?.4+.6*twk(t,q.ph,q.tw):1)*U.smooth(U.inv(0,.03,a));
    const c=rampHex(q.h2+k*.7), sz=q.sz*S;
    if(a<.45&&!landed){const [qx,qy]=ball(x,y-6*S,q.vx*S,q.vy*S,q.k,1100,Math.max(0,a-.045));
      streak(ctx,qx,qy,px,py,sz,c,al*(1-a/.45));}
    glow(ctx,px,py,sz*5,c,al*.28);
    dot(ctx,px,py,sz*1.1,c,al);
    if(q.big)spark(ctx,px,py,sz*6*(.75+.25*Math.sin(t*q.tw+q.ph)),c,al*.9,q.dg);
  }
  // dust of light: ground-hugging glows spreading out
  const Dd=table('impd',26,seed+1,r=>({dir:r()<.5?-1:1,sp:150+r()*420,sz:30+r()*60,h:.25+r()*.5}));
  for(const q of Dd){const k=U.clamp(a/3.5), px=x+q.dir*q.sp*S*U.easeOut(U.clamp(a/2.5)), al=.13*(1-k)*U.smooth(U.inv(.05,.3,a));
    ctx.save();ctx.translate(px,gy-q.sz*.2*S);ctx.scale(1.6,.5);glow(ctx,0,0,q.sz*S,rampHex(q.h),al);ctx.restore();}
  // lingering motes rising slowly
  const M=table('impm',46,seed+2,r=>({ox:(r()-.5)*420,rise:25+r()*80,ph:r()*TAU,d:.3+r()*1.4,sz:1.4+r()*2.6,h:.1+r()*.6}));
  for(const q of M){const b=a-q.d; if(b<0)continue; const k=U.clamp(b/4);
    const px=x+q.ox*S*(.4+.6*U.easeOut(U.clamp(b)))+Math.sin(b*1.7+q.ph)*14*S, py=gy-12*S-q.rise*b*S;
    const al=Math.sin(Math.PI*k)*twk(t,q.ph,5);
    glow(ctx,px,py,q.sz*6*S,rampHex(q.h),al*.3);dot(ctx,px,py,q.sz*S,rampHex(q.h),al*.9);}
  ctx.restore();
};

// ---------------------------------------------------------------- absorb (charge-up)
// o: {x,y,t0,t1, radius=520, scale=1, seed=21, count=170}. Active t0 .. t1 (+0.15)
FX.absorb = function(ctx,t,o){
  const t0=o.t0,t1=o.t1; if(t<t0||t>t1+.12)return;
  const S=o.scale||1, R=(o.radius||520)*S, seed=o.seed||21, x=o.x, y=o.y, D=t1-t0;
  const prog=U.clamp((t-t0)/D), build=U.easeIn(prog)*.7+prog*.3, out=1-U.clamp((t-t1)/.12);
  ctx.save(); ctx.globalCompositeOperation='lighter'; ctx.lineCap='round'; ctx.lineJoin='round';
  if(out<1)ctx.globalAlpha=1;
  // ambient warm gathering haze
  texAt(ctx,bakeGlow('ahaze',[[1,'#ff9a30',.12],[.55,'#ffc040',.2]]),x,y,R*1.1,0,build);
  const SQ=o.squash||.7;
  const N=o.count||170;
  const P=table('abs',N,seed,r=>({off:r()*3,per:1+r()*1,a0:r()*TAU,spin:(1.6+r()*1.6)*(r()<.85?1:-1),rr:.5+r()*.65,sz:1.4+r()*3.6,h:.25+r()*.7,thr:r(),ph:r()*TAU,dg:r()<.5}));
  for(const q of P){
    if(q.thr>.3+.7*build)continue;
    const per=q.per*(1-.4*prog), cyc=(t-t0+q.off)/per, k=cyc-Math.floor(cyc);
    if(t-t0<k*per)continue;
    const rot=Math.floor(cyc)*1.7;
    const pr=r=>{const ee=Math.pow(r,2.2); const rad=R*q.rr*(1-ee), an=q.a0+rot+q.spin*Math.pow(r,1.6)*2.4;
      return [x+Math.cos(an)*rad, y+Math.sin(an)*rad*SQ];};
    const al=U.smooth(U.inv(0,.2,k))*(1-U.smooth(U.inv(.88,1,k)))*(.55+.45*build)*out;
    const c=rampHex(q.h*(1-Math.pow(k,1.5)));
    // curved streak: 4 samples behind
    if(al<.02)continue;
    const pts=[];for(let j=0;j<=5;j++)pts.push(pr(Math.max(0,k-.14+.14*j/5)));
    ctx.strokeStyle=U.rgba(c,al*.5);ctx.lineWidth=q.sz*S*.7;ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);for(let j=1;j<=5;j++)ctx.lineTo(pts[j][0],pts[j][1]);ctx.stroke();
    const [px,py]=pts[5];
    glow(ctx,px,py,q.sz*5*S,c,al*.3); dot(ctx,px,py,q.sz*1.1*S,c,al);
    if(q.sz>4.2)spark(ctx,px,py,q.sz*5*S,c,al*.8,q.dg);
  }
  // contracting rings
  for(let i=0;i<3;i++){
    const per=1.2-.5*prog, k=(((t-t0)/per+i/3)%1);
    const rr=R*.95*(1-U.easeIn(k)), al=Math.sin(Math.PI*k)*.35*build*out;
    chromaRing(ctx,x,y,rr,SQ,2*S,al,1.5);
  }
  // core: growing, pulsing shimmer
  const pulse=.82+.12*Math.sin(t*(8+34*prog))+.06*Math.sin(t*57);
  texAt(ctx,bakeGlow('acore',[[1,'#ff9a30',.4],[.42,'#ffd060',.7],[.12,'#ffffff',1]]),x,y,(150+380*build)*S,0,build*pulse*(.4+.6*out));
  // rays leaking out near the end
  if(prog>.45){const rk=U.smooth(U.inv(.45,1,prog));
    texAt(ctx,warmRays(rayTex(seed+5,26,.008,.03)),x,y,R*(.5+.7*rk),t*.25,.75*rk*pulse);
    sparkRot(ctx,x,y,(80+220*rk)*S*pulse,'#fff0b8',.9*rk,Math.PI/4*Math.sin(t*.7));}
  ctx.restore();
};

// ---------------------------------------------------------------- burst
// o: {x,y,t0, scale=1, seed=31, groundY (optional: glitter settles there), count=320}. Active t0 .. t0+8
FX.burst = function(ctx,t,o){
  const a=t-o.t0; if(a<0||a>8)return;
  const S=o.scale||1, seed=o.seed||31, x=o.x, y=o.y;
  ctx.save(); ctx.globalCompositeOperation='lighter'; ctx.lineCap='round';
  const fl=Math.exp(-a*5.5), ag=Math.exp(-a*.5)*U.smooth(U.inv(0,.08,a));
  // --- flash + afterglow (warm, layered)
  if(fl>.03)texAt(ctx,bakeGlow('bflash',[[1,'#ff9a30',.6],[.55,'#ffd060',.8],[.17,'#ffffff',1]]),x,y,1500*S*(.75+.25*U.easeOut(U.clamp(a*5))),0,fl);
  texAt(ctx,bakeGlow('bafter',[[1,'#ffb020',.2],[.55,'#ffd040',.32],[.25,'#ffe890',.5],[.09,'#fffbe6',.7]]),x,y,700*S,0,ag);
  // --- god rays: blurred ray textures, slow counter-rotation, grow fast & fade
  const rg=U.easeOut(U.clamp(a/.45)), rf=Math.pow(1-U.clamp(a/3.6),1.7)*U.smooth(U.inv(0,.04,a));
  if(rf>.01){
    texAt(ctx,burstRays(seed),x,y,1800*S*(.38+.62*rg),a*.06,rf);
  }
  // anamorphic flare
  if(fl>.02){ctx.save();ctx.translate(x,y);ctx.scale(12,.22);glow(ctx,0,0,120*S,'#ffe7a0',fl);ctx.restore();}
  // --- shockwave rings with chromatic edge
  for(let ri=0;ri<3;ri++){
    const ra=a-ri*.2, dur=1.5+ri*.5; if(ra<0||ra>dur)continue;
    const k=U.easeOut(ra/dur), R=(40+(1300-ri*330)*k)*S, f=Math.pow(1-ra/dur,1.5);
    chromaRing(ctx,x,y,R,1,(ri?3.5:7)*S*(1-k*.5),f*(ri?.7:1),1.6);
  }
  // --- hero glint
  sparkRot(ctx,x,y,(760*U.easeOut(U.clamp(a*3))*fl+200*ag)*S,'#fff0b8',fl+.55*ag,.08*Math.sin(a*1.3));
  sparkRot(ctx,x,y,(300*fl+90*ag)*S,'#ffffff',.7*fl+.35*ag,Math.PI/4);
  // --- radial sparkles (hundreds): drag + slight gravity, white-hot -> gold -> rose
  const N=o.count||320;
  const P=table('bst',N,seed,r=>{const an=r()*TAU, sp=(220+r()*1400)*(r()<.15?1.45:1);
    return {vx:Math.cos(an)*sp,vy:Math.sin(an)*sp,k:1.5+r()*1.6,life:1.4+r()*2.8,sz:1.4+r()*4.6,ph:r()*TAU,h:r()*.15,dg:r()<.5,big:r()<.2,tw:6+r()*14};});
  for(const q of P){
    if(a>q.life)continue; const k=a/q.life;
    const [px,py]=ball(x,y,q.vx*S,q.vy*S,q.k,90,a);
    const al=Math.pow(1-k,1.1)*(k>.25?.35+.65*twk(t,q.ph,q.tw):1);
    const c=rampHex(q.h+k*.8), sz=q.sz*S*(1-k*.35);
    if(a<.55){const [qx,qy]=ball(x,y,q.vx*S,q.vy*S,q.k,90,Math.max(0,a-.05));streak(ctx,qx,qy,px,py,sz,c,al*(1-a/.55));}
    glow(ctx,px,py,sz*5,c,al*.26); dot(ctx,px,py,sz*1.1,c,al);
    if(q.big)spark(ctx,px,py,sz*7,c,al,q.dg);
  }
  // --- glitter shower: slow falling, swaying, twinkling
  const G=table('bsg',150,seed+5,r=>({ox:(r()-.5)*1500,oy:-r()*700-80,d:.15+r()*.9,vy:35+r()*80,sw:20+r()*50,ph:r()*TAU,sz:1.4+r()*3.2,h:.08+r()*.75,life:3+r()*4,dg:r()<.5,tw:4+r()*9}));
  for(const q of G){
    const b=a-q.d; if(b<0||b>q.life)continue; const k=b/q.life;
    const out=U.easeOut(U.clamp(b/1.1));
    const px=x+q.ox*S*out+Math.sin(b*1.6+q.ph)*q.sw*S; let py=y+q.oy*S*out+q.vy*b*S;
    if(o.groundY!==undefined&&py>o.groundY)py=o.groundY;
    const al=Math.sin(Math.PI*Math.pow(k,.6))*(.25+.75*Math.pow(twk(t,q.ph,q.tw),2));
    const c=rampHex(q.h);
    glow(ctx,px,py,q.sz*5*S,c,al*.28); dot(ctx,px,py,q.sz*S,c,al);
    if(q.sz>3.2)spark(ctx,px,py,q.sz*6*S,c,al*.9,q.dg);
  }
  ctx.restore();
};


// ---------------------------------------------------------------- trail
// o: {path:(tt)=>[x,y], t0, t1, life=1.4, rate=70 (particles/s), seed=41, scale=1, head=true}
FX.trail = function(ctx,t,o){
  const t0=o.t0,t1=o.t1, life=o.life||1.4; if(t<t0||t>t1+life)return;
  const S=o.scale||1, seed=o.seed||41, rate=o.rate||90, path=o.path;
  ctx.save(); ctx.globalCompositeOperation='lighter';
  // ribbon: recent path drawn as 4 overlapping sub-paths (no bead artefacts at joints), tapering
  const tn=Math.min(t,t1), ribT=o.ribbon||.45, M=24, fade=(t>t1?Math.exp(-(t-t1)*4):1);
  const rp=[]; for(let i=0;i<=M;i++){const tt=tn-ribT*i/M; if(tt<t0)break; rp.push(path(tt));}
  ctx.lineCap='round'; ctx.lineJoin='round'; ctx.globalAlpha=1;
  if(rp.length>1&&fade>.01){
    for(let q=0;q<4;q++){const n=Math.ceil(rp.length*(1-q/4)); if(n<2)continue; const k=q/4;
      ctx.beginPath();ctx.moveTo(rp[0][0],rp[0][1]);for(let i=1;i<n;i++)ctx.lineTo(rp[i][0],rp[i][1]);
      ctx.strokeStyle=U.rgba('#ffc040',.07*fade);ctx.lineWidth=(10+12*k)*S;ctx.stroke();
      ctx.strokeStyle=U.rgba('#fff3c4',.22*fade);ctx.lineWidth=(1+2.2*k)*S;ctx.stroke();}
  }
  // particles emitted at discrete times
  const dt=1/rate, iA=Math.ceil(Math.max(t0,t-life)/dt), iB=Math.floor(Math.min(t,t1)/dt);
  for(let i=iA;i<=iB;i++){
    const te=i*dt, a=t-te; const r=U.rng((i*2654435761+seed)>>>0);
    const lf=life*(.5+.5*r()); if(a>lf)continue;
    const [ex,ey]=path(te), k=a/lf;
    const [px,py]=ball(ex+(r()-.5)*16*S,ey+(r()-.5)*16*S,(r()-.5)*90*S,(r()-.5)*90*S-10*S,1.8,70*S,a);
    const sz=(1.5+r()*3.5)*S, ph=r()*TAU, c=rampHex(r()*.2+k*.75), al=Math.pow(1-k,1.3)*twk(t,ph,10+r()*10);
    glow(ctx,px,py,sz*5,c,al*.3); dot(ctx,px,py,sz,c,al);
    if(r()<.3)spark(ctx,px,py,sz*6,c,al,r()<.5);
  }
  if(o.head!==false&&t<=t1){const [hx,hy]=path(t);glow(ctx,hx,hy,60*S,'#fff3c4',.4);}
  ctx.restore();
};

// ---------------------------------------------------------------- motes
// o: {x,y,w,h,count=30,seed=51,color='#fff3c4',size=1,alpha=1,rise=14 (px/s)}
FX.motes = function(ctx,t,o){
  const n=o.count||30, seed=o.seed||51, col=o.color||'#fff3c4', S=o.size||1, A=o.alpha===undefined?1:o.alpha, rise=o.rise===undefined?14:o.rise;
  if(A<=0)return;
  const P=table('mot',n,seed,r=>({fx:r(),fy:r(),ph:r()*TAU,ph2:r()*TAU,sp:.3+r()*.6,sz:1+r()*2.6,tw:1+r()*3,wx:15+r()*40,wy:8+r()*20}));
  ctx.save(); ctx.globalCompositeOperation='lighter';
  for(const q of P){
    const yy=((q.fy*o.h - rise*t*q.sp)%o.h+o.h)%o.h;
    const px=o.x+q.fx*o.w+Math.sin(t*q.sp*.9+q.ph)*q.wx+U.noise(t*.3+q.ph,seed)*q.wx*.5;
    const py=o.y+yy+Math.sin(t*q.sp*1.3+q.ph2)*q.wy;
    const edge=Math.min(1,yy/(o.h*.15),(o.h-yy)/(o.h*.15));
    const al=A*edge*Math.pow(.5+.5*Math.sin(t*q.tw+q.ph),2)*.9+A*edge*.1;
    glow(ctx,px,py,q.sz*7*S,col,al*.3); dot(ctx,px,py,q.sz*S*1.2,col,al);
  }
  ctx.restore();
};

// ---------------------------------------------------------------- constellation (screen space)
// o: {points:[[x,y]..], lines:[[i,j]..], t0, t1, cam?, alpha=1, color='#cfe0ff', size=1, hero?:index (extra bright star)}
// If cam is given, points are world coords (converted with U.toScreen); otherwise screen px.
FX.constellation = function(ctx,t,o){
  const t0=o.t0,t1=o.t1; if(t<t0)return;
  const A=o.alpha===undefined?1:o.alpha; if(A<=0)return;
  const S=o.size||1, col=o.color||'#cfe0ff', D=t1-t0;
  const pts=o.cam?o.points.map(p=>U.toScreen(o.cam,p[0],p[1])):o.points;
  const np=pts.length, nl=o.lines.length;
  ctx.save(); ctx.globalCompositeOperation='lighter'; ctx.lineCap='round';
  // stars pop in over first 35%, lines draw over 25%..100%
  const popEnd=t0+D*.35, lineA=t0+D*.25;
  const lineSlot=(D*.75)/nl;
  // line drawing — each line in sequence, with travelling bright tip
  for(let i=0;i<nl;i++){
    const [ia,ib]=o.lines[i], s=lineA+i*lineSlot*.85, k=U.easeInOut(U.clamp((t-s)/(lineSlot*1.3)));
    if(k<=0)continue;
    const [ax,ay]=pts[ia],[bx,by]=pts[ib], ex=ax+(bx-ax)*k, ey=ay+(by-ay)*k;
    const shimmer=.75+.25*Math.sin(t*1.3-i*.9);
    ctx.globalAlpha=A;
    ctx.strokeStyle=U.rgba(col,.10*shimmer);ctx.lineWidth=10*S;ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(ex,ey);ctx.stroke();
    ctx.strokeStyle=U.rgba(col,.22*shimmer);ctx.lineWidth=4*S;ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(ex,ey);ctx.stroke();
    ctx.strokeStyle=U.rgba('#ffffff',.55*shimmer);ctx.lineWidth=1.3*S;ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(ex,ey);ctx.stroke();
    // travelling pulse along finished lines
    if(k>=1){const pk=((t*.35+i*.37)%1.6); if(pk<1){const px=ax+(bx-ax)*pk,py=ay+(by-ay)*pk;glow(ctx,px,py,14*S,col,.35*A*Math.sin(Math.PI*pk));}}
    if(k<1){glow(ctx,ex,ey,30*S,'#fff3c4',.7*A);spark(ctx,ex,ey,26*S,'#ffffff',A);}
  }
  // stars
  for(let i=0;i<np;i++){
    const s=t0+(popEnd-t0)*(np>1?i/(np-1):0)*.85+U.hash(i*3.3)*.12*D, k=U.clamp((t-s)/.6);
    if(k<=0)continue;
    const [px,py]=pts[i], pop=U.easeOutBack(k), flash=Math.exp(-(t-s)*4)*(t>=s?1:0);
    const hero=o.hero===i, tw=.8+.2*Math.sin(t*2.3+i*1.7);
    const r=(hero?2.2:1)*S*pop*tw;
    glow(ctx,px,py,46*r,col,.35*A);
    glow(ctx,px,py,16*r,'#fffbe6',.8*A);
    spark(ctx,px,py,(hero?70:30)*r,'#fff6d8',A*(.85+flash));
    if(flash>.01){glow(ctx,px,py,90*S,'#fff3c4',flash*.6*A);spark(ctx,px,py,90*S*flash,'#ffffff',flash*A,true);}
  }
  ctx.restore();
};

// ---------------------------------------------------------------- post: bloom
// Downsample whole canvas to small buffers, crude threshold via contrast, blur, add back with 'lighter'.
// o (optional): {threshold=0.35 (0..1), tint}
FX.bloom = function(ctx,strength,o){
  if(!(strength>0))return;
  const cv=ctx.canvas, W=cv.width, H=cv.height;
  const w1=W>>2,h1=H>>2,w2=W>>3,h2=H>>3,w3=W>>4,h3=H>>4;
  const A=U.buf('fxBloomA',w1,h1),B=U.buf('fxBloomB',w2,h2),C=U.buf('fxBloomC',w3,h3),M=U.buf('fxBloomM',w1,h1);
  const ga=A.getContext('2d'),gb=B.getContext('2d'),gc=C.getContext('2d'),gm=M.getContext('2d');
  const th=(o&&o.threshold!==undefined)?o.threshold:.42;
  // brightness(b) then contrast(c) == (v - th)/(1 - th): a soft threshold
  const bb=1/(1+th), cc=(1+th)/(1-th);
  // luminance mask (grayscale -> threshold) multiplied by the colour image, so bloom keeps the true hue
  const D=U.buf('fxBloomD',w1,h1),gd=D.getContext('2d');
  gd.globalCompositeOperation='copy';gd.globalAlpha=1;gd.drawImage(cv,0,0,w1,h1);
  ga.globalCompositeOperation='copy';ga.globalAlpha=1;ga.filter=`grayscale(1) brightness(${bb.toFixed(4)}) contrast(${cc.toFixed(4)})`;
  ga.drawImage(D,0,0); ga.filter='none';
  ga.globalCompositeOperation='multiply';ga.drawImage(D,0,0);
  gb.globalCompositeOperation='copy';gb.filter='blur(2px)';gb.drawImage(A,0,0,w2,h2);gb.filter='none';
  gc.globalCompositeOperation='copy';gc.filter='blur(3px)';gc.drawImage(B,0,0,w3,h3);gc.filter='none';
  // mix the three scales into one quarter-res buffer, then ONE full-screen additive pass
  gm.globalCompositeOperation='copy';gm.globalAlpha=1;gm.drawImage(A,0,0);
  gm.globalCompositeOperation='lighter';gm.imageSmoothingEnabled=true;
  gm.drawImage(B,0,0,w1,h1);gm.drawImage(C,0,0,w1,h1);
  ctx.save();
  ctx.globalCompositeOperation='lighter'; ctx.imageSmoothingEnabled=true;
  ctx.globalAlpha=Math.min(1,.5*strength); ctx.drawImage(M,0,0,W,H);
  ctx.restore();
};

// ---------------------------------------------------------------- post: vignette
FX.vignette = function(ctx,amount){
  if(!(amount>0))return;
  const W=ctx.canvas.width,H=ctx.canvas.height;
  const V=U.buf('fxVig',W>>2,H>>2);
  if(!V._done){const g=V.getContext('2d'),w=V.width,h=V.height;
    const gr=g.createRadialGradient(w/2,h*.48,h*.25,w/2,h*.5,Math.hypot(w,h)*.56);
    gr.addColorStop(0,'rgba(4,3,18,0)');gr.addColorStop(.5,'rgba(4,3,18,.25)');gr.addColorStop(1,'rgba(4,3,18,1)');
    g.fillStyle=gr;g.fillRect(0,0,w,h);V._done=true;}
  ctx.save();ctx.globalAlpha=Math.min(1,amount);ctx.drawImage(V,0,0,W,H);ctx.restore();
};

// ---------------------------------------------------------------- post: film grain
FX.grain = function(ctx,t,amount){
  if(!(amount>0))return;
  const W=ctx.canvas.width,H=ctx.canvas.height, n=4, frame=Math.floor(t*24)%n;
  const T=U.buf('fxGrain'+frame,256,256);
  if(!T._done){const g=T.getContext('2d'),id=g.createImageData(256,256),r=U.rng(9001+frame*77);
    for(let i=0;i<id.data.length;i+=4){const v=128+(r()+r()+r()-1.5)*110;id.data[i]=id.data[i+1]=id.data[i+2]=v;id.data[i+3]=255;}
    g.putImageData(id,0,0);T._done=true;}
  ctx.save();
  ctx.globalCompositeOperation='overlay'; ctx.globalAlpha=Math.min(1,amount);
  const ox=Math.floor(U.hash(Math.floor(t*24))*256), oy=Math.floor(U.hash(Math.floor(t*24)+.5)*256);
  ctx.translate(-ox,-oy); ctx.fillStyle=ctx.createPattern(T,'repeat'); ctx.fillRect(ox,oy,W,H);
  ctx.restore();
};

// Fast path: vignette + grain baked into 4 full-res frames, applied in ONE source-over pass (~8 ms instead of ~19 ms).
// Amounts are quantised to 0.02 and a new pair triggers a one-off rebuild (~0.3 s), so keep them constant over a shot.
FX.vignetteGrain = function(ctx,t,vAmount,gAmount){
  const W=ctx.canvas.width,H=ctx.canvas.height, va=Math.round((vAmount||0)*50)/50, gaq=Math.round((gAmount||0)*50)/50;
  if(va<=0&&gaq<=0)return;
  const n=4, frame=Math.floor(t*24)%n, key='fxVG'+frame+'_'+va+'_'+gaq;
  const T=U.buf(key,W,H);
  if(!T._done){
    const g=T.getContext('2d'),id=g.createImageData(W,H),d=id.data,r=U.rng(4242+frame*131);
    const cx=W/2,cy=H*.49,r0=H*.25,r1=Math.hypot(W,H)*.56;
    const amp=gaq*.55;
    for(let y=0;y<H;y++)for(let x=0;x<W;x++){
      const u=Math.min(1,Math.max(0,(Math.hypot(x-cx,y-cy)-r0)/(r1-r0)));
      let v=u<.5?.25*(u/.5):.25+.75*((u-.5)/.5); v=v*v*(3-2*v)*va;
      const nz=(r()+r()-1)*amp, a=Math.abs(nz);
      const al=Math.min(1,v+a), i=(y*W+x)*4;
      const c=nz>0&&al>0?Math.round(255*a/al):0;
      d[i]=d[i+1]=d[i+2]=c; d[i]=Math.min(255,c+0); d[i+2]=Math.min(255,c+(c?0:18)); // slight navy tint in the darks
      d[i+3]=Math.round(al*255);
    }
    g.putImageData(id,0,0);T._done=true;
  }
  ctx.save();ctx.globalAlpha=1;ctx.globalCompositeOperation='source-over';ctx.drawImage(T,0,0);ctx.restore();
};

FX.fade = function(ctx,alpha){
  if(!(alpha>0))return;
  ctx.save();ctx.globalAlpha=Math.min(1,alpha);ctx.fillStyle='#000';ctx.fillRect(0,0,ctx.canvas.width,ctx.canvas.height);ctx.restore();
};

// ---------------------------------------------------------------- title (screen space)
// o: {t0, x=W/2, y=H*0.34, size=118, alpha=1, sub='The Star Lighthouse'}
const TITLE='ほしのとうだい';
// cached blurred glow of one glyph (built once per glyph/size)
function glyphGlow(ch,FS,font){
  const k='gl'+ch+FS; if(sprites[k])return sprites[k];
  const S=Math.ceil(FS*2.4),c=mkCanvas(S,S),g=c.getContext('2d');
  g.font=font;g.textAlign='center';g.textBaseline='middle';
  g.filter=`blur(${(FS*.16).toFixed(1)}px)`;g.fillStyle='#ffc46a';g.fillText(ch,S/2,S/2);
  g.globalCompositeOperation='lighter';
  g.filter=`blur(${(FS*.05).toFixed(1)}px)`;g.fillStyle='rgba(255,243,196,.8)';g.fillText(ch,S/2,S/2);
  g.filter=`blur(${(FS*.4).toFixed(1)}px)`;g.fillStyle='rgba(255,150,120,.7)';g.fillText(ch,S/2,S/2);
  g.filter='none';
  return sprites[k]=c;
}
FX.title = function(ctx,t,o){
  const t0=o.t0; if(t<t0-.1)return;
  const A=o.alpha===undefined?1:o.alpha; if(A<=0)return;
  const X=o.x===undefined?U.W/2:o.x, Y=o.y===undefined?U.H*.34:o.y, FS=o.size||118;
  const chars=[...TITLE], n=chars.length, stag=.2;
  ctx.save();
  ctx.font=`${FS}px IPAPGothic, IPAGothic, serif`; ctx.textAlign='center'; ctx.textBaseline='middle';
  const ls=FS*.22; // letter spacing
  const widths=chars.map(c=>ctx.measureText(c).width), total=widths.reduce((s,w)=>s+w,0)+ls*(n-1);
  let cx=X-total/2;
  const centers=widths.map(w=>{const c=cx+w/2;cx+=w+ls;return c;});
  // soft backing haze (behind, normal blend) for legibility
  const hz=U.smooth(U.inv(t0,t0+1.5,t))*A;
  if(hz>0){ctx.save();ctx.globalAlpha=hz*.35;ctx.translate(X,Y);ctx.scale(total*.75/100,FS*1.2/100);
    const g=ctx.createRadialGradient(0,0,0,0,0,100);g.addColorStop(0,'rgba(10,8,40,.8)');g.addColorStop(1,'rgba(10,8,40,0)');
    ctx.fillStyle=g;ctx.beginPath();ctx.arc(0,0,100,0,TAU);ctx.fill();ctx.restore();}
  for(let i=0;i<n;i++){
    const s=t0+i*stag, k=U.clamp((t-s)/1.1); if(k<=0)continue;
    const e=U.easeOut(k), px=centers[i], py=Y+(1-e)*FS*.18, flash=Math.exp(-Math.max(0,t-s-.25)*3)*U.smooth(U.inv(0,.25,t-s));
    // glow layer
    ctx.globalCompositeOperation='lighter';
    const gg=glyphGlow(chars[i],FS,ctx.font), breathe=.85+.15*Math.sin(t*1.6+i*.8);
    ctx.globalAlpha=Math.min(1,A*e*(.75*breathe+flash*.6)); ctx.drawImage(gg,px-gg.width/2,py-gg.height/2);
    // crisp letter, cream -> warm gold vertical gradient
    ctx.globalCompositeOperation='source-over';
    ctx.globalAlpha=A*U.smooth(k);
    const lg=ctx.createLinearGradient(0,py-FS*.5,0,py+FS*.5);lg.addColorStop(0,'#ffffff');lg.addColorStop(.55,'#fff4d6');lg.addColorStop(1,'#ffd9a0');
    ctx.fillStyle=lg; ctx.fillText(chars[i],px,py);
    // birth flash & glint
    if(flash>.01){ctx.globalCompositeOperation='lighter';glow(ctx,px,py,FS*.9,'#ffe28a',flash*.5*A);spark(ctx,px+FS*.28,py-FS*.3,FS*.5*flash,'#ffffff',flash*A);}
  }
  ctx.globalCompositeOperation='lighter';
  // travelling sparkle across the title after all letters are in
  const sweepS=t0+n*stag+.6, sk=(t-sweepS)/1.4;
  if(sk>0&&sk<1){const sx=X-total/2+total*U.easeInOut(sk), sy=Y-FS*.15+Math.sin(sk*Math.PI*3)*FS*.1, f=Math.sin(Math.PI*sk);
    glow(ctx,sx,sy,FS*.8,'#fff3c4',.35*f*A);sparkRot(ctx,sx,sy,FS*.55*f,'#ffffff',f*A,sk*2);}
  // ornament: thin line drawn outward from centre with tiny star at centre
  const ok=U.easeInOut(U.clamp((t-t0-.8)/1.6));
  if(ok>0){const oy=Y+FS*.78, hw=total*.42*ok;
    ctx.globalAlpha=A*.9;const g=ctx.createLinearGradient(X-hw,0,X+hw,0);
    g.addColorStop(0,'rgba(255,220,150,0)');g.addColorStop(.5,'rgba(255,236,190,.85)');g.addColorStop(1,'rgba(255,220,150,0)');
    ctx.fillStyle=g;ctx.fillRect(X-hw,oy-1,hw*2,2);
    sparkRot(ctx,X,oy,22*ok*(.85+.15*Math.sin(t*3)),'#fff0b8',A,t*.5);
    // tiny accent stars at ends
    dot(ctx,X-hw,oy,3,'#ffe7a0',A*ok);dot(ctx,X+hw,oy,3,'#ffe7a0',A*ok);}
  // subtitle
  const sub=o.sub===undefined?'The Star Lighthouse':o.sub;
  const sbk=U.smooth(U.clamp((t-t0-1.4)/1.2));
  if(sub&&sbk>0){ctx.globalCompositeOperation='source-over';ctx.globalAlpha=A*sbk*.85;
    ctx.font=`italic ${Math.round(FS*.24)}px 'Liberation Serif', 'DejaVu Serif', serif`;
    try{ctx.letterSpacing=`${Math.round(FS*.06)}px`;}catch(e){}
    ctx.fillStyle='#f3e7ff';
    ctx.fillText(sub,X,Y+FS*1.12+(1-sbk)*8);}
  // a few drifting sparkles around the title (deterministic)
  const tk=U.smooth(U.inv(t0+.5,t0+2,t))*A;
  if(tk>0){ctx.globalCompositeOperation='lighter';
    for(let i=0;i<14;i++){const h1=U.hash(i*5.1+3),h2=U.hash(i*9.7+1),ph=h1*TAU;
      const sx=X-total*.6+total*1.2*h1+Math.sin(t*.4+ph)*12, sy=Y-FS*.9+FS*1.9*h2+Math.cos(t*.3+ph)*8;
      const tw=Math.pow(Math.max(0,Math.sin(t*(1.2+h2*1.5)+ph)),6);
      spark(ctx,sx,sy,(8+16*h2)*tw+2,'#fff0b8',tk*tw,i%2===0);dot(ctx,sx,sy,2,'#fff6d8',tk*(.3+.5*tw));}}
  ctx.restore();
};

// ---------------------------------------------------------------- credits (screen space, bottom)
// o: {t0, alpha=1, y=H-74}
FX.credits = function(ctx,t,o){
  const k=U.smooth(U.clamp((t-o.t0)/1.2)); const A=(o.alpha===undefined?1:o.alpha)*k; if(A<=0)return;
  const Y=o.y===undefined?U.H-74:o.y;
  ctx.save(); ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.font='25px IPAPGothic, IPAGothic, sans-serif';
  try{ctx.letterSpacing='3px';}catch(e){}
  ctx.shadowColor='rgba(0,0,10,.85)';ctx.shadowBlur=8;
  ctx.globalAlpha=A*.92;ctx.fillStyle='#ece6ff';
  ctx.fillText('VOICEVOX:冥鳴ひまり　VOICEVOX:雨晴はう　VOICEVOX:No.7',U.W/2,Y+(1-k)*6);
  ctx.font="italic 21px 'Liberation Serif', 'DejaVu Serif', serif";
  try{ctx.letterSpacing='2px';}catch(e){}
  const k2=U.smooth(U.clamp((t-o.t0-.4)/1.2));
  ctx.globalAlpha=A*k2*.75;ctx.fillStyle='#d8d0f0';
  ctx.fillText('Animation · Music · Sound — all generated with code',U.W/2,Y+38+(1-k2)*6);
  // hairline
  ctx.shadowBlur=0;ctx.globalAlpha=A*.35;ctx.fillStyle='#e8dcff';ctx.fillRect(U.W/2-60,Y-28,120,1);
  ctx.restore();
};

window.FX = FX;
})();
