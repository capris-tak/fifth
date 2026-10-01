// FX — light, sparkles, post-processing. Every effect is a pure function of t.
// World-space effects expect the caller to have applied the camera transform.
// Screen-space ones (constellation, bloom, vignette, grain, fade, title, credits) expect an identity transform.
(function(){
const FX = {};
const TAU = Math.PI*2;

// ---------------------------------------------------------------- palette / sprites (deterministic caches)
// heat ramp: 0 = white-hot -> gold -> amber -> rose -> soft violet
const RAMP = ['#ffffff','#fff8e2','#fff0b8','#ffe28a','#ffcf6e','#ffb35a','#ff9a78','#ff8ea6','#e79bdc','#b9a6ff'];
const NR = RAMP.length;
const sprites = {};
function mkCanvas(w,h){const c=document.createElement('canvas');c.width=w;c.height=h;return c;}
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
    gr.addColorStop(0,U.rgba('#ffffff',a));gr.addColorStop(.12,U.rgba(hex,a*.9));
    gr.addColorStop(.45,U.rgba(hex,a*.3));gr.addColorStop(1,U.rgba(hex,0));
    g.fillStyle=gr;
    for(let i=0;i<2;i++){g.save();g.rotate(i*Math.PI/2);g.beginPath();
      g.moveTo(-len,0);g.quadraticCurveTo(0,-wid,len,0);g.quadraticCurveTo(0,wid,-len,0);g.fill();g.restore();}
  };
  ray(S/2,5,1); ray(S/2*.55,10,.5);
  const cg=g.createRadialGradient(0,0,0,0,0,S*.16);
  cg.addColorStop(0,'rgba(255,255,255,1)');cg.addColorStop(.3,U.rgba(hex,.7));cg.addColorStop(1,U.rgba(hex,0));
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

// ---------------------------------------------------------------- shooting star
// o: {x0,y0,x1,y1,t0,t1, arc=0 (perpendicular bulge px), size=1, seed=7, tail=0.45 (s), ease=1.5}
FX.shootingStar = function(ctx,t,o){
  const t0=o.t0,t1=o.t1; if(t<t0||t>t1+1.6)return;
  const S=o.size||1, seed=o.seed||7, arc=o.arc||0, ep=o.ease||1.5, tailT=o.tail||.45, D=t1-t0;
  const dx=o.x1-o.x0, dy=o.y1-o.y0, L=Math.hypot(dx,dy), nx=-dy/L, ny=dx/L;
  const pos=tt=>{const u=U.clamp((tt-t0)/D), p=Math.pow(u,ep), b=Math.sin(Math.PI*p)*arc;
    return [o.x0+dx*p+nx*b, o.y0+dy*p+ny*b];};
  const live=t<=t1, fadeAfter=live?1:Math.exp(-(t-t1)*9);
  ctx.save(); ctx.globalCompositeOperation='lighter';
  // --- tail: sampled history, tapered layered glow + bright core line
  const N=46, pts=[];
  for(let i=0;i<=N;i++){const tt=t-tailT*i/N; if(tt<t0)break; pts.push(pos(tt));}
  if(pts.length>1&&fadeAfter>.01){
    // wide soft ribbon
    for(let pass=0;pass<3;pass++){
      const w=[34,12,3.2][pass]*S, col=['#ff9ac2','#ffd98a','#ffffff'][pass], al=[.10,.30,.9][pass];
      ctx.lineCap='round';
      for(let i=0;i<pts.length-1;i++){
        const k=i/(pts.length-1), taper=Math.pow(1-k,1.6);
        ctx.strokeStyle=U.rgba(col,al*taper*fadeAfter); ctx.lineWidth=Math.max(.5,w*(.25+.75*taper));
        ctx.beginPath(); ctx.moveTo(pts[i][0],pts[i][1]); ctx.lineTo(pts[i+1][0],pts[i+1][1]); ctx.stroke();
      }
    }
    // glow beads along tail (heat ramp, white->gold->rose)
    for(let i=1;i<pts.length;i+=2){const k=i/(pts.length-1);
      glow(ctx,pts[i][0],pts[i][1],(40-28*k)*S,rampHex(.15+k*.7),.22*(1-k)*fadeAfter);}
  }
  // --- shed sparks: emitted at fixed times along the flight, ballistic with drag & gravity
  const NS=Math.round(150*(o.density||1));
  const sp=table('ss',NS,seed,r=>({u:r(),vx:(r()-.5)*260,vy:(r()-.5)*260-40,life:.5+r()*1.0,sz:2+r()*5,ph:r()*TAU,h:r()*.25,dg:r()<.5}));
  for(const q of sp){
    const te=t0+q.u*D, a=t-te; if(a<0||a>q.life)continue;
    const [hx,hy]=pos(te); const [px,py]=ball(hx,hy,q.vx*S,q.vy*S,2.2,260,a);
    const k=a/q.life, al=(1-k)*(1-k)*twk(t,q.ph,40);
    const c=rampHex(q.h+k*.75);
    glow(ctx,px,py,q.sz*4*S,c,al*.35); dot(ctx,px,py,q.sz*S*(1-k*.5),c,al);
    if(q.sz>5.2)spark(ctx,px,py,q.sz*4*S,c,al*.8,q.dg);
  }
  // --- head
  if(fadeAfter>.01){
    const [hx,hy]=pos(Math.min(t,t1)); const fl=.85+.15*Math.sin(t*53)+.08*Math.sin(t*31);
    const grow=U.smooth(U.inv(t0,t0+.25,t));
    const hs=S*grow*fadeAfter;
    glow(ctx,hx,hy,260*hs,'#ff9ac2',.18*fl);
    glow(ctx,hx,hy,140*hs,'#ffd27a',.45*fl);
    glow(ctx,hx,hy,60*hs,'#fff3c4',.9);
    glow(ctx,hx,hy,22*hs,'#ffffff',1);
    // anamorphic streak along flight dir + perpendicular glint
    const ang=Math.atan2(dy,dx);
    sparkRot(ctx,hx,hy,120*hs*fl,'#fff0b8',.95,ang+.2);
    sparkRot(ctx,hx,hy,70*hs,'#ffffff',.8,ang+Math.PI/4+t*2);
  }
  ctx.restore();
};

// ---------------------------------------------------------------- impact
// o: {x,y,t0, scale=1, seed=11, groundY=y (particles settle there)}
FX.impact = function(ctx,t,o){
  const a=t-o.t0; if(a<0||a>5.5)return;
  const S=o.scale||1, seed=o.seed||11, x=o.x, y=o.y, gy=o.groundY!==undefined?o.groundY:y;
  ctx.save(); ctx.globalCompositeOperation='lighter';
  // flash (huge, fast decay) + slow warm afterglow
  const fl=Math.exp(-a*7), ag=Math.exp(-a*.9)*U.smooth(U.inv(0,.08,a));
  glow(ctx,x,y,900*S*(.6+.4*U.easeOut(U.clamp(a*4))),'#fff3c4',fl*.9);
  glow(ctx,x,y,380*S,'#ffffff',fl);
  glow(ctx,x,y-20*S,260*S,'#ffb347',ag*.55);
  glow(ctx,x,y-10*S,110*S,'#fff3c4',ag*.7);
  // horizontal anamorphic flare
  if(fl>.02){ctx.save();ctx.translate(x,y-10*S);ctx.scale(9,.35);glow(ctx,0,0,90*S,'#ffe7a0',fl*.8);ctx.restore();}
  // flash glint
  sparkRot(ctx,x,y-8*S,(420*S)*(.5+.5*U.easeOut(U.clamp(a*3))),'#fff0b8',fl*1.2+ag*.25,0);
  sparkRot(ctx,x,y-8*S,240*S,'#ffffff',fl,Math.PI/4);
  // ground ring (perspective ellipse) — chromatic edge
  for(let ri=0;ri<2;ri++){
    const ra=a-ri*.18; if(ra<0||ra>1.4)continue;
    const k=U.easeOut(U.clamp(ra/1.4)), R=(60+620*k)*S*(ri?0.75:1), fade=Math.pow(1-ra/1.4,1.5);
    const cols=['#ff7aa8','#ffe28a','#8fe8ff'];
    for(let c=0;c<3;c++){
      ctx.strokeStyle=U.rgba(cols[c],(c===1?.75:.45)*fade);
      ctx.lineWidth=(c===1?10:5)*S*(1-k*.7);
      ctx.beginPath();ctx.ellipse(x,y,R+(c-1)*7*S,(R+(c-1)*7*S)*.22,0,0,TAU);ctx.stroke();
    }
  }
  // airborne shock sphere (fast)
  if(a<.6){const k=U.easeOut(a/.6), R=(40+420*k)*S, f=1-a/.6;
    ctx.strokeStyle=U.rgba('#fff3c4',.5*f);ctx.lineWidth=16*S*f;ctx.beginPath();ctx.arc(x,y-10*S,R,Math.PI,TAU);ctx.stroke();
    ctx.strokeStyle=U.rgba('#ff9ac2',.3*f);ctx.lineWidth=6*S*f;ctx.beginPath();ctx.arc(x,y-10*S,R+12*S,Math.PI,TAU);ctx.stroke();}
  // splash particles: arc up & fall, settle on ground and glimmer
  const P=table('imp',130,seed,r=>{const an=-Math.PI*(.06+.88*r()), sp=280+r()*900*(r()<.25?1.5:1);
    return {vx:Math.cos(an)*sp,vy:Math.sin(an)*sp,k:1.2+r()*1.4,life:1.6+r()*2.4,sz:2+r()*6,ph:r()*TAU,h:r()*.3,dg:r()<.5,big:r()<.18};});
  for(const q of P){
    if(a>q.life)continue; const k=a/q.life;
    let [px,py]=ball(x,y-6*S,q.vx*S,q.vy*S,q.k,1100,a);
    let landed=false; if(py>gy){py=gy;landed=true;}
    const al=Math.pow(1-k,1.3)*(landed?twk(t,q.ph,9):1)*U.smooth(U.inv(0,.03,a));
    const c=rampHex(q.h+k*.8), sz=q.sz*S;
    glow(ctx,px,py,sz*5,c,al*.3);
    dot(ctx,px,py,sz*1.2,c,al);
    if(q.big)spark(ctx,px,py,sz*7*(.7+.3*Math.sin(t*12+q.ph)),c,al*.9,q.dg);
    // motion streak while fast
    if(a<.5){const [qx,qy]=ball(x,y-6*S,q.vx*S,q.vy*S,q.k,1100,Math.max(0,a-.05));
      ctx.globalAlpha=1;ctx.strokeStyle=U.rgba(c,.5*al*(1-a/.5));ctx.lineWidth=sz*.8;ctx.beginPath();ctx.moveTo(qx,Math.min(qy,gy));ctx.lineTo(px,py);ctx.stroke();}
  }
  // dust of light: ground-hugging glows spreading out
  const Dd=table('impd',26,seed+1,r=>({dir:r()<.5?-1:1,sp:150+r()*420,sz:30+r()*60,h:.2+r()*.5}));
  for(const q of Dd){const k=U.clamp(a/3.5), px=x+q.dir*q.sp*S*U.easeOut(U.clamp(a/2.5)), al=.22*(1-k)*U.smooth(U.inv(.05,.3,a));
    glow(ctx,px,gy-q.sz*.3*S,q.sz*S,rampHex(q.h),al);}
  // lingering motes rising
  const M=table('impm',40,seed+2,r=>({ox:(r()-.5)*360,rise:30+r()*90,ph:r()*TAU,d:.3+r()*1.2,sz:1.5+r()*3,h:.1+r()*.6}));
  for(const q of M){const b=a-q.d; if(b<0)continue; const k=U.clamp(b/4);
    const px=x+q.ox*S*(.4+.6*U.easeOut(U.clamp(b)))+Math.sin(b*1.7+q.ph)*14*S, py=gy-12*S-q.rise*b*S;
    const al=Math.sin(Math.PI*k)*twk(t,q.ph,5);
    glow(ctx,px,py,q.sz*6*S,rampHex(q.h),al*.35);dot(ctx,px,py,q.sz*S,rampHex(q.h),al*.9);}
  ctx.restore();
};

// ---------------------------------------------------------------- absorb (charge-up)
// o: {x,y,t0,t1, radius=520, scale=1, seed=21, count=160, color (outer tint, default gold ramp)}
FX.absorb = function(ctx,t,o){
  const t0=o.t0,t1=o.t1; if(t<t0||t>t1+.15)return;
  const S=o.scale||1, R=(o.radius||520)*S, seed=o.seed||21, x=o.x, y=o.y, D=t1-t0;
  const prog=U.clamp((t-t0)/D), build=U.easeIn(prog)*.75+prog*.25;
  ctx.save(); ctx.globalCompositeOperation='lighter';
  // converging particles: continuous emission, each repeats on its own period
  const N=o.count||160;
  const P=table('abs',N,seed,r=>({off:r()*3,per:.9+r()*.9,a0:r()*TAU,spin:(1.2+r()*1.8)*(r()<.85?1:-1),rr:.55+r()*.6,sz:1.5+r()*4,h:.15+r()*.75,thr:r(),ph:r()*TAU,dg:r()<.5}));
  for(const q of P){
    if(q.thr>.25+.75*build)continue; // more particles as it builds
    const per=q.per*(1-.45*prog), age=((t-t0+q.off)%per+per)%per, k=age/per;
    if(t-t0<age)continue; // not emitted yet
    const e=U.easeIn(k)*.85+k*.15;
    const pr=(r)=>{const ee=U.easeIn(r)*.85+r*.15; const rad=R*q.rr*(1-ee), an=q.a0+q.spin*ee*2.2+Math.floor((t-t0+q.off)/per)*1.7;
      return [x+Math.cos(an)*rad, y+Math.sin(an)*rad*.85];};
    const [px,py]=pr(k), [bx,by]=pr(Math.max(0,k-.07));
    const al=U.smooth(U.inv(0,.25,k))*(1-U.smooth(U.inv(.85,1,k)))*(.5+.5*build);
    const c=rampHex(q.h*(1-e)); // warm/rose far away, white-hot near the core
    ctx.globalAlpha=1; ctx.strokeStyle=U.rgba(c,.55*al); ctx.lineWidth=q.sz*S*.9; ctx.lineCap='round';
    ctx.beginPath();ctx.moveTo(bx,by);ctx.lineTo(px,py);ctx.stroke();
    glow(ctx,px,py,q.sz*5*S,c,al*.35); dot(ctx,px,py,q.sz*1.1*S,c,al);
    if(q.sz>4.6)spark(ctx,px,py,q.sz*6*S,c,al*.8,q.dg);
  }
  // contracting rings
  for(let i=0;i<4;i++){
    const per=1.1-.5*prog, k=(((t-t0)/per+i/4)%1);
    const rr=R*.9*(1-U.easeIn(k)), al=Math.sin(Math.PI*k)*.28*build;
    ctx.globalAlpha=1;ctx.strokeStyle=U.rgba(i%2?'#ffe28a':'#ffb0c8',al);ctx.lineWidth=(2+3*k)*S;
    ctx.beginPath();ctx.arc(x,y,rr,0,TAU);ctx.stroke();
  }
  // core: growing, pulsing shimmer
  const pulse=.8+.2*Math.sin(t*(10+30*prog))+.1*Math.sin(t*47);
  glow(ctx,x,y,(120+380*build)*S,'#ffb347',.25*build*pulse);
  glow(ctx,x,y,(60+180*build)*S,'#fff3c4',.6*build*pulse);
  glow(ctx,x,y,(20+50*build)*S,'#ffffff',.9*build);
  // rays starting to leak at the end
  if(prog>.5){const rk=U.inv(.5,1,prog); const n=12;
    ctx.globalAlpha=1;
    const gr=ctx.createRadialGradient(x,y,0,x,y,R*1.1*rk);
    gr.addColorStop(0,U.rgba('#fff3c4',.35*rk));gr.addColorStop(1,U.rgba('#ffd27a',0));
    ctx.fillStyle=gr; ctx.beginPath();
    for(let i=0;i<n;i++){const an=i/n*TAU+t*.6+U.hash(i+seed)*.4, w=.025+.02*U.hash(i*3+seed);
      ctx.moveTo(x,y);ctx.lineTo(x+Math.cos(an-w)*R*1.1,y+Math.sin(an-w)*R*1.1);ctx.lineTo(x+Math.cos(an+w)*R*1.1,y+Math.sin(an+w)*R*1.1);ctx.closePath();}
    ctx.fill();
    sparkRot(ctx,x,y,(90+200*rk)*S*pulse,'#fff0b8',.9*rk,t*1.5);
  }
  ctx.restore();
};

// ---------------------------------------------------------------- burst
// o: {x,y,t0, scale=1, seed=31, groundY (optional: shower settles there), count=320}
FX.burst = function(ctx,t,o){
  const a=t-o.t0; if(a<0||a>8)return;
  const S=o.scale||1, seed=o.seed||31, x=o.x, y=o.y;
  ctx.save(); ctx.globalCompositeOperation='lighter';
  const fl=Math.exp(-a*5), ag=Math.exp(-a*.45)*U.smooth(U.inv(0,.06,a));
  // --- flash + afterglow
  glow(ctx,x,y,1500*S,'#fff3c4',fl*.85);
  glow(ctx,x,y,520*S,'#ffffff',fl);
  glow(ctx,x,y,700*S,'#ffb347',ag*.22);
  glow(ctx,x,y,320*S,'#ffe28a',ag*.35);
  glow(ctx,x,y,130*S,'#fffbe6',ag*.5);
  // --- god rays: two rotating layers, single path each, radial gradient fill
  const rayK=U.easeOut(U.clamp(a/.5)), rayF=Math.pow(1-U.clamp(a/4.5),1.6);
  if(rayF>.01){
    const layers=[{n:28,len:1500,w:.035,rot:.05,col:'#ffe7a0',al:.55,s:seed},{n:18,len:1100,w:.012,rot:-.09,col:'#ffffff',al:.7,s:seed+9},{n:14,len:1800,w:.06,rot:.02,col:'#ff9ac2',al:.18,s:seed+17}];
    for(const L of layers){
      const len=L.len*S*(.3+.7*rayK);
      const gr=ctx.createRadialGradient(x,y,0,x,y,len);
      gr.addColorStop(0,U.rgba(L.col,L.al*rayF));gr.addColorStop(.25,U.rgba(L.col,L.al*.5*rayF));gr.addColorStop(1,U.rgba(L.col,0));
      ctx.globalAlpha=1; ctx.fillStyle=gr; ctx.beginPath();
      for(let i=0;i<L.n;i++){const h=U.hash(i*7.31+L.s), an=(i+h*.8)/L.n*TAU+a*L.rot, w=L.w*(.4+1.2*U.hash(i*3.7+L.s)), ll=len*(.55+.45*U.hash(i*1.9+L.s));
        ctx.moveTo(x+Math.cos(an+Math.PI/2)*3*S,y+Math.sin(an+Math.PI/2)*3*S);ctx.lineTo(x+Math.cos(an-w)*ll,y+Math.sin(an-w)*ll);ctx.lineTo(x+Math.cos(an+w)*ll,y+Math.sin(an+w)*ll);ctx.closePath();}
      ctx.fill();
    }
  }
  // --- shockwave rings with chromatic edge
  for(let ri=0;ri<3;ri++){
    const ra=a-ri*.22, dur=1.6+ri*.5; if(ra<0||ra>dur)continue;
    const k=U.easeOut(ra/dur), R=(40+(1250-ri*300)*k)*S, f=Math.pow(1-ra/dur,1.4);
    const cols=['#ff6f9f','#fff0b8','#7fe0ff'], wd=(ri?8:18)*S*(1-k*.6);
    for(let c=0;c<3;c++){ctx.globalAlpha=1;ctx.strokeStyle=U.rgba(cols[c],(c===1?.8:.5)*f);ctx.lineWidth=wd*(c===1?1:.55);
      ctx.beginPath();ctx.arc(x,y,R+(c-1)*wd*.8,0,TAU);ctx.stroke();}
    // soft inner fill of the ring (heat haze)
    if(ri===0){const gr=ctx.createRadialGradient(x,y,R*.6,x,y,R);gr.addColorStop(0,'rgba(255,240,190,0)');gr.addColorStop(1,U.rgba('#ffe7a0',.12*f));
      ctx.fillStyle=gr;ctx.beginPath();ctx.arc(x,y,R,0,TAU);ctx.fill();}
  }
  // --- hero glint
  sparkRot(ctx,x,y,(700*U.easeOut(U.clamp(a*3))*fl+220*ag)*S,'#fff0b8',.9*fl+.5*ag,a*.3);
  sparkRot(ctx,x,y,(380*fl+120*ag)*S,'#ffffff',.9*fl+.4*ag,Math.PI/4-a*.2);
  // --- radial sparkles (hundreds): drag + slight gravity
  const N=o.count||320;
  const P=table('bst',N,seed,r=>{const an=r()*TAU, sp=(250+r()*1500)*(r()<.15?1.4:1);
    return {vx:Math.cos(an)*sp,vy:Math.sin(an)*sp,k:1.6+r()*1.6,life:1.4+r()*2.8,sz:1.5+r()*5.5,ph:r()*TAU,h:r()*.25,dg:r()<.5,big:r()<.2,tw:6+r()*14};});
  for(const q of P){
    if(a>q.life)continue; const k=a/q.life;
    const [px,py]=ball(x,y,q.vx*S,q.vy*S,q.k,90,a);
    const al=Math.pow(1-k,1.2)*(k>.3?twk(t,q.ph,q.tw):1);
    const c=rampHex(q.h+k*.9), sz=q.sz*S*(1-k*.4);
    if(a<.6){const [qx,qy]=ball(x,y,q.vx*S,q.vy*S,q.k,90,Math.max(0,a-.06));
      ctx.globalAlpha=1;ctx.strokeStyle=U.rgba(c,.6*al*(1-a/.6));ctx.lineWidth=sz*.9;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(qx,qy);ctx.lineTo(px,py);ctx.stroke();}
    glow(ctx,px,py,sz*5,c,al*.28); dot(ctx,px,py,sz*1.2,c,al);
    if(q.big)spark(ctx,px,py,sz*8,c,al,q.dg);
  }
  // --- glitter shower: slow falling, swaying, twinkling (starts ~0.3s)
  const G=table('bsg',140,seed+5,r=>({ox:(r()-.5)*1400,oy:-r()*700-100,d:.15+r()*.9,vy:40+r()*90,sw:20+r()*50,ph:r()*TAU,sz:1.5+r()*3.5,h:.05+r()*.8,life:3+r()*4,dg:r()<.5,tw:4+r()*9}));
  for(const q of G){
    const b=a-q.d; if(b<0||b>q.life)continue; const k=b/q.life;
    const out=U.easeOut(U.clamp(b/.9));
    let px=x+q.ox*S*out+Math.sin(b*1.6+q.ph)*q.sw*S, py=y+q.oy*S*out+q.vy*b*S;
    if(o.groundY!==undefined&&py>o.groundY)py=o.groundY;
    const al=Math.sin(Math.PI*Math.pow(k,.6))*twk(t,q.ph,q.tw);
    const c=rampHex(q.h);
    glow(ctx,px,py,q.sz*5*S,c,al*.3); dot(ctx,px,py,q.sz*S,c,al);
    if(q.sz>3.5)spark(ctx,px,py,q.sz*6*S,c,al*.9,q.dg);
  }
  ctx.restore();
};

// ---------------------------------------------------------------- trail
// o: {path:(tt)=>[x,y], t0, t1, life=1.4, rate=70 (particles/s), seed=41, scale=1, head=true}
FX.trail = function(ctx,t,o){
  const t0=o.t0,t1=o.t1, life=o.life||1.4; if(t<t0||t>t1+life)return;
  const S=o.scale||1, seed=o.seed||41, rate=o.rate||70, path=o.path;
  ctx.save(); ctx.globalCompositeOperation='lighter';
  // ribbon: recent path, tapered
  const tn=Math.min(t,t1), ribT=.5, M=24;
  let prev=null;
  ctx.lineCap='round';
  for(let i=0;i<=M;i++){const tt=tn-ribT*i/M; if(tt<t0)break; const p=path(tt);
    if(prev){const k=i/M, fade=(t>t1?Math.exp(-(t-t1)*4):1);
      ctx.globalAlpha=1;
      ctx.strokeStyle=U.rgba('#ffd27a',.18*(1-k)*fade);ctx.lineWidth=26*S*(1-k*.7);ctx.beginPath();ctx.moveTo(prev[0],prev[1]);ctx.lineTo(p[0],p[1]);ctx.stroke();
      ctx.strokeStyle=U.rgba('#fffbe6',.7*(1-k)*fade);ctx.lineWidth=4*S*(1-k*.8);ctx.beginPath();ctx.moveTo(prev[0],prev[1]);ctx.lineTo(p[0],p[1]);ctx.stroke();}
    prev=p;}
  // particles emitted at discrete times
  const dt=1/rate, iA=Math.ceil(Math.max(t0,t-life)/dt), iB=Math.floor(Math.min(t,t1)/dt);
  for(let i=iA;i<=iB;i++){
    const te=i*dt, a=t-te; const r=U.rng((i*2654435761+seed)>>>0);
    const lf=life*(.5+.5*r()); if(a>lf)continue;
    const [ex,ey]=path(te), k=a/lf;
    const [px,py]=ball(ex+(r()-.5)*16*S,ey+(r()-.5)*16*S,(r()-.5)*90*S,(r()-.5)*90*S-10*S,1.8,70*S,a);
    const sz=(1.5+r()*3.5)*S, ph=r()*TAU, c=rampHex(r()*.2+k*.75), al=Math.pow(1-k,1.3)*twk(t,ph,10+r()*10);
    glow(ctx,px,py,sz*5,c,al*.3); dot(ctx,px,py,sz,c,al);
    if(r()<.22)spark(ctx,px,py,sz*6,c,al,r()<.5);
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
  const A=U.buf('fxBloomA',w1,h1),B=U.buf('fxBloomB',w2,h2),C=U.buf('fxBloomC',w3,h3),Bb=U.buf('fxBloomBb',w2,h2);
  const ga=A.getContext('2d'),gb=B.getContext('2d'),gc=C.getContext('2d'),gbb=Bb.getContext('2d');
  const th=(o&&o.threshold!==undefined)?o.threshold:.35;
  const c=1/(1-th); // contrast factor so that `th` maps to ~0 after brightness shift
  ga.globalCompositeOperation='copy';ga.filter=`brightness(${(1-th*.5).toFixed(3)}) contrast(${c.toFixed(3)}) saturate(1.25)`;
  ga.drawImage(cv,0,0,w1,h1); ga.filter='none';
  gb.globalCompositeOperation='copy';gb.filter='blur(2px)';gb.drawImage(A,0,0,w2,h2);gb.filter='none';
  gc.globalCompositeOperation='copy';gc.filter='blur(3px)';gc.drawImage(B,0,0,w3,h3);gc.filter='none';
  gbb.globalCompositeOperation='copy';gbb.filter='blur(4px)';gbb.drawImage(B,0,0);gbb.filter='none';
  ctx.save();
  ctx.globalCompositeOperation='lighter'; ctx.imageSmoothingEnabled=true; ctx.imageSmoothingQuality='high';
  ctx.globalAlpha=Math.min(1,.55*strength); ctx.drawImage(Bb,0,0,W,H);
  ctx.globalAlpha=Math.min(1,.7*strength); ctx.drawImage(C,0,0,W,H);
  ctx.globalAlpha=Math.min(1,.35*strength); ctx.drawImage(A,0,0,W,H);
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

FX.fade = function(ctx,alpha){
  if(!(alpha>0))return;
  ctx.save();ctx.globalAlpha=Math.min(1,alpha);ctx.fillStyle='#000';ctx.fillRect(0,0,ctx.canvas.width,ctx.canvas.height);ctx.restore();
};

// ---------------------------------------------------------------- title (screen space)
// o: {t0, x=W/2, y=H*0.34, size=118, alpha=1, sub='The Star Lighthouse'}
const TITLE='ほしのとうだい';
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
    ctx.globalAlpha=A*e*.9; ctx.shadowColor=U.rgba('#ffcf6e',1); ctx.shadowBlur=FS*.45; ctx.fillStyle=U.rgba('#ffcf6e',.35);
    ctx.fillText(chars[i],px,py);
    ctx.shadowBlur=FS*.12; ctx.shadowColor=U.rgba('#fff3c4',1); ctx.fillStyle=U.rgba('#fff3c4',.25);
    ctx.fillText(chars[i],px,py);
    ctx.shadowBlur=0;
    // crisp letter
    ctx.globalCompositeOperation='source-over';
    ctx.globalAlpha=A*U.smooth(k);
    ctx.fillStyle=U.mixHex('#ffffff','#fff1d0',.5); ctx.fillText(chars[i],px,py);
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
    ctx.shadowColor='rgba(255,200,120,.7)';ctx.shadowBlur=12;ctx.fillStyle='#f3e7ff';
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
