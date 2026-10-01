// Shared helpers. Everything must be a pure function of time t (seconds) — frames render out of order in parallel.
(function(){
const U = {};
U.W = 1920; U.H = 1080; U.FPS = 30; U.DUR = 60;
U.clamp = (v,a=0,b=1)=>Math.max(a,Math.min(b,v));
U.lerp = (a,b,k)=>a+(b-a)*k;
U.inv = (a,b,v)=>U.clamp((v-a)/(b-a));           // 0..1 progress of v between a and b
U.smooth = k=>k*k*(3-2*k);
U.easeInOut = k=>k<.5?4*k*k*k:1-Math.pow(-2*k+2,3)/2;
U.easeOut = k=>1-Math.pow(1-k,3);
U.easeIn = k=>k*k*k;
U.easeOutBack = k=>{const c1=1.70158,c3=c1+1;return 1+c3*Math.pow(k-1,3)+c1*Math.pow(k-1,2);};
U.easeOutElastic = k=>k===0?0:k===1?1:Math.pow(2,-10*k)*Math.sin((k*10-.75)*(2*Math.PI)/3)+1;
// seeded PRNG (mulberry32) — create with U.rng(seed), call r() -> [0,1)
U.rng = seed=>{let a=seed>>>0;return ()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};};
U.hash = (n)=>{const s=Math.sin(n*127.1+311.7)*43758.5453;return s-Math.floor(s);}; // deterministic 0..1
// smooth 1D value noise, ~[-1,1]
U.noise = (x,seed=0)=>{const i=Math.floor(x),f=x-i,a=U.hash(i+seed*57.3)*2-1,b=U.hash(i+1+seed*57.3)*2-1;return U.lerp(a,b,U.smooth(f));};
U.fbm = (x,seed=0)=>U.noise(x,seed)*.6+U.noise(x*2.1,seed+1)*.3+U.noise(x*4.3,seed+2)*.1;
U.rgba = (hex,a=1)=>{const n=parseInt(hex.slice(1),16);return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`;};
U.mixHex = (h1,h2,k)=>{const a=parseInt(h1.slice(1),16),b=parseInt(h2.slice(1),16);const r=Math.round(U.lerp(a>>16&255,b>>16&255,k)),g=Math.round(U.lerp(a>>8&255,b>>8&255,k)),bl=Math.round(U.lerp(a&255,b&255,k));return '#'+((1<<24)|(r<<16)|(g<<8)|bl).toString(16).slice(1);};
// Camera: cam = {x, y, zoom, rot?}. World point (x,y) is shown at screen center when cam.x=x, cam.y=y.
// U.camApply(ctx, cam, parallax=1): sets transform so subsequent drawing is in world coords.
// parallax <1 makes a layer move less (far background).
U.camApply = (ctx,cam,par=1)=>{ctx.setTransform(1,0,0,1,0,0);ctx.translate(U.W/2,U.H/2);if(cam.rot)ctx.rotate(cam.rot);const z=1+(cam.zoom-1)*par;ctx.scale(z,z);ctx.translate(-cam.x*par,-cam.y*par);};
U.toScreen = (cam,x,y)=>[U.W/2+(x-cam.x)*cam.zoom, U.H/2+(y-cam.y)*cam.zoom];
// Offscreen canvas cache (create once, reuse)
const pool={}; U.buf=(name,w=U.W,h=U.H)=>{let c=pool[name];if(!c||c.width!==w||c.height!==h){c=pool[name]=document.createElement('canvas');c.width=w;c.height=h;}return c;};
window.U = U;
})();
