# Final mix: voices (placed per timeline) + music (ducked under dialogue) + sfx + ambience -> audio/mix.wav (48k stereo)
import json, numpy as np, os
from scipy.io import wavfile
from scipy.signal import resample_poly, butter, sosfilt
SR=48000; N=SR*60
def load(p):
    if not os.path.exists(p): print('missing',p); return np.zeros((N,2))
    sr,x=wavfile.read(p); x=x.astype(np.float64)/(32768 if x.dtype==np.int16 else 1)
    if x.ndim==1: x=np.stack([x,x],1)
    if sr!=SR: x=resample_poly(x,SR,sr,axis=0)
    out=np.zeros((N,2)); out[:min(N,len(x))]=x[:N]; return out
lines=json.load(open('tools/voices.json'))
starts={l['id']:l['start'] for l in json.loads(open('src/timeline.js').read().split('window.LINES = ')[1].split(';\n')[0])}
voice=np.zeros((N,2)); duck=np.zeros(N)
PAN={'mina':-0.12,'hoshi':0.12,'narrator':0.0}
for L in lines:
    sr,x=wavfile.read(f'audio/voice/{L["id"]}.wav'); x=x.astype(np.float64)/32768
    x=resample_poly(x,SR,sr)
    # gentle voice EQ: high-pass 90 Hz, slight presence
    x=sosfilt(butter(2,90,'hp',fs=SR,output='sos'),x)
    rms=np.sqrt(np.mean(x[np.abs(x)>0.01]**2)); x*=0.13/rms   # loudness normalize
    p=PAN[L['who']]; s=int(starts[L['id']]*SR); e=min(N,s+len(x))
    voice[s:e,0]+=x[:e-s]*np.sqrt(0.5-p/2)*1.41; voice[s:e,1]+=x[:e-s]*np.sqrt(0.5+p/2)*1.41
    duck[max(0,s-int(.15*SR)):min(N,e+int(.3*SR))]=1
# smooth duck envelope
k=int(0.25*SR); duck=np.convolve(duck,np.ones(k)/k,'same')
# small voice room reverb for narrator-ish air
ir_len=int(0.6*SR); rng=np.random.default_rng(1); ir=rng.standard_normal(ir_len)*np.exp(-np.arange(ir_len)/(0.12*SR))
from scipy.signal import fftconvolve
ir2=rng.standard_normal(ir_len)*np.exp(-np.arange(ir_len)/(0.12*SR))
vrev=np.stack([fftconvolve(voice[:,0],ir)[:N],fftconvolve(voice[:,1],ir2)[:N]],1)
vrev*=0.12*np.abs(voice).max()/(np.abs(vrev).max()+1e-9)
music=load('audio/music.wav'); sfx=load('audio/sfx.wav'); amb=load('audio/ambience.wav')
mix=voice+vrev+music*(1-0.3*duck)[:,None]*0.85+sfx*(1-0.3*duck)[:,None]*0.9+amb*0.7
# master: soft-knee limiter
pk=np.abs(mix).max(); print('pre-peak',pk)
g=np.ones(N); env=np.abs(mix).max(1)
thr=0.85
over=np.maximum(env/thr,1.0)
# gain smoothing (attack instantaneous via max filter, release 80ms)
from scipy.ndimage import maximum_filter1d
over=maximum_filter1d(over,int(0.005*SR))
rel=np.exp(-1/(0.08*SR)); gg=np.empty(N); cur=1.0
for i in range(0,N,48):
    target=over[i:i+48].max(); cur=max(target,cur*rel**48+ (1-rel**48)*1.0) if target<cur else target; gg[i:i+48]=cur
mix=mix/gg[:,None]
mix=np.clip(mix,-0.97,0.97)
# fade tail
f=int(0.8*SR); mix[-f:]*=np.linspace(1,0,f)[:,None]
wavfile.write('audio/mix.wav',SR,(mix*32767).astype(np.int16))
for sec in range(0,60,5): print(sec, round(20*np.log10(np.sqrt(np.mean(mix[sec*SR:(sec+5)*SR]**2))+1e-9),1),'dB')
