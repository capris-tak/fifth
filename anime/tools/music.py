#!/usr/bin/env python3
"""Original score for 「ほしのとうだい」 — composed and synthesized entirely in numpy/scipy.

Run:   /opt/tts/bin/python tools/music.py            -> audio/music.wav + audio/music_cues.txt
       /opt/tts/bin/python tools/music.py --analyze  -> also prints loudness analysis

Key: D major (B minor colour for the meeting).  Main motif ("Hoshi's theme"):
    A  D  E  F#  |  E  D  B  A  ...   (5-1-2-3 | 2-1-6-5)
Everything is placed in absolute seconds so it locks to the master timeline in SPEC.md.
"""
import os
import sys
import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

SR = 48000
DUR = 60.0
N = int(round(SR * DUR))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_WAV = os.path.join(ROOT, 'audio', 'music.wav')
OUT_CUES = os.path.join(ROOT, 'audio', 'music_cues.txt')

_seed = np.random.default_rng(20261001)


def nxt():
    return int(_seed.integers(0, 2**31 - 1))


# ----------------------------------------------------------------------------- pitch helpers
_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def M(n):
    if not isinstance(n, str):
        return n
    pc = _PC[n[0]]
    i = 1
    while i < len(n) and n[i] in '#b':
        pc += 1 if n[i] == '#' else -1
        i += 1
    return pc + 12 * (int(n[i:]) + 1)


def F(n):
    return 440.0 * 2 ** ((M(n) - 69) / 12)


def scale_notes(lo, hi, pcs):
    return [m for m in range(M(lo), M(hi) + 1) if m % 12 in pcs]


D_MAJ = {2, 4, 6, 7, 9, 11, 1}
D_PENT = {2, 4, 6, 9, 11}


# ----------------------------------------------------------------------------- dsp helpers
def pan2(sig, p):
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.vstack([sig * np.cos(a), sig * np.sin(a)])


class Bus:
    def __init__(self, name):
        self.name = name
        self.x = np.zeros((2, N))

    def add(self, t0, st):
        i0 = int(round(t0 * SR))
        if i0 >= N:
            return
        if i0 < 0:
            st = st[:, -i0:]
            i0 = 0
        L = min(st.shape[1], N - i0)
        self.x[:, i0:i0 + L] += st[:, :L]


def ar_env(L, att, hold, rel):
    t = np.arange(L) / SR
    e = np.ones(L)
    if att > 0:
        a = t < att
        e[a] = 0.5 - 0.5 * np.cos(np.pi * t[a] / att)
    r = t > hold
    x = np.clip((t[r] - hold) / max(rel, 1e-3), 0, 1)
    e[r] *= (0.5 + 0.5 * np.cos(np.pi * x)) ** 1.5
    return e


def db_ramp(L, hold, db0, db1):
    t = np.arange(L) / SR
    x = np.clip(t / max(hold, 1e-3), 0, 1)
    return 10 ** ((db0 + (db1 - db0) * x) / 20)


def shaped_noise(L, wfun, r):
    x = r.standard_normal(L)
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(L, 1 / SR)
    X *= wfun(fr)
    y = np.fft.irfft(X, L)
    return y / (np.std(y) + 1e-12)


def harm(fi, a, ph):
    """Band-limited harmonic oscillator (sum of a_k sin(k*phi + ph_k)) via complex recurrence."""
    phi = 2 * np.pi * np.cumsum(fi) / SR
    z = np.exp(1j * phi)
    w = z.copy()
    out = np.zeros(len(fi))
    c, s = np.cos(ph), np.sin(ph)
    for k in range(len(a)):
        out += a[k] * c[k] * w.imag
        out += a[k] * s[k] * w.real
        w *= z
    return out


# ----------------------------------------------------------------------------- formants (choir "aah")
FORM_S = [(800, 80, 0), (1150, 90, -6), (2900, 120, -32), (3900, 130, -20), (4950, 140, -50)]
FORM_T = [(650, 80, 0), (1080, 90, -6), (2650, 120, -7), (2900, 130, -8), (3250, 140, -22)]


def formant_fun(table):
    def w(fr):
        out = 0.015 + np.zeros_like(fr, dtype=float)
        for Fq, bw, g in table:
            out = out + 10 ** (g / 20) / (1 + ((fr - Fq) / (bw * 1.6)) ** 2)
        return out
    return w


# ----------------------------------------------------------------------------- instruments
def bowed(f, dur, amp, *, att=0.6, rel=0.8, voices=5, detune=8.0, vib=0.0025, vib_rate=5.3,
          fc=2000.0, tilt=1.0, spread=0.6, formant=None, swell=None, seed=None, pan=0.0,
          hmax=64, breath=0.0):
    """Ensemble of detuned band-limited saws ('supersaw' built additively, so no aliasing),
    static low-pass spectral tilt, per-voice vibrato with delayed onset and slow pitch drift."""
    r = np.random.default_rng(seed if seed is not None else nxt())
    L = int((dur + rel) * SR)
    t = np.arange(L) / SR
    H = int(max(1, min(hmax, min(15000.0, fc * 4.5) / f)))
    n = np.arange(1, H + 1)
    fr = n * f
    a = n ** (-tilt) / np.sqrt(1 + (fr / fc) ** 4)
    if formant is not None:
        a = a * formant(fr)
    a /= np.sqrt(np.sum(a ** 2))
    out = np.zeros((2, L))
    onset = np.clip((t - min(att, 0.5) * 0.6) / 0.7, 0, 1)
    for v in range(voices):
        pos = (v / (voices - 1)) * 2 - 1 if voices > 1 else 0.0
        d = detune * pos + r.normal(0, detune * 0.15 + 0.3)
        rate = vib_rate * r.uniform(0.85, 1.15)
        depth = vib * r.uniform(0.7, 1.3)
        drift = 0.0012 * np.sin(2 * np.pi * r.uniform(0.05, 0.25) * t + r.uniform(0, 6.28))
        fi = f * 2 ** (d / 1200) * (1 + depth * onset * np.sin(2 * np.pi * rate * t + r.uniform(0, 6.28)) + drift)
        g = r.uniform(0.8, 1.0)
        s = harm(fi, a * g, r.uniform(0, 2 * np.pi, H))
        out += pan2(s, pan + spread * pos)
    out /= np.sqrt(voices)
    if breath > 0:
        wf = formant if formant is not None else (lambda fr_: 1.0 / (1 + (fr_ / 3000) ** 2))
        nz = shaped_noise(L, lambda fr_: wf(fr_) * (fr_ > 200), r) * breath
        out += pan2(nz, pan)
    env = ar_env(L, att, dur, rel)
    if swell is not None:
        env = env * db_ramp(L, dur, swell[0], swell[1])
    return out * env * amp


def piano(note, dur, vel, seed=None):
    """Soft felt piano: inharmonic additive partials, 1-3 detuned strings per note,
    two-stage decay, felt hammer noise, damper release."""
    r = np.random.default_rng(seed if seed is not None else nxt())
    m = M(note)
    f = F(m)
    T60 = float(np.clip(16 * 2 ** (-(m - 21) / 17), 0.9, 16))
    L = int(min(dur + 0.6, T60 + 0.3) * SR)
    t = np.arange(L) / SR
    B = 7e-5 * 2 ** ((m - 48) / 14)
    fmax = min(9000.0, 1800 + 5500 * vel)
    nstr = 1 if m < 34 else (2 if m < 48 else 3)
    k = 0.10 + 0.30 * (1 - vel)
    felt = 1300 + 3000 * vel
    out = np.zeros(L)
    asum = 0.0
    nn = 1
    while nn <= 48:
        fn = nn * f * np.sqrt(1 + B * nn * nn)
        if fn > fmax:
            break
        a = nn ** -0.8 * np.exp(-(nn - 1) * k) / np.sqrt(1 + (fn / felt) ** 2)
        T = T60 / (1 + 0.10 * (nn - 1) * np.sqrt(f / 130))
        tau = T / 6.91
        env = 0.62 * np.exp(-t / (tau * 0.3)) + 0.38 * np.exp(-t / tau)
        part = np.zeros(L)
        for s in range(nstr):
            dc = (s - (nstr - 1) / 2) * r.uniform(0.4, 1.0)
            part += np.sin(2 * np.pi * fn * 2 ** (dc / 1200) * t + r.uniform(0, 2 * np.pi))
        out += a * env * part / nstr
        asum += a * a
        nn += 1
    out /= np.sqrt(asum)
    att = 0.003 + 0.006 * (1 - vel)
    out *= np.clip(t / att, 0, 1)
    hl = int(0.04 * SR)
    hn = shaped_noise(hl, lambda fr: 1 / (1 + (fr / (900 + 1500 * vel)) ** 2), r)
    hn *= np.exp(-np.arange(hl) / SR / 0.006) * 0.05 * vel
    out[:hl] += hn
    rel = t > dur
    out[rel] *= np.exp(-(t[rel] - dur) / 0.09)
    fl = min(L, int(0.01 * SR))
    out[L - fl:] *= np.linspace(1, 0, fl)
    out *= vel ** 1.6
    p = np.clip((m - 62) / 45, -0.55, 0.55)
    return pan2(out, p)


BELLS = {
    # (ratio, amp, T60 at A5)
    'musicbox': [(1.0, 1.0, 2.8), (2.0, 0.10, 1.2), (2.756, 0.26, 0.9), (5.404, 0.10, 0.45), (8.933, 0.04, 0.22)],
    'celesta':  [(1.0, 1.0, 2.4), (2.0, 0.12, 1.0), (3.0, 0.06, 0.6), (4.0, 0.14, 0.35), (6.27, 0.03, 0.15)],
    'glock':    [(1.0, 1.0, 3.6), (2.71, 0.32, 1.3), (5.15, 0.12, 0.5), (8.8, 0.04, 0.2)],
}


def bell(note, vel, kind='celesta', pan=0.2, seed=None):
    r = np.random.default_rng(seed if seed is not None else nxt())
    f = F(note)
    sc = (F('A5') / f) ** 0.35
    parts = BELLS[kind]
    T = max(p[2] for p in parts) * sc
    L = int(T * 1.05 * SR)
    t = np.arange(L) / SR
    out = np.zeros(L)
    for i, (ra, a, T60) in enumerate(parts):
        fr = f * ra
        if fr > 19000:
            continue
        a = a * (vel ** 0.6 if i > 0 else 1.0)
        env = np.exp(-6.91 * t / (T60 * sc))
        if i < 2:   # slightly detuned pair -> gentle beating like a real bar/tine
            dc = r.uniform(0.6, 1.4)
            s = 0.5 * (np.sin(2 * np.pi * fr * 2 ** (dc / 1200) * t + r.uniform(0, 6.28)) +
                       np.sin(2 * np.pi * fr * 2 ** (-dc / 1200) * t + r.uniform(0, 6.28)))
        else:
            s = np.sin(2 * np.pi * fr * t + r.uniform(0, 6.28))
        out += a * env * s
    att = 0.0015 if kind != 'celesta' else 0.003
    out *= np.clip(t / att, 0, 1)
    tl = int(0.006 * SR)
    tick = shaped_noise(tl, lambda fr: (fr > 3000) / (1 + (fr / 9000) ** 2), r) * np.exp(-np.arange(tl) / SR / 0.0012)
    out[:tl] += tick * 0.04
    return pan2(out * vel, pan + r.uniform(-0.15, 0.15))


def pluck(note, vel, *, T60=None, p=0.13, tilt=1.4, fmax=7000, pan=-0.3, seed=None, noise=0.03):
    """Harp / pizzicato: additive plucked string with pluck-position comb and partial-dependent decay."""
    r = np.random.default_rng(seed if seed is not None else nxt())
    m = M(note)
    f = F(m)
    if T60 is None:
        T60 = float(np.clip(5.0 * 2 ** (-(m - 40) / 18), 0.6, 6.0))
    L = int(T60 * SR)
    t = np.arange(L) / SR
    out = np.zeros(L)
    asum = 0
    for nn in range(1, 31):
        fn = nn * f * np.sqrt(1 + 1.5e-5 * nn * nn)
        if fn > fmax:
            break
        a = (abs(np.sin(nn * np.pi * p)) + 0.02) / nn ** tilt * np.exp(-(nn - 1) * (1 - vel) * 0.15)
        Tn = T60 / (1 + 0.35 * (nn - 1) * (f / 200) ** 0.3)
        out += a * np.exp(-6.91 * t / Tn) * np.sin(2 * np.pi * fn * t + r.uniform(0, 6.28))
        asum += a * a
    out /= np.sqrt(asum)
    out *= np.clip(t / 0.0012, 0, 1)
    nl = int(0.012 * SR)
    nz = shaped_noise(nl, lambda fr: 1 / (1 + ((fr - 2 * f) / 1500) ** 2), r) * np.exp(-np.arange(nl) / SR / 0.003)
    out[:nl] += nz * noise
    return pan2(out * vel, pan + r.uniform(-0.1, 0.1))


def pizz(note, vel, pan=-0.15):
    m = M(note)
    T60 = float(np.clip(1.3 * 2 ** (-(m - 40) / 24), 0.3, 1.3))
    return pluck(note, vel, T60=T60, p=0.22, tilt=1.9, fmax=3500, pan=pan, noise=0.06)


def timpani(note, vel, decay=2.8, pan=-0.1, seed=None):
    r = np.random.default_rng(seed if seed is not None else nxt())
    f0 = F(note)
    L = int(decay * SR)
    t = np.arange(L) / SR
    modes = [(1.0, 1.0, 1.0), (1.504, 0.55, 0.7), (1.742, 0.35, 0.55), (2.0, 0.28, 0.5),
             (2.245, 0.18, 0.4), (2.494, 0.12, 0.35), (2.8, 0.07, 0.3), (2.98, 0.05, 0.25)]
    glide = 1 + 0.018 * vel * np.exp(-t / 0.07)
    out = np.zeros(L)
    for i, (ra, a, dk) in enumerate(modes):
        a = a * (vel ** 0.8 if i > 0 else 1.0)
        ph = 2 * np.pi * np.cumsum(f0 * ra * glide) / SR + r.uniform(0, 6.28)
        out += a * np.exp(-6.91 * t / (decay * dk)) * np.sin(ph)
    tl = int(0.08 * SR)
    th = shaped_noise(tl, lambda fr: 1 / (1 + (fr / 250) ** 2), r) * np.exp(-np.arange(tl) / SR / 0.025)
    out[:tl] += th * 0.5
    cl = int(0.01 * SR)
    out[:cl] += shaped_noise(cl, lambda fr: 1 / (1 + (fr / 2500) ** 2), r) * np.exp(-np.arange(cl) / SR / 0.002) * 0.25 * vel
    out *= np.clip(t / 0.002, 0, 1)
    return pan2(out * vel ** 1.3 * 0.5, pan)


def bassdrum(vel, seed=None):
    r = np.random.default_rng(seed if seed is not None else nxt())
    L = int(3.0 * SR)
    t = np.arange(L) / SR
    f = 44 + 16 * np.exp(-t / 0.08)
    out = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-6.91 * t / 2.6)
    out += 0.35 * np.sin(2 * np.pi * np.cumsum(f * 1.6) / SR) * np.exp(-6.91 * t / 1.2)
    tl = int(0.1 * SR)
    out[:tl] += shaped_noise(tl, lambda fr: 1 / (1 + (fr / 160) ** 2), r) * np.exp(-np.arange(tl) / SR / 0.03) * 0.6
    out *= np.clip(t / 0.003, 0, 1)
    return pan2(out * vel * 0.6, 0.0)


def _metal(L, r, n=48, lo=2500, hi=11000, T=3.0):
    t = np.arange(L) / SR
    out = np.zeros(L)
    for _ in range(n):
        fr = np.exp(r.uniform(np.log(lo), np.log(hi)))
        out += r.uniform(0.3, 1) * np.sin(2 * np.pi * fr * t + r.uniform(0, 6.28)) * np.exp(-6.91 * t / (T * r.uniform(0.3, 1.0)))
    return out / np.sqrt(n)


def cymbal_crash(vel, T60=5.0, seed=None):
    r = np.random.default_rng(seed if seed is not None else nxt())
    L = int(T60 * SR)
    t = np.arange(L) / SR
    lo = shaped_noise(L, lambda fr: (fr / 3000) ** 2 / (1 + (fr / 3000) ** 2) / (1 + (fr / 9000) ** 4), r)
    hi = shaped_noise(L, lambda fr: (fr / 7000) ** 4 / (1 + (fr / 7000) ** 4) / (1 + (fr / 15000) ** 4), r)
    body = (lo * (0.6 * np.exp(-6.91 * t / T60) + 0.4 * np.exp(-t / 0.12)) +
            0.6 * hi * np.exp(-6.91 * t / (T60 * 0.5)) +
            0.8 * _metal(L, r, T=T60))
    body *= np.clip(t / 0.002, 0, 1)
    L2 = shaped_noise(L, lambda fr: (fr / 3000) ** 2 / (1 + (fr / 3000) ** 2) / (1 + (fr / 9000) ** 4), r)
    st = np.vstack([body, 0.7 * body + 0.3 * L2 * (0.6 * np.exp(-6.91 * t / T60))])
    return st * vel * 0.12


def cymbal_roll(dur, a0, a1, seed=None):
    """Suspended cymbal with soft mallets: noise + metallic partials, exponential crescendo."""
    r = np.random.default_rng(seed if seed is not None else nxt())
    L = int((dur + 0.03) * SR)
    t = np.arange(L) / SR
    x = np.clip(t / dur, 0, 1)
    env = a0 * (a1 / a0) ** (x ** 1.3)
    env *= 1 + 0.25 * np.sin(2 * np.pi * 13.5 * t) * np.sin(2 * np.pi * 0.7 * t + 1)
    env[t > dur] *= np.exp(-(t[t > dur] - dur) / 0.006)
    nzl = shaped_noise(L, lambda fr: (fr / 2500) ** 2 / (1 + (fr / 2500) ** 2) / (1 + (fr / 7000) ** 4), r)
    nzr = shaped_noise(L, lambda fr: (fr / 2500) ** 2 / (1 + (fr / 2500) ** 2) / (1 + (fr / 7000) ** 4), r)
    met = _metal(L, r, T=1e6)   # sustained shimmer partials
    return np.vstack([nzl + 0.7 * met, nzr + 0.7 * met]) * env * 0.1


def shimmer_noise(dur, a0, a1, seed=None):
    r = np.random.default_rng(seed if seed is not None else nxt())
    L = int(dur * SR)
    t = np.arange(L) / SR
    env = a0 + (a1 - a0) * (t / dur) ** 2
    env *= np.clip((dur - t) / 0.05, 0, 1)
    w = lambda fr: np.exp(-0.5 * (np.log(np.maximum(fr, 1) / 8000) / 0.35) ** 2)
    return np.vstack([shaped_noise(L, w, r), shaped_noise(L, w, r)]) * env * 0.022


# ----------------------------------------------------------------------------- score
DIALOGUE = [(1.0, 10.06), (13.4, 16.08), (17.2, 22.43), (22.9, 24.95), (25.4, 28.1),
            (28.8, 30.72), (39.0, 42.25), (43.0, 44.65), (45.3, 48.93), (51.0, 58.54)]


def compose():
    B = {k: Bus(k) for k in ['strings', 'bass', 'choir', 'horn', 'piano', 'celesta', 'harp', 'perc', 'cymbal', 'fx']}
    hum = np.random.default_rng(99)

    def H(t, amt=0.012):        # rubato / humanize
        return t + hum.uniform(-amt, amt)

    def pad(t0, t1, notes, amp, **kw):
        kw.setdefault('att', 1.0)
        kw.setdefault('rel', 1.2)
        kw.setdefault('fc', 900)
        for nm in notes:
            B['strings'].add(t0, bowed(F(nm), t1 - t0, amp, **kw))

    def line(bus, notes, amp, **kw):
        """legato melodic line: list of (t, dur, note) ; amp may be callable(i)"""
        for i, (t, d, nm) in enumerate(notes):
            a = amp(i) if callable(amp) else amp
            B[bus].add(t, bowed(F(nm), d, a, **kw))

    def bass(t0, t1, nm, amp=0.10, **kw):
        kw.setdefault('att', 0.6)
        kw.setdefault('rel', 1.0)
        B['bass'].add(t0, bowed(F(nm), t1 - t0, amp * 0.75, voices=2, detune=3, fc=330, tilt=1.1,
                                vib=0.0012, spread=0.15, **kw))

    def choir(t0, t1, notes, amp, **kw):
        kw.setdefault('att', 0.6)
        kw.setdefault('rel', 1.2)
        for nm in notes:
            tab = FORM_S if M(nm) >= 60 else FORM_T
            B['choir'].add(t0, bowed(F(nm), t1 - t0, amp, voices=6, detune=11, vib=0.006, vib_rate=5.0,
                                     fc=5000, tilt=1.0, spread=0.7, formant=formant_fun(tab), breath=0.08, **kw))

    def horn(t0, t1, nm, amp, **kw):
        kw.setdefault('att', 0.12)
        kw.setdefault('rel', 0.5)
        f = F(nm)
        B['horn'].add(t0, bowed(f, t1 - t0, amp, voices=2, detune=4, vib=0.0018, fc=f * 3.5 + 300, tilt=1.3,
                                spread=0.2, pan=-0.2, **kw))

    def pno(t, nm, dur, vel):
        B['piano'].add(t, piano(nm, dur, vel))

    def cel(t, nm, vel, kind='celesta', pan=0.25):
        B['celesta'].add(t, bell(nm, vel, kind, pan=pan))

    def harp(t, nm, vel, **kw):
        B['harp'].add(t, pluck(nm, vel, **kw))

    def roll(t, notes, vel, sp=0.07):
        for i, nm in enumerate(notes):
            harp(t + i * sp, nm, vel * (0.85 + 0.3 * i / max(1, len(notes) - 1)))

    def gliss(t0, t1, lo, hi, pcs, v0, v1, power=0.7, down=False):
        ns = scale_notes(lo, hi, pcs)
        if down:
            ns = ns[::-1]
        K = len(ns)
        for i, m_ in enumerate(ns):
            u = i / (K - 1)
            tt = t0 + (t1 - t0) * (u ** power if not down else 1 - (1 - u) ** power)
            harp(tt, m_, v0 + (v1 - v0) * u, T60=float(np.clip(4.0 * 2 ** (-(m_ - 40) / 18), 0.5, 4.0)))

    def pings(t0, t1, n, pcs, octs, v0, v1, kind='glock', density_pow=0.5, seed=1):
        r = np.random.default_rng(seed)
        us = np.sort(r.uniform(0, 1, n)) ** density_pow
        cands = [m_ for m_ in range(12 * (octs[0] + 1), 12 * (octs[1] + 2)) if m_ % 12 in pcs]
        for i, u in enumerate(us):
            cel(t0 + (t1 - t0) * u, int(r.choice(cands)), v0 + (v1 - v0) * u, kind, pan=r.uniform(-0.6, 0.6))

    # ================= S1 OPENING 0–10.5 : music box over soft pad (D | Bm7 | Gmaj7 | A7) =====
    pad(0.0, 4.6, ['D3', 'A3', 'E4', 'F#4'], 0.038, att=2.8, rel=1.4)
    pad(4.5, 7.6, ['B2', 'F#3', 'A3', 'D4'], 0.038, att=1.2, rel=1.4)
    pad(7.5, 9.1, ['G2', 'D3', 'F#3', 'B3'], 0.038, att=1.0, rel=1.2)
    pad(9.0, 10.7, ['A2', 'E3', 'G3', 'C#4'], 0.042, att=1.0, rel=0.9, fc=1100)
    bass(1.5, 4.6, 'D2', 0.065, att=1.2)
    bass(4.5, 7.6, 'B1', 0.065)
    bass(7.5, 9.1, 'G1', 0.065)
    bass(9.0, 10.8, 'A1', 0.07)
    mb = [(0.75, 'A5'), (1.5, 'D6'), (2.25, 'E6'), (3.0, 'F#6'), (4.5, 'E6'), (5.25, 'D6'), (6.0, 'B5'),
          (7.125, 'A5'), (7.5, 'B5'), (8.25, 'D6'), (9.0, 'A5'), (9.375, 'C#6'), (9.78, 'E6')]
    for i, (t, nm) in enumerate(mb):
        cel(H(t + (0.04 if i in (3, 7, 12) else 0)), nm, 0.34 if i else 0.26, 'musicbox', pan=0.15)
    for t, nm in [(1.5, 'D4'), (1.53, 'A4'), (4.5, 'B3'), (4.53, 'F#4'), (7.5, 'G3'), (7.53, 'D4'), (9.0, 'A3'), (9.03, 'E4')]:
        cel(H(t), nm, 0.20, 'musicbox', pan=-0.2)
    for t, nm in [(0.25, 'D7'), (2.6, 'A7'), (5.6, 'F#7'), (8.6, 'E7')]:
        cel(t, nm, 0.09, 'glock', pan=0.5)
    roll(1.5, ['D2', 'A2', 'D3', 'F#3', 'A3'], 0.30)
    roll(4.5, ['B1', 'F#2', 'B2', 'D3', 'F#3'], 0.28)
    roll(7.5, ['G1', 'D2', 'G2', 'B2', 'D3'], 0.28)
    roll(9.0, ['A1', 'E2', 'A2', 'C#3', 'E3'], 0.28)

    # ================= S2 SHOOTING STAR 10.5–12.5 : harp gliss + shimmer, stinger at 12.5 ==========
    pad(10.5, 12.5, ['A2', 'E3', 'A3', 'B3', 'E4'], 0.055, att=1.5, rel=0.08, fc=1700, swell=(-9, 0))
    pad(10.6, 12.5, ['A5', 'E6'], 0.018, att=1.4, rel=0.08, fc=8000, swell=(-12, 0), voices=6)
    bass(10.5, 12.5, 'A1', 0.1, rel=0.1)
    gliss(10.6, 12.42, 'A2', 'E7', D_MAJ, 0.25, 0.62, power=0.7)
    pings(11.0, 12.45, 16, D_PENT, (6, 7), 0.08, 0.2, seed=11)
    B['cymbal'].add(11.3, cymbal_roll(1.2, 0.03, 0.15))
    B['fx'].add(10.8, shimmer_noise(1.7, 0.1, 1.0))
    # stinger (impact 12.5)
    B['perc'].add(12.5, timpani('D2', 0.42))
    B['perc'].add(12.5, bassdrum(0.3))
    pad(12.5, 12.72, ['D3', 'F#3', 'A3', 'D4', 'F#4', 'A4'], 0.05, att=0.012, rel=0.55, fc=3500, voices=6)
    bass(12.5, 12.7, 'D2', 0.10, att=0.01, rel=0.5)
    for i, nm in enumerate(['D7', 'A6', 'F#6', 'D6', 'A5']):
        cel(12.5 + i * 0.03, nm, 0.2 - 0.025 * i, 'glock', pan=0.4 - 0.2 * i)
    B['cymbal'].add(12.5, cymbal_crash(0.22, T60=3.0))

    # ----- curious staccato 12.6–17.0 : pizzicato tiptoe G | A | F#m | F#7  -> Bm
    e = 0.367
    pz = ['G2', 'D3', 'G3', 'A2', 'E3', 'A3', 'F#2', 'C#3', 'F#3', 'F#2', 'A#2', 'C#3']
    for k, nm in enumerate(pz):
        t = 12.75 + k * e
        acc = 1.0 if k % 3 == 0 else 0.75
        B['strings'].add(H(t, 0.008), pizz(nm, 0.3 * acc))
        if k % 3 == 2:   # little upper answer on the off-beat
            B['strings'].add(H(t + e * 0.5, 0.006), pizz(M(nm) + 12, 0.16, pan=0.25))
    pad(12.8, 13.95, ['G3', 'B3', 'D4'], 0.022, att=0.4, rel=0.5, fc=800)
    pad(13.85, 15.05, ['A3', 'C#4', 'E4'], 0.022, att=0.4, rel=0.5, fc=800)
    pad(14.95, 16.1, ['F#3', 'A3', 'C#4'], 0.022, att=0.4, rel=0.5, fc=800)
    pad(16.0, 17.2, ['F#3', 'A#3', 'C#4', 'E4'], 0.024, att=0.4, rel=0.6, fc=800)
    for t, nm in [(12.92, 'F#6'), (13.07, 'A6'), (13.22, 'E7')]:
        cel(t, nm, 0.26)
    for t, nm in [(16.12, 'A#5'), (16.32, 'C#6'), (16.52, 'E6'), (16.72, 'F#6')]:
        cel(t, nm, 0.18)

    # ================= S3 MEETING 17–28.6 : B minor, sparse felt piano, very quiet ================
    pad(17.0, 20.0, ['B2', 'F#3', 'C#4', 'D4'], 0.036, att=1.4, rel=1.4, fc=700)
    pad(19.9, 22.9, ['G2', 'D3', 'F#3', 'B3'], 0.036, rel=1.4, fc=700)
    pad(22.8, 25.8, ['E2', 'B2', 'F#3', 'G3', 'D4'], 0.032, rel=1.4, fc=700)
    pad(25.7, 27.3, ['A2', 'E3', 'G3', 'D4'], 0.036, rel=1.0, fc=750)
    pad(27.2, 28.8, ['A2', 'E3', 'G3', 'C#4'], 0.038, att=0.8, rel=0.8, fc=850)
    bass(17.0, 20.0, 'B1', 0.085, att=1.0)
    bass(19.9, 22.9, 'G1', 0.085)
    bass(22.8, 25.8, 'E2', 0.08)
    bass(25.7, 28.8, 'A1', 0.085)
    P = [(17.0, 'B1', 2.95, 0.36), (17.0, 'B2', 2.95, 0.30), (17.6, 'D5', 2.4, 0.24), (18.3, 'F#5', 1.7, 0.22),
         (19.0, 'C#6', 1.0, 0.20),
         (19.9, 'G1', 2.95, 0.34), (19.9, 'G2', 2.95, 0.28), (20.5, 'B4', 2.3, 0.21), (21.2, 'D5', 1.6, 0.20),
         (21.9, 'F#5', 0.9, 0.20),
         (22.46, 'B5', 0.5, 0.30), (22.66, 'A5', 0.4, 0.28),
         (22.8, 'E2', 2.95, 0.33), (22.8, 'B2', 2.95, 0.26), (23.5, 'G4', 2.2, 0.20), (24.2, 'D5', 0.8, 0.20),
         (24.98, 'F#5', 0.3, 0.27), (25.16, 'E5', 0.5, 0.25),
         (25.7, 'A1', 2.95, 0.33), (25.7, 'A2', 2.95, 0.26), (26.4, 'E4', 2.2, 0.20), (27.15, 'G4', 1.5, 0.20),
         (27.3, 'C#5', 1.3, 0.20),
         (28.12, 'A4', 0.6, 0.28), (28.27, 'C#5', 0.5, 0.31), (28.42, 'E5', 0.5, 0.34)]
    for t, nm, d, v in P:
        pno(H(t, 0.01), nm, d, v)

    # ================= S4a DETERMINATION 28.6–33.5 : major lift, pulse begins, building =========
    e = 0.30625
    pad(28.6, 29.9, ['D3', 'A3', 'E4', 'F#4'], 0.040, att=0.35, rel=0.6, fc=950)
    pad(29.825, 31.1, ['C#3', 'A3', 'E4'], 0.044, att=0.4, rel=0.6, fc=1050)
    pad(31.05, 32.35, ['B2', 'F#3', 'A3', 'D4'], 0.05, att=0.4, rel=0.6, fc=1200)
    pad(32.275, 32.95, ['G2', 'D3', 'B3', 'D4'], 0.056, att=0.3, rel=0.5, fc=1350)
    pad(32.9, 33.6, ['A2', 'E3', 'A3', 'C#4'], 0.062, att=0.3, rel=0.5, fc=1500, swell=(-2, 1))
    bass(28.6, 29.9, 'D2', 0.10, att=0.3)
    bass(29.825, 31.1, 'C#2', 0.10, att=0.3)
    bass(31.05, 32.35, 'B1', 0.11, att=0.3)
    bass(32.275, 32.95, 'G1', 0.12, att=0.2)
    bass(32.9, 33.6, 'A1', 0.12, att=0.2)
    pulse = [('D3', 'A3')] * 4 + [('C#3', 'A3')] * 4 + [('B2', 'F#3')] * 4 + [('G2', 'D3')] * 2 + [('A2', 'E3')] * 2
    for k in range(16):
        t = 28.6 + k * e
        lo, hi = pulse[k]
        a = 0.028 * 10 ** (k / 15 * 8 / 20) * (1.25 if k % 2 == 0 else 0.9)
        nm = lo if k % 2 == 0 else hi
        B['strings'].add(t, bowed(F(nm), 0.13, a, att=0.012, rel=0.14, voices=3, detune=6, vib=0, fc=2200))
        B['strings'].add(t, bowed(F(nm) * 2, 0.13, a * 0.5, att=0.012, rel=0.14, voices=3, detune=6, vib=0, fc=2600))
    arp = [(31.05, ['B3', 'D4', 'F#4', 'B4']), (31.66, ['A3', 'D4', 'F#4', 'A4']),
           (32.275, ['G3', 'B3', 'D4', 'G4']), (32.9, ['A3', 'C#4', 'E4', 'A4'])]
    for j, (t0, ns) in enumerate(arp):
        step = e / 2 if j < 3 else 0.15
        for i, nm in enumerate(ns):
            pno(t0 + i * step, nm, 0.9, 0.30 + 0.04 * j + 0.02 * i)
    horn(31.05, 31.66, 'D4', 0.035)
    horn(31.66, 32.275, 'F#4', 0.04)
    horn(32.275, 32.9, 'G4', 0.045)
    horn(32.9, 33.55, 'A4', 0.05, swell=(-1, 2))
    B['perc'].add(32.275, timpani('G2', 0.32))
    B['perc'].add(32.9, timpani('A2', 0.40))

    # ================= S4b BEAM & CHARGING 33.5–37.6 : G A Bm Bb C -> D (bVI-bVII-I) =============
    Ts = [33.5, 34.3, 35.1, 35.9, 36.7, 37.6]
    V = [['G2', 'D3', 'B3', 'D4', 'G4', 'B4'], ['A2', 'E3', 'C#4', 'E4', 'A4', 'C#5'],
         ['B2', 'F#3', 'D4', 'F#4', 'B4', 'D5'], ['Bb2', 'F3', 'D4', 'F4', 'Bb4', 'D5'],
         ['C3', 'G3', 'E4', 'G4', 'C5', 'D5', 'E5']]
    CH = [['G3', 'B3', 'D4', 'G4'], ['A3', 'C#4', 'E4', 'A4'], ['B3', 'D4', 'F#4', 'B4'],
          ['Bb3', 'D4', 'F4', 'Bb4'], ['C4', 'E4', 'G4', 'C5', 'D5']]
    BS = ['G1', 'A1', 'B1', 'Bb1', 'C2']
    HN = ['G4', 'A4', 'B4', 'Bb4', 'C5']
    for k in range(5):
        t0, t1 = Ts[k], Ts[k + 1]
        g = 10 ** (k * 1.2 / 20) * 0.9
        last = k == 4
        pad(t0, t1 + 0.04, V[k], 0.05 * g, att=0.22, rel=0.06 if last else 0.3, fc=1400 + 550 * k, voices=6,
            swell=(-2, 1.5))
        choir(t0, t1 + 0.04, CH[k], 0.028 * g * (1.25 if k >= 3 else 1.0), att=0.35 if k == 0 else 0.2,
              rel=0.06 if last else 0.3, swell=(-2, 1.5))
        horn(t0, t1 + 0.02, HN[k], 0.05 * g, att=0.1, rel=0.06 if last else 0.25)
        bass(t0, t1 + 0.04, BS[k], 0.12 * g, att=0.15, rel=0.06 if last else 0.3)
    mel = [(33.5, 0.42, 'B4'), (33.9, 0.42, 'D5'), (34.3, 0.42, 'C#5'), (34.7, 0.42, 'E5'), (35.1, 0.42, 'D5'),
           (35.5, 0.42, 'F#5'), (35.9, 0.42, 'F5'), (36.3, 0.42, 'A5'), (36.7, 0.32, 'E5'), (37.0, 0.32, 'G5'),
           (37.3, 0.28, 'C6')]
    line('strings', mel, lambda i: 0.05 * 10 ** (i * 7 / 10 / 20), att=0.07, rel=0.2, voices=6, fc=3500, vib=0.004)
    line('strings', [(t, d, M(nm) - 12) for t, d, nm in mel], lambda i: 0.03 * 10 ** (i * 7 / 10 / 20),
         att=0.07, rel=0.2, voices=5, fc=2500, vib=0.004)
    # timpani roll (D pedal) 35.0 -> 37.58
    rr = np.random.default_rng(5)
    t = 35.0
    while t < 37.56:
        u = (t - 35.0) / 2.58
        B['perc'].add(t, timpani('D2', 0.07 + 0.33 * u ** 1.8, decay=1.4, pan=-0.1 + 0.08 * rr.uniform(-1, 1)))
        t += 0.058 + rr.uniform(-0.008, 0.008)
    B['cymbal'].add(35.6, cymbal_roll(2.0, 0.025, 0.24))
    B['fx'].add(35.0, shimmer_noise(2.6, 0.15, 1.2))
    gliss(36.95, 37.56, 'C3', 'C7', {0, 2, 4, 6, 7, 9, 11}, 0.22, 0.5, power=0.8)
    pings(35.0, 37.55, 34, {2, 4, 6, 9, 11}, (6, 7), 0.06, 0.17, density_pow=0.6, seed=7)

    # ================= BURST 37.6 : glorious D major ===========================================
    tb = 37.6
    pad(tb, tb + 1.15, ['D2', 'A2', 'D3', 'A3', 'D4', 'F#4', 'A4', 'D5', 'F#5', 'A5'], 0.2, att=0.02, rel=1.3,
        fc=4500, voices=6, swell=(0, -7))
    line('strings', [(tb, 1.15, 'D6')], 0.15, att=0.02, rel=1.3, voices=6, fc=6000, vib=0.005, swell=(0, -6))
    choir(tb, tb + 1.2, ['A3', 'D4', 'F#4', 'A4', 'D5', 'F#5'], 0.15, att=0.05, rel=1.4, swell=(0, -6))
    for nm in ['D4', 'F#4', 'A4', 'D5']:
        horn(tb, tb + 1.1, nm, 0.13, att=0.04, rel=1.0, swell=(0, -6))
    bass(tb, tb + 1.5, 'D2', 0.16, att=0.02, rel=1.2)
    bass(tb, tb + 1.5, 'D1', 0.10, att=0.02, rel=1.2)
    B['perc'].add(tb, timpani('D2', 1.0, decay=3.2))
    B['perc'].add(tb, timpani('A1', 0.6, decay=2.5, pan=0.1))
    B['perc'].add(tb, bassdrum(1.0))
    B['cymbal'].add(tb, cymbal_crash(1.0, T60=5.0))
    for nm in ['D1', 'D2', 'A2', 'D5', 'F#5', 'A5', 'D6']:
        pno(tb, nm, 2.2, 0.36)
    casc = scale_notes('A4', 'D7', D_PENT)[::-1]
    for i, m_ in enumerate(casc):
        cel(tb + i * 0.04, m_, 0.26 - 0.12 * i / len(casc), 'glock', pan=0.6 - 1.2 * i / len(casc))
    gliss(tb + 0.03, tb + 0.9, 'D3', 'D7', D_MAJ, 0.6, 0.35, power=0.75, down=True)
    for t, nm in [(38.5, 'A6'), (38.62, 'D7'), (38.74, 'E7'), (38.86, 'F#7')]:
        cel(t, nm, 0.28)

    # ================= S5 FAREWELL 39.0–48.6 : main theme on piano over descending bass ==========
    S5 = [(39.0, 40.7, ['D3', 'A3', 'F#4'], 'D2'), (40.6, 42.3, ['C#3', 'A3', 'E4'], 'C#2'),
          (42.2, 43.9, ['B2', 'F#3', 'A3', 'D4'], 'B1'), (43.8, 45.5, ['A2', 'F#3', 'C#4'], 'A1'),
          (45.4, 47.1, ['G2', 'D3', 'F#3', 'B3'], 'G1'), (47.0, 48.7, ['F#2', 'D3', 'A3'], 'F#1')]
    for i, (t0, t1, ns, bn) in enumerate(S5):
        pad(t0, t1, ns, 0.032, att=1.2 if i == 0 else 0.9, rel=1.1, fc=900)
        bass(t0, t1, bn, 0.075, att=0.6 if i == 0 else 0.4)
    LH = [(39.0, ['D2', 'A2', 'F#3']), (40.6, ['C#2', 'A2', 'E3']), (42.2, ['B1', 'F#2', 'D3']),
          (43.8, ['A1', 'E2', 'C#3']), (45.4, ['G1', 'D2', 'B2']), (47.0, ['F#2', 'D3', 'A3'])]
    for t0, ns in LH:
        for i, nm in enumerate(ns):
            pno(H(t0 + i * 0.4, 0.01), nm, 1.6 - i * 0.4 + 0.15, 0.27 if i == 0 else 0.18)
    MEL = [(39.0, 'A4', 0.8), (39.8, 'D5', 0.8), (40.6, 'E5', 0.8), (41.4, 'F#5', 0.85), (42.27, 'E5', 0.35),
           (42.62, 'D5', 0.38), (43.0, 'B4', 1.6), (44.62, 'A4', 0.8), (45.4, 'B4', 0.8), (46.2, 'D5', 0.8),
           (47.0, 'F#5', 1.2), (48.2, 'E5', 0.45)]
    for t, nm, d in MEL:
        pno(H(t + 0.02, 0.012), nm, d + 0.1, 0.40)
    roll(45.4, ['G2', 'D3', 'B3'], 0.22, sp=0.09)
    roll(47.0, ['F#2', 'D3', 'A3'], 0.22, sp=0.09)

    # ================= RISE 48.6–51.0 : soaring line + harp gliss =================================
    pad(48.6, 49.5, ['E2', 'B2', 'D3', 'F#3', 'G3'], 0.045, att=0.5, rel=0.6, fc=1200)
    pad(49.4, 50.3, ['A2', 'E3', 'G3', 'D4'], 0.05, att=0.4, rel=0.6, fc=1300)
    pad(50.2, 51.1, ['A2', 'E3', 'G3', 'C#4'], 0.055, att=0.4, rel=0.6, fc=1400)
    bass(48.6, 49.5, 'E2', 0.09)
    bass(49.4, 51.1, 'A1', 0.10)
    asc = ['E4', 'F#4', 'G4', 'A4', 'B4', 'C#5', 'D5', 'E5']
    ln = [(48.6 + k * 0.3, 0.33, nm) for k, nm in enumerate(asc)]
    line('strings', ln, lambda i: 0.045 * 10 ** (i * 4 / 7 / 20), att=0.1, rel=0.25, voices=6, fc=3000, vib=0.004)
    line('strings', [(t, d, M(nm) + 12) for t, d, nm in ln], lambda i: 0.018 * 10 ** (i * 4 / 7 / 20),
         att=0.1, rel=0.25, voices=6, fc=5000, vib=0.004)
    line('strings', [(51.0, 1.3, 'F#5')], 0.05, att=0.1, rel=1.0, voices=6, fc=3000, vib=0.004, swell=(0, -9))
    gliss(48.65, 50.9, 'D3', 'A6', D_MAJ, 0.32, 0.5, power=0.85)
    choir(48.8, 50.3, ['G3', 'B3', 'E4'], 0.018, att=0.8, rel=0.6)
    choir(50.2, 51.6, ['A3', 'C#4', 'G4'], 0.02, att=0.4, rel=0.9)
    pings(48.8, 51.5, 12, D_PENT, (6, 7), 0.16, 0.10, density_pow=1.0, seed=21)

    # ================= S6 CONSTELLATION 51.0–60 : motif on celesta + strings, plagal cadence =====
    S6 = [(51.0, 52.5, ['D3', 'A3', 'E4', 'F#4'], 'D2'), (52.4, 53.9, ['B2', 'F#3', 'A3', 'D4'], 'B1'),
          (53.8, 55.3, ['G2', 'D3', 'F#3', 'B3'], 'G1'), (55.2, 55.95, ['G2', 'D3', 'B3', 'E4'], None),
          (55.85, 56.6, ['G2', 'D3', 'Bb3', 'E4'], None)]
    for t0, t1, ns, bn in S6:
        pad(t0, t1, ns, 0.033, att=0.8, rel=1.0, fc=900)
    bass(51.0, 52.5, 'D2', 0.08)
    bass(52.4, 53.9, 'B1', 0.08)
    bass(53.8, 56.6, 'G1', 0.08)
    pad(56.5, 59.6, ['D2', 'A2', 'D3', 'F#3', 'A3', 'E4', 'F#4', 'A4'], 0.036, att=0.7, rel=2.2, fc=1000)
    bass(56.5, 59.6, 'D2', 0.085, rel=2.0)
    bass(56.5, 59.6, 'D1', 0.05, rel=2.0)
    TH = [(51.0, 'A5', 0.7), (51.7, 'D6', 0.7), (52.4, 'F#6', 1.4), (53.8, 'E6', 0.7), (54.5, 'D6', 0.35),
          (54.85, 'B5', 0.35), (55.2, 'D6', 0.65), (55.85, 'E6', 0.65), (56.5, 'F#6', 2.6)]
    for t, nm, d in TH:
        cel(H(t + 0.015), nm, 0.33, 'celesta', pan=0.2)
        cel(H(t + 0.015), nm, 0.14, 'musicbox', pan=0.3)
    line('strings', [(t, d + 0.05, M(nm) - 12) for t, nm, d in TH], 0.022, att=0.15, rel=0.5, voices=6,
         fc=1800, vib=0.004)
    for t, nm in [(52.8, 'A7'), (53.6, 'F#7'), (54.3, 'D7'), (55.0, 'E7'), (55.6, 'B6')]:
        cel(t, nm, 0.08, 'glock', pan=-0.5)
    for t, nm, v in [(56.78, 'A6', 0.24), (57.05, 'D7', 0.21), (57.35, 'E7', 0.18), (57.7, 'F#7', 0.16),
                     (58.2, 'A7', 0.10)]:
        cel(t, nm, v, 'celesta', pan=0.4)
    roll(56.5, ['D2', 'A2', 'D3', 'F#3', 'A3', 'D4', 'F#4', 'A4'], 0.30, sp=0.07)
    choir(56.5, 59.6, ['D4', 'F#4', 'A4'], 0.010, att=1.0, rel=2.0)
    pno(56.5, 'D1', 3.4, 0.28)
    pno(56.5, 'D2', 3.4, 0.22)
    return B


CUES = """\
music.wav cue sheet — 「ほしのとうだい」 original score (48 kHz / 16-bit / stereo / 60.000 s)
Key D major (B minor colour in S3).  Main motif = A D E F# | E D B A (5-1-2-3 | 2-1-6-5).
Dialogue windows get an automatic -5 dB dip in 300 Hz-3 kHz on the whole score.

 0.00-10.50  S1 Opening      music-box states the main motif over soft string pad + contrabass + harp rolls.
                             D(add9) | Bm7 | Gmaj7 | A7, ~80 BPM, faint glock twinkles (stars).  Fade-in from 0.
10.50-12.50  S2 Fall         rising harp glissando (A2->E7, accelerating), high string harmonics, glock shimmer,
                             noise shimmer + suspended-cymbal swell on A pedal (crescendo).
12.50        STINGER         timpani D + bass drum + short string stab D major + glock splash + soft crash.
12.60-17.00  Curious         staccato pizzicato tiptoe  G | A | F#m | F#7 (-> Bm), celesta questions at 12.9 / 16.1.
17.00-28.60  S3 Meeting      B minor, very quiet: Bm(add9) | Gmaj7 | Em9 | A7sus4->A7.  Sparse felt piano,
                             bass notes on chord changes, melodic sighs only in dialogue gaps (22.46, 24.98);
                             hopeful rising piano A-C#-E at 28.12 into the major lift.
28.60-33.50  S4a Determination  D | A/C# | Bm | G | A at 98 BPM: pulsing spiccato eighths begin and grow
                             (+8 dB), piano arpeggios from 31.05, horn line D-F#-G-A, timpani taps 32.28 / 32.90.
33.50-37.60  S4b Beam & charge  rising sequence G | A | Bm | Bb | C (bVI-bVII), choir 'aah' enters 33.5,
                             strings soar B4->C6, horns, timpani roll on D from 35.0, cymbal roll 35.6,
                             building glock shimmer, harp gliss up 36.95 -> peak.
37.60        BURST           tutti D major (strings D2-D6, choir, horns, piano, contrabass D1/D2),
                             timpani + bass drum + crash cymbal, glock cascade, harp gliss down,
                             celesta motif quote 38.5.  Decays toward 39.0.
39.00-48.60  S5 Farewell     main theme on soft piano over descending bass D C# B A G F# (warm, quiet).
48.60-51.00  Rise            Em9 | A7sus4 | A7 ; soaring string line E4->E5 (+octave), harp gliss up, choir halo.
51.00-56.50  S6 Constellation  main motif on celesta (+music box, strings an octave below):
                             D | Bm7 | Gmaj7 | G6 -> Gm6 (minor plagal), soft star pings 52.8-55.6.
56.50        TITLE CADENCE   arrival on D(add9): harp roll, celesta arpeggio upward, low piano D, choir pp.
57.50-60.00  Ring out        reverb tail; master fade 58.4 -> 60.0 (exact digital silence at 60.000).
"""


# ----------------------------------------------------------------------------- reverb & master
def make_ir(T60=2.5, length=3.6, seed=3):
    r = np.random.default_rng(seed)
    L = int(length * SR)
    t = np.arange(L) / SR
    bands = [((None, 250), T60 * 1.15), ((250, 2000), T60), ((2000, 6000), T60 * 0.72), ((6000, None), T60 * 0.45)]
    irs = []
    for ch in range(4):
        x = r.standard_normal(L)
        y = np.zeros(L)
        for (lo, hi), T in bands:
            if lo is None:
                sos = signal.butter(4, hi, 'low', fs=SR, output='sos')
            elif hi is None:
                sos = signal.butter(4, lo, 'high', fs=SR, output='sos')
            else:
                sos = signal.butter(2, [lo, hi], 'band', fs=SR, output='sos')
            y += signal.sosfilt(sos, x) * np.exp(-6.91 * t / T)
        pre = 0.022
        y *= np.clip((t - pre) / 0.045, 0, 1) ** 1.5
        for _ in range(10):    # sparse early reflections
            d = r.uniform(0.008, 0.075)
            i = int(d * SR)
            y[i] += r.choice([-1, 1]) * r.uniform(0.3, 1.0) * 6 * np.exp(-d / 0.06)
        y = signal.sosfilt(signal.butter(2, 9000, 'low', fs=SR, output='sos'), y)
        y = signal.sosfilt(signal.butter(2, 90, 'high', fs=SR, output='sos'), y)
        irs.append(y / np.sqrt(np.sum(y ** 2)))
    return irs


def chorus(x, mix=0.35):
    n = np.arange(x.shape[1], dtype=float)
    t = n / SR
    y = x.copy()
    for ch in range(2):
        for k, (base, depth, rate) in enumerate([(0.011, 0.0025, 0.31), (0.017, 0.003, 0.23)]):
            d = (base + depth * np.sin(2 * np.pi * rate * t + ch * 1.7 + k * 2.1)) * SR
            y[ch] += mix * 0.5 * np.interp(n - d, n, x[ch if k == 0 else 1 - ch], left=0)
    return y / (1 + mix * 0.5)


def dialogue_dip(x, depth=0.45):
    g = np.zeros(N)
    t = np.arange(N) / SR
    for a, b in DIALOGUE:
        ramp = np.clip(np.minimum((t - (a - 0.25)) / 0.25, ((b + 0.2) - t) / 0.25), 0, 1)
        g = np.maximum(g, ramp)
    sos = signal.butter(1, [300, 3000], 'band', fs=SR, output='sos')
    out = x.copy()
    for ch in range(2):
        bp = signal.sosfiltfilt(sos, x[ch])
        out[ch] = x[ch] - depth * g * bp
    return out


def compress(x, thr_db=-15.0, ratio=1.8, tau=0.06, gtau=0.08):
    p = (x[0] ** 2 + x[1] ** 2) / 2
    al = np.exp(-1 / (tau * SR))
    env = signal.lfilter([1 - al], [1, -al], p)
    lvl = 10 * np.log10(env + 1e-12)
    gr = -np.maximum(lvl - thr_db, 0) * (1 - 1 / ratio)
    ag = np.exp(-1 / (gtau * SR))
    gr = signal.lfilter([1 - ag], [1, -ag], gr)
    return x * 10 ** (gr / 20), gr


def limiter(x, ceil_db=-1.2):
    c = 10 ** (ceil_db / 20)
    pk = np.maximum(np.abs(x[0]), np.abs(x[1]))
    need = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    W1 = int(0.003 * SR)
    g = minimum_filter1d(need, 2 * W1 + 1)
    g = uniform_filter1d(g, W1 + 1)
    W2 = int(0.06 * SR)
    g2 = minimum_filter1d(g, 2 * W2 + 1)
    g2 = uniform_filter1d(g2, W2 + 1)
    g = np.minimum(g, g2 * 0.5 + g * 0.5)
    y = x * g
    return np.clip(y, -c, c), g


def master(B):
    sends = {'strings': 0.38, 'bass': 0.12, 'choir': 0.55, 'horn': 0.35, 'piano': 0.32, 'celesta': 0.55,
             'harp': 0.38, 'perc': 0.25, 'cymbal': 0.28, 'fx': 0.5}
    gains = {'strings': 1.0, 'bass': 1.0, 'choir': 1.0, 'horn': 1.0, 'piano': 1.0, 'celesta': 0.9,
             'harp': 0.30, 'perc': 1.0, 'cymbal': 1.0, 'fx': 1.0}
    B['strings'].x = chorus(B['strings'].x)
    B['choir'].x = chorus(B['choir'].x, 0.25)
    dry = np.zeros((2, N))
    send = np.zeros((2, N))
    for k, b in B.items():
        dry += b.x * gains[k]
        send += b.x * gains[k] * sends[k]
    irA, irB, irC, irD = make_ir()
    wet = np.zeros((2, N))
    wet[0] = signal.fftconvolve(send[0], irA)[:N] + 0.35 * signal.fftconvolve(send[1], irC)[:N]
    wet[1] = signal.fftconvolve(send[1], irB)[:N] + 0.35 * signal.fftconvolve(send[0], irD)[:N]
    mix = dry + wet * 0.9
    mix = signal.sosfiltfilt(signal.butter(2, 30, 'high', fs=SR, output='sos'), mix, axis=1)
    mix = dialogue_dip(mix)
    lp = signal.sosfiltfilt(signal.butter(1, 160, 'low', fs=SR, output='sos'), mix, axis=1)
    mix = mix - 0.37 * lp          # gentle low shelf (-4 dB) to keep the low end from getting muddy
    mix /= np.max(np.abs(mix))
    mix, gr = compress(mix)
    mix *= 10 ** (1.0 / 20) / np.max(np.abs(mix))      # +1 dBFS pre-limiter -> limiter only kisses the burst
    mix, lg = limiter(mix, -1.2)
    t = np.arange(N) / SR
    fade = np.ones(N)
    fi = t < 0.05
    fade[fi] = t[fi] / 0.05
    fo = t > 58.4
    fade[fo] = np.cos(np.clip((t[fo] - 58.4) / 1.6, 0, 1) * np.pi / 2) ** 2
    fade[-48:] = 0
    mix *= fade
    return mix, gr, lg


def write_wav(path, x):
    import wave
    r = np.random.default_rng(1)
    d = (r.uniform(-0.5, 0.5, x.shape) + r.uniform(-0.5, 0.5, x.shape))   # TPDF dither (LSB units)
    nz = np.abs(x) > 0
    q = np.round(x * 32767 + d * nz)
    q = np.clip(q, -32768, 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(q.T.copy().tobytes())


def analyze(path=OUT_WAV):
    import wave
    with wave.open(path, 'rb') as w:
        assert w.getframerate() == SR and w.getnchannels() == 2 and w.getsampwidth() == 2
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), '<i2').astype(float).reshape(-1, 2).T / 32768
    print(f'frames={n}  duration={n / SR:.4f}s  nan={np.isnan(x).any()}')
    pk = np.max(np.abs(x))
    print(f'peak={20 * np.log10(pk):.2f} dBFS  full-scale samples={(np.abs(x) >= 32767 / 32768).sum()}')
    print(f'overall RMS={10 * np.log10(np.mean(x ** 2)):.2f} dBFS   L/R corr={np.corrcoef(x[0], x[1])[0, 1]:.3f}')
    sos = signal.butter(2, [300, 3000], 'band', fs=SR, output='sos')
    bp = signal.sosfilt(sos, x.mean(0))
    lo = signal.sosfilt(signal.butter(2, 300, 'low', fs=SR, output='sos'), x.mean(0))
    hi = signal.sosfilt(signal.butter(2, 3000, 'high', fs=SR, output='sos'), x.mean(0))
    print(' sec   RMS dBFS   peak dBFS   <300   300-3k  >3k  (band dB)   bar')
    for s in range(60):
        seg = x[:, s * SR:(s + 1) * SR]
        rms = 10 * np.log10(np.mean(seg ** 2) + 1e-20)
        p = 20 * np.log10(np.max(np.abs(seg)) + 1e-20)
        b = [10 * np.log10(np.mean(v[s * SR:(s + 1) * SR] ** 2) + 1e-20) for v in (lo, bp, hi)]
        print(f'{s:3d}  {rms:8.1f}  {p:9.1f}   {b[0]:6.1f} {b[1]:6.1f} {b[2]:6.1f}   ' + '#' * max(0, int((rms + 60) / 1.5)))
    return x


def main():
    os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
    print('composing / synthesizing ...', flush=True)
    B = compose()
    print('mixing / reverb / mastering ...', flush=True)
    mix, gr, lg = master(B)
    assert np.isfinite(mix).all()
    print(f'compressor max GR {gr.min():.1f} dB, limiter max GR {20 * np.log10(lg.min()):.1f} dB')
    write_wav(OUT_WAV, mix)
    with open(OUT_CUES, 'w') as f:
        f.write(CUES)
    print('wrote', OUT_WAV, OUT_CUES)
    if '--analyze' in sys.argv:
        analyze()


if __name__ == '__main__':
    main()
