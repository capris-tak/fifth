#!/usr/bin/env python3
"""Procedural SFX + ambience for 「ほしのとうだい」.

Outputs (48 kHz, stereo, 16-bit PCM, exactly 60.000 s):
  audio/sfx.wav        all one-shot / cue effects, ducked under dialogue
  audio/ambience.wav   continuous wind / chime / shimmer bed
  audio/sfx_cues.txt   cue list with times

Run:  /opt/tts/bin/python tools/sfx.py [--plot out.png]
Deterministic (fixed seeds).
"""
import os
import sys
import wave

import numpy as np
from scipy import signal

SR = 48000
DUR = 60.0
N = int(round(DUR * SR))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(ROOT, "audio")

# Dialogue windows (start, end) from src/timeline.js
DIALOGUE = [(1.0, 10.06), (13.4, 16.08), (17.2, 22.43), (22.9, 24.95), (25.4, 28.10),
            (28.8, 30.72), (39.0, 42.25), (43.0, 44.65), (45.3, 48.93), (51.0, 58.54)]

CUES = []


def cue(t0, t1, name):
    CUES.append((t0, t1, name))


# ----------------------------------------------------------------- helpers
def db(x):
    return 10 ** (x / 20.0)


def tarr(dur):
    return np.arange(int(round(dur * SR))) / SR


def fades(x, fin=0.005, fout=0.01):
    """Raised-cosine fade in/out on a mono or (n,2) signal (in place copy)."""
    x = np.array(x, dtype=np.float64)
    n = x.shape[0]
    a = min(int(fin * SR), n // 2)
    b = min(int(fout * SR), n // 2)
    if a > 0:
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
        x[:a] *= w if x.ndim == 1 else w[:, None]
    if b > 0:
        w = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, b))
        x[n - b:] *= w if x.ndim == 1 else w[:, None]
    return x


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, btype="lowpass", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, btype="highpass", fs=SR, output="sos"), x)


def sweep_bp_fast(x, fc, q=2.0, nb=24):
    """Vectorised approximation of a sweeping band-pass: crossfade a bank of
    fixed band-passes, weighting each by proximity (in octaves) to fc(t)."""
    fmin, fmax = max(40.0, fc.min() * 0.7), min(SR * 0.45, fc.max() * 1.4)
    centers = np.geomspace(fmin, fmax, nb)
    y = np.zeros_like(x)
    bw = 1.0 / q
    lfc = np.log2(np.maximum(fc, 1.0))
    wsum = np.zeros_like(x)
    for c in centers:
        lo, hi = c * 2 ** (-bw / 2), min(c * 2 ** (bw / 2), SR * 0.49)
        band = bp(x, lo, hi, 2)
        w = np.exp(-0.5 * ((lfc - np.log2(c)) / 0.35) ** 2)
        y += band * w
        wsum += w
    return y / np.maximum(wsum, 1e-3)


def panlaw(pan):
    """pan -1..1 (scalar or array) -> (gl, gr) equal power."""
    p = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return np.cos(p), np.sin(p)


def place(buf, t0, x, gain=1.0, pan=0.0, fin=0.005, fout=0.02):
    """Add mono (n,) or stereo (n,2) signal x to buf at time t0."""
    x = fades(x, fin, fout) * gain
    i0 = int(round(t0 * SR))
    if x.ndim == 1:
        gl, gr = panlaw(pan)
        x = np.stack([x * gl, x * gr], axis=1)
    n = x.shape[0]
    if i0 < 0:
        x = x[-i0:]
        i0 = 0
    n = min(x.shape[0], buf.shape[0] - i0)
    if n > 0:
        buf[i0:i0 + n] += x[:n]


def bell(freq, dur, rng, ratios=(1.0, 2.0, 2.76, 4.07, 5.43), decay=1.0, bright=1.0):
    """Small inharmonic bell / glass ting."""
    t = tarr(dur)
    y = np.zeros_like(t)
    for k, r in enumerate(ratios):
        f = freq * r * (1 + rng.uniform(-0.003, 0.003))
        if f > SR * 0.45:
            continue
        amp = (bright ** k) / (1 + k * 1.2)
        d = decay / (1 + 0.9 * k)
        y += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * np.exp(-t / d)
    att = np.minimum(1, t / 0.002)
    return y * att


def ting(freq, dur, decay):
    t = tarr(dur)
    y = np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(2 * np.pi * freq * 2.0 * t) * np.exp(-t / (decay * 0.4))
    y += 0.06 * np.sin(2 * np.pi * freq * 3.01 * t) * np.exp(-t / (decay * 0.2))
    return y * np.exp(-t / decay) * np.minimum(1, t / 0.003)


def grain(rng, fmin=2500, fmax=9000, dmin=0.03, dmax=0.18):
    f = rng.uniform(fmin, fmax)
    d = rng.uniform(dmin, dmax)
    t = tarr(d * 4)
    y = np.sin(2 * np.pi * f * t) * np.exp(-t / d) * np.minimum(1, t / 0.0015)
    y += 0.3 * np.sin(2 * np.pi * f * 2.01 * t) * np.exp(-t / (d * 0.5))
    return y


def sparkle_field(buf, rng, t0, t1, rate_fn, gain_fn, pan_fn, **gk):
    """Poisson grains with time-varying rate (per s), gain and pan."""
    t = t0
    while t < t1:
        r = max(rate_fn(t), 0.1)
        t += rng.exponential(1.0 / r)
        if t >= t1:
            break
        g = grain(rng, **gk)
        place(buf, t, g, gain_fn(t) * rng.uniform(0.4, 1.0), pan_fn(t) + rng.uniform(-0.35, 0.35),
              fin=0.0015, fout=0.01)


def make_ir(dur, rt60, rng, lpf=6000, pre=0.012):
    t = tarr(dur)
    env = np.exp(-6.9 * t / rt60)
    ir = np.zeros((len(t), 2))
    for c in range(2):
        n = rng.standard_normal(len(t)) * env
        ir[:, c] = lp(n, lpf)
    ip = int(pre * SR)
    ir = np.vstack([np.zeros((ip, 2)), ir])[: len(t)]
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True))
    return ir


def reverb(x, ir, wet):
    y = np.zeros_like(x)
    for c in range(2):
        y[:, c] = signal.fftconvolve(x[:, c], ir[:, c])[: len(x)]
    return x + wet * y


def env_pts(t, pts):
    """piecewise-linear envelope from [(time, value), ...]."""
    xs, ys = zip(*pts)
    return np.interp(t, xs, ys)


# ------------------------------------------------------------- SFX cues
def lamp_idle(buf, rng):
    """0.5–10: faint distant lamp mechanism hum + slow clockwork tick."""
    t0, t1 = 0.5, 10.0
    t = tarr(t1 - t0)
    hum = (np.sin(2 * np.pi * 58 * t) + 0.5 * np.sin(2 * np.pi * 116.3 * t)
           + 0.25 * np.sin(2 * np.pi * 174.2 * t)) * (1 + 0.25 * np.sin(2 * np.pi * 0.31 * t))
    hum = lp(hum, 400)
    e = env_pts(t, [(0, 0), (2.0, 1), (t1 - t0 - 2.0, 1), (t1 - t0, 0)])
    place(buf, t0, hum * e, db(-46), pan=0.25, fin=0.05, fout=0.05)
    # clockwork tick every ~0.85 s, alternating tick/tock
    k = 0
    tt = 1.2
    while tt < 9.6:
        n = rng.standard_normal(int(0.03 * SR)) * np.exp(-tarr(0.03) / 0.004)
        f = 1900 if k % 2 == 0 else 1500
        click = bp(n, f * 0.8, f * 1.25) + 0.5 * bell(f * 0.6, 0.03, rng, decay=0.01)
        click = lp(click, 3500)  # distant
        place(buf, tt, click, db(-44), pan=0.3, fin=0.001, fout=0.005)
        tt += 0.85 + rng.uniform(-0.02, 0.02)
        k += 1
    cue(0.5, 10.0, "lamp mechanism idle hum + distant tick (very faint)")


def shooting_star(buf, rng):
    t0, t1 = 10.8, 12.45
    d = t1 - t0
    t = tarr(d)
    u = t / d
    # doppler-ish: rises, peaks near pass-by (~0.8), slight drop at the end
    fc = 900 * 2 ** (2.6 * u ** 1.6) * (1 - 0.18 * np.clip((u - 0.85) / 0.15, 0, 1))
    noise = rng.standard_normal(len(t))
    wh = sweep_bp_fast(noise, fc, q=2.2)
    amp = (u ** 1.8) * (1 - np.clip((u - 0.96) / 0.04, 0, 1) * 0.3)
    wh *= amp
    # shimmering high partials gliding with doppler
    sh = np.zeros_like(t)
    for k in range(9):
        base = rng.uniform(2200, 4200)
        f = base * (fc / fc[0]) ** 0.35
        ph = 2 * np.pi * np.cumsum(f) / SR
        trem = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(9, 23) * t + rng.uniform(0, 6))
        sh += np.sin(ph) * trem / 9
    sh *= amp
    pan = -0.85 + 1.0 * u ** 1.2  # left -> slightly left of centre (crash at x=-140)
    gl, gr = panlaw(pan)
    sig = wh * 0.9 + sh * 0.55
    st = np.stack([sig * gl, sig * gr], axis=1)
    place(buf, t0, st, db(-11), fin=0.2, fout=0.03)
    sparkle_field(buf, rng, t0 + 0.3, t1, lambda x: 10 + 50 * ((x - t0) / d) ** 2,
                  lambda x: db(-30) * ((x - t0) / d), lambda x: -0.85 + (x - t0) / d)
    cue(10.8, 12.4, "shooting star whoosh L->R with doppler bend + sparkles")


def impact(buf, rng):
    T = 12.5
    pan = -0.15
    # weighty magical thump: pitch-dropping sine + soft low noise
    t = tarr(1.2)
    f = 45 + 95 * np.exp(-t / 0.07)
    th = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28)
    th += 0.4 * lp(rng.standard_normal(len(t)), 250) * np.exp(-t / 0.08)
    th = np.tanh(1.5 * th) / np.tanh(1.5)
    place(buf, T, th, db(-9), pan=pan, fin=0.002, fout=0.05)
    # soft "poof" of air
    n = bp(rng.standard_normal(int(0.8 * SR)), 300, 2500) * np.exp(-tarr(0.8) / 0.15)
    place(buf, T, n, db(-20), pan=pan, fin=0.004, fout=0.05)
    # glassy crystalline shatter cascade: dense tings decaying in density
    tt = T
    while tt < T + 1.4:
        dens = 90 * np.exp(-(tt - T) / 0.35) + 6
        tt += rng.exponential(1 / dens)
        fr = rng.uniform(1800, 6500) * (1 - 0.15 * (tt - T))
        b = bell(fr, 0.6, rng, ratios=(1.0, 2.32, 4.25, 6.63), decay=rng.uniform(0.08, 0.3))
        g = db(-17) * np.exp(-(tt - T) / 0.6) * rng.uniform(0.35, 1)
        place(buf, tt, b, g, pan=pan + rng.uniform(-0.6, 0.6), fin=0.001, fout=0.02)
    # falling debris sparkles 12.6–14.0 (pitch drifting down)
    sparkle_field(buf, rng, T + 0.1, 14.0, lambda x: 30 * np.exp(-(x - T) / 0.7) + 4,
                  lambda x: db(-27) * np.exp(-(x - T) / 0.9), lambda x: pan, fmin=2000, fmax=7000)
    cue(12.5, 14.0, "IMPACT: magical thump + crystalline shatter cascade + falling sparkles")


def door(buf, rng, T, pan, dur=0.38, gain=db(-20), latch_gain=db(-22)):
    # latch: metallic click
    n = rng.standard_normal(int(0.04 * SR)) * np.exp(-tarr(0.04) / 0.003)
    latch = hp(n, 1500) + 0.6 * bell(2300, 0.04, rng, ratios=(1, 2.7, 4.1), decay=0.015)
    place(buf, T, latch, latch_gain, pan=pan, fin=0.0005, fout=0.005)
    # creak: stick-slip pulse train through wood resonances
    t = tarr(dur)
    u = t / dur
    f0 = 55 + 70 * np.sin(np.pi * u) + 12 * np.sin(2 * np.pi * 7 * t)
    ph = np.cumsum(f0) / SR
    pulses = (np.diff(np.floor(ph), prepend=0) > 0).astype(float)
    pulses *= 1 + 0.4 * rng.standard_normal(len(t))
    src = signal.lfilter([1], [1, -0.6], pulses)
    cr = bp(src, 450, 650) * 1.0 + bp(src, 1100, 1450) * 0.8 + bp(src, 2200, 2800) * 0.5
    cr *= np.sin(np.pi * u) ** 0.6
    cr /= np.max(np.abs(cr)) + 1e-9
    place(buf, T + 0.04, cr, gain, pan=pan, fin=0.01, fout=0.03)


def grass_step(rng, weight=1.0):
    d = 0.09
    t = tarr(d)
    crunch = np.zeros_like(t)
    # granular crackle of grass blades
    for _ in range(int(30 * weight)):
        i = rng.integers(0, int(0.05 * SR))
        crunch[i] += rng.standard_normal()
    crunch = bp(crunch, 1800, 7000)
    swish = bp(rng.standard_normal(len(t)), 900, 4000) * np.exp(-t / 0.025)
    thud = np.sin(2 * np.pi * 95 * t) * np.exp(-t / 0.018) * 0.6
    y = crunch * 0.5 + swish * 0.6 + thud
    return y / (np.max(np.abs(y)) + 1e-9)


def metal_step(rng):
    d = 0.35
    t = tarr(d)
    n = rng.standard_normal(len(t)) * np.exp(-t / 0.006)
    y = 0.6 * hp(n, 800)
    for f, a, dd in ((410, 1.0, 0.09), (1130, 0.6, 0.06), (2350, 0.35, 0.04), (3720, 0.2, 0.03)):
        y += a * np.sin(2 * np.pi * f * rng.uniform(0.97, 1.03) * t) * np.exp(-t / dd)
    y += 0.5 * np.sin(2 * np.pi * 120 * t) * np.exp(-t / 0.02)
    return y / (np.max(np.abs(y)) + 1e-9)


def mina_out(buf, rng):
    door(buf, rng, 12.9, pan=0.35)
    times = np.linspace(13.0, 13.72, 5) + rng.uniform(-0.015, 0.015, 5)
    for k, tt in enumerate(times):
        place(buf, tt, grass_step(rng, 0.8), db(-25) * (0.8 + 0.2 * (k % 2)),
              pan=0.3 - 0.3 * k / 4 + (0.05 if k % 2 else -0.05), fin=0.001, fout=0.01)
    cue(12.9, 13.3, "lighthouse door latch + wooden creak")
    cue(13.0, 13.8, "Mina: 5 quick footsteps on grass")


def flicker(buf, rng):
    """17–28.5: Hoshi's weak dying-sparkler sputter, intermittent, very quiet."""
    t0, t1 = 17.0, 28.5
    pan = -0.15
    tt = t0 + 0.2
    while tt < t1 - 0.3:
        # burst of crackles
        blen = rng.uniform(0.15, 0.6)
        bt = tarr(blen)
        nb = len(bt)
        x = np.zeros(nb)
        k = rng.poisson(blen * 70)
        idx = rng.integers(0, nb, k)
        x[idx] = rng.standard_normal(k) * rng.uniform(0.3, 1.0, k)
        cr = hp(x, 2500) + 0.25 * bp(rng.standard_normal(nb), 4000, 9000)
        # faint electrical buzz under it
        cr += 0.12 * np.sign(np.sin(2 * np.pi * 120 * bt)) * lp(np.abs(rng.standard_normal(nb)), 30)
        cr *= np.sin(np.pi * bt / blen) ** 0.5
        cr /= np.max(np.abs(cr)) + 1e-9
        place(buf, tt, cr, db(-31) * rng.uniform(0.6, 1.0), pan=pan + rng.uniform(-0.1, 0.1),
              fin=0.01, fout=0.03)
        tt += blen + rng.uniform(0.4, 1.6)
    cue(17.0, 28.5, "Hoshi weak flicker: intermittent crackle/sputter (very quiet)")


def run_to_lamp(buf, rng):
    times = 30.8 + np.arange(6) * 0.14 + rng.uniform(-0.01, 0.01, 6)
    for k, tt in enumerate(times):
        place(buf, tt, grass_step(rng, 1.0), db(-24) * (1 - 0.25 * k / 5),
              pan=0.0 + 0.35 * k / 5, fin=0.001, fout=0.01)
    door(buf, rng, 31.62, pan=0.4, dur=0.22, gain=db(-26), latch_gain=db(-24))
    # door thump shut
    t = tarr(0.25)
    th = np.sin(2 * np.pi * 85 * t) * np.exp(-t / 0.04) + 0.3 * lp(rng.standard_normal(len(t)), 600) * np.exp(-t / 0.02)
    place(buf, 31.85, th, db(-22), pan=0.4, fin=0.001, fout=0.03)
    cue(30.8, 31.8, "Mina runs (6 steps on grass) + door")
    # metal stairs with echo, receding
    stairs = np.zeros((int(1.4 * SR), 2))
    for k in range(6):
        tt = 0.0 + k * 0.13
        place(stairs, tt, metal_step(rng), (1 - 0.13 * k) * db(-24), pan=0.4 + (0.06 if k % 2 else -0.06),
              fin=0.0005, fout=0.02)
    stairs[:, 0] = lp(stairs[:, 0], 5000)
    stairs[:, 1] = lp(stairs[:, 1], 5000)
    ir = make_ir(1.2, 0.9, rng, lpf=4000)
    stairs = reverb(stairs, ir, 0.9)
    place(buf, 31.95, stairs, 1.0, fin=0.001, fout=0.2)
    cue(31.9, 32.6, "footsteps up metal stairs (echoey, receding)")


def lamp_ignite(buf, rng):
    T = 32.0
    pan = 0.35
    # heavy clunk
    t = tarr(0.6)
    cl = np.sin(2 * np.pi * (55 + 40 * np.exp(-t / 0.03)) * t) * np.exp(-t / 0.12)
    for f, a, dd in ((310, 0.5, 0.08), (780, 0.35, 0.06), (1460, 0.2, 0.04)):
        cl += a * np.sin(2 * np.pi * f * t) * np.exp(-t / dd)
    cl += 0.5 * lp(rng.standard_normal(len(t)), 1200) * np.exp(-t / 0.015)
    place(buf, T, cl, db(-10), pan=pan, fin=0.001, fout=0.05)
    # gear ratchet 32.15–32.9, accelerating
    tt = 32.15
    k = 0
    while tt < 32.9:
        n = rng.standard_normal(int(0.03 * SR)) * np.exp(-tarr(0.03) / 0.003)
        c = bp(n, 1400, 4200) + 0.7 * bell(1250 + 90 * (k % 3), 0.03, rng, ratios=(1, 2.6, 4.3), decay=0.012)
        place(buf, tt, c, db(-23), pan=pan + (0.05 if k % 2 else -0.05), fin=0.0005, fout=0.005)
        tt += max(0.035, 0.11 - 0.008 * k)
        k += 1
    # rising warm "vwoom" hum, ignites ~32.7 -> full 33.5, sustains & fades by 37.4
    t0 = 32.55
    d = 37.4 - t0
    t = tarr(d)
    f0 = 38 + 34 * (1 - np.exp(-t / 0.35))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    hum = np.zeros_like(t)
    for h in range(1, 14):
        hum += np.sin(h * ph + 0.3 * h) / h ** 1.1
    # opening low-pass: bank crossfade (dark -> warm)
    dark = lp(hum, 180)
    warm = lp(hum, 1200)
    mix = np.clip(t / 0.9, 0, 1)
    hum = dark * (1 - mix) + warm * mix
    hum *= 1 + 0.08 * np.sin(2 * np.pi * 5.5 * t)
    e = env_pts(t, [(0, 0), (0.15, 0.25), (0.95, 1.0), (1.6, 0.55), (3.0, 0.35), (d, 0)])
    place(buf, t0, hum * e, db(-13), pan=pan * 0.6, fin=0.05, fout=0.1)
    cue(32.0, 32.1, "lamp: heavy mechanism clunk")
    cue(32.15, 32.9, "lamp: gear ratchet (accelerating)")
    cue(32.55, 33.5, "lamp: rising electric hum 'vwoom' ignition (sustains, fades by 37.4)")


def beam_sweep(buf, rng):
    t0, t1 = 33.4, 35.2
    d = t1 - t0
    t = tarr(d)
    u = t / d
    fc = 250 + 500 * np.sin(np.pi * u) ** 1.5
    noise = rng.standard_normal((len(t), 2))
    w = np.stack([sweep_bp_fast(noise[:, c], fc, q=1.4) for c in range(2)], axis=1)
    w += 0.5 * np.stack([lp(noise[:, c], 160) for c in range(2)], axis=1) * np.sin(np.pi * u)[:, None]
    e = np.sin(np.pi * u) ** 1.3
    pan = 0.6 - 0.8 * u  # from the lamp (right) down onto Hoshi (left)
    gl, gr = panlaw(pan)
    sig = np.stack([w[:, 0] * gl, w[:, 1] * gr], axis=1) * e[:, None]
    sig /= np.max(np.abs(sig))
    place(buf, t0, sig, db(-12), fin=0.1, fout=0.1)
    cue(33.5, 35.0, "beam sweep: deep airy swoosh R->L")


def absorb_riser(buf, rng):
    t0, t1 = 35.0, 37.6
    d = t1 - t0
    t = tarr(d)
    u = t / d
    # gliding sine partials (D major pentatonic) rising ~1.4 octaves
    notes = [293.66, 329.63, 369.99, 440.0, 493.88, 587.33, 659.25, 739.99, 880.0, 987.77, 1174.66, 1479.98]
    glide = 2 ** (1.4 * u ** 1.7)
    sh = np.zeros_like(t)
    st = np.zeros((len(t), 2))
    for k, f in enumerate(notes):
        fr = f * glide * (1 + 0.004 * np.sin(2 * np.pi * rng.uniform(4, 7) * t))
        ph = 2 * np.pi * np.cumsum(fr) / SR + rng.uniform(0, 6)
        trem = 0.65 + 0.35 * np.sin(2 * np.pi * (6 + 10 * u) * t + rng.uniform(0, 6))
        s = np.sin(ph) * trem / len(notes)
        gl, gr = panlaw(rng.uniform(-0.6, 0.6))
        st[:, 0] += s * gl
        st[:, 1] += s * gr
    # airy noise riser
    nz = rng.standard_normal((len(t), 2))
    for c in range(2):
        st[:, c] += 0.2 * bp(nz[:, c], 3000, 11000) * u ** 2
    e = (u ** 2.0) * 0.9 + 0.1 * u
    st *= e[:, None]
    st /= np.max(np.abs(st))
    place(buf, t0, st, db(-11), pan=-0.1, fin=0.3, fout=0.025)
    sparkle_field(buf, rng, t0, t1 - 0.02, lambda x: 6 + 110 * ((x - t0) / d) ** 2.2,
                  lambda x: db(-29) + db(-22) * ((x - t0) / d) ** 2, lambda x: -0.15)
    cue(35.0, 37.6, "absorption riser: gliding partials + densifying sparkles")


def burst(buf, rng):
    T = 37.6
    # bright bell cluster (D major)
    for f, g in ((587.33, 1.0), (739.99, 0.8), (880.0, 0.8), (1174.66, 0.7), (1760.0, 0.5),
                 (2349.3, 0.4), (2959.96, 0.3)):
        b = bell(f, 4.5, rng, ratios=(1.0, 2.0, 3.0, 4.2, 5.4), decay=1.6, bright=0.55)
        place(buf, T + rng.uniform(0, 0.03), b, db(-19) * g, pan=rng.uniform(-0.5, 0.5),
              fin=0.001, fout=0.3)
    # soft sub boom
    t = tarr(1.8)
    f = 34 + 40 * np.exp(-t / 0.12)
    sb = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.45) * np.minimum(1, t / 0.01)
    place(buf, T, sb, db(-8), pan=0.0, fin=0.005, fout=0.2)
    # airy whoosh outward (decorrelated, wide)
    t = tarr(2.0)
    fc = 3500 * np.exp(-t / 0.5) + 500
    w = np.stack([sweep_bp_fast(rng.standard_normal(len(t)), fc, q=1.2) for _ in range(2)], axis=1)
    w *= (np.minimum(1, t / 0.02) * np.exp(-t / 0.4))[:, None]
    w /= np.max(np.abs(w))
    place(buf, T, w, db(-14), fin=0.003, fout=0.2)
    # long glittering tail ~4 s
    sparkle_field(buf, rng, T, T + 4.2, lambda x: 120 * np.exp(-(x - T) / 0.9) + 5,
                  lambda x: db(-24) * np.exp(-(x - T) / 1.5), lambda x: 0.0, fmin=2500, fmax=10000)
    cue(37.6, 41.8, "BURST: bell cluster + sub boom + outward whoosh + 4 s glitter tail")


def happy_sparkle(buf, rng):
    notes = [1174.66, 1479.98, 1760.0, 2349.32, 2959.96, 3520.0]
    for k, f in enumerate(notes):
        place(buf, 38.05 + k * 0.09, ting(f, 0.6, 0.18), db(-24) * (1 - 0.06 * k),
              pan=-0.3 + 0.12 * k, fin=0.001, fout=0.05)
    sparkle_field(buf, rng, 38.0, 39.0, lambda x: 25, lambda x: db(-30), lambda x: -0.1)
    cue(38.0, 39.0, "happy twinkly sparkle arpeggio")


def ascend(buf, rng):
    t0, t1 = 48.6, 51.5
    d = t1 - t0
    t = tarr(d)
    u = t / d
    fc = 500 * 2 ** (3.6 * u)
    w = np.stack([sweep_bp_fast(rng.standard_normal(len(t)), fc, q=2.0) for _ in range(2)], axis=1)
    # gliding magical partials
    s = np.zeros_like(t)
    for f in (587.33, 880.0, 1174.66, 1760.0):
        fr = f * 2 ** (1.5 * u)
        s += np.sin(2 * np.pi * np.cumsum(fr) / SR) * (0.6 + 0.4 * np.sin(2 * np.pi * 11 * t))
    s /= 4
    e = env_pts(u, [(0, 0), (0.15, 0.35), (0.55, 1.0), (0.8, 0.6), (1.0, 0)])
    pan = -0.2 + 0.45 * u
    gl, gr = panlaw(pan)
    sig = w * 0.8 + np.stack([s * gl, s * gr], axis=1) * 0.5
    sig *= e[:, None]
    sig /= np.max(np.abs(sig))
    place(buf, t0, sig, db(-15), fin=0.15, fout=0.15)
    sparkle_field(buf, rng, t0 + 0.1, t1, lambda x: 35, lambda x: db(-28) * env_pts((x - t0) / d, [(0, 0.4), (0.5, 1), (1, 0.1)]),
                  lambda x: -0.2 + 0.45 * (x - t0) / d, fmin=3000, fmax=10000)
    cue(48.6, 51.5, "Hoshi ascends: rising whoosh + sparkle trail, panning up/away")


def settle_ting(buf, rng):
    place(buf, 51.5, ting(1760.0, 3.0, 0.9), db(-22), pan=0.15, fin=0.001, fout=0.3)
    cue(51.5, 51.5, "star settles: single pure crystal ting (A6)")


def constellation(buf, rng):
    penta = [587.33, 659.25, 739.99, 880.0, 987.77]
    freqs = [f * o for o in (2, 4) for f in penta] + [1174.66 * 0 + 587.33 * 2]
    times = np.sort(rng.uniform(52.5, 55.9, 12))
    times[0] = 52.5
    for tt in times:
        f = rng.choice(freqs)
        place(buf, tt, ting(f, 1.5, rng.uniform(0.3, 0.6)), db(-28) * rng.uniform(0.7, 1.0),
              pan=rng.uniform(-0.6, 0.6), fin=0.001, fout=0.1)
    cue(52.5, 56.0, "constellation stars pop in: 12 soft pentatonic tings (D major)")


def title_chime(buf, rng):
    T = 56.5
    d = 3.4  # to 59.9
    t = tarr(d)
    chord = [587.33, 739.99, 880.0, 1174.66, 1318.51, 1760.0, 2349.32]
    st = np.zeros((len(t), 2))
    for k, f in enumerate(chord):
        trem = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(3, 7) * t + rng.uniform(0, 6))
        s = (np.sin(2 * np.pi * f * t) + 0.2 * np.sin(2 * np.pi * f * 2.003 * t)) * trem / (1 + 0.15 * k)
        gl, gr = panlaw(-0.6 + 1.2 * k / (len(chord) - 1))
        st[:, 0] += s * gl
        st[:, 1] += s * gr
    e = env_pts(t, [(0, 0), (0.9, 1.0), (1.6, 0.8), (2.7, 0.3), (d, 0)])
    st *= e[:, None]
    st /= np.max(np.abs(st))
    place(buf, T, st, db(-24), fin=0.05, fout=0.2)
    sparkle_field(buf, rng, T + 0.2, T + 2.8, lambda x: 12, lambda x: db(-34), lambda x: 0.0)
    cue(56.5, 59.9, "title: soft shimmering chime swell (D major add9)")


def duck_env():
    t = np.arange(N) / SR
    g = np.ones(N)
    for a, b in DIALOGUE:
        g[(t >= a) & (t < b)] = db(-5)
    # smooth (~80 ms)
    k = int(0.08 * SR)
    win = np.hanning(2 * k + 1)
    win /= win.sum()
    return signal.fftconvolve(g, win, mode="same")


def build_sfx():
    rng = np.random.default_rng(20261001)
    dry = np.zeros((N, 2))
    for fn in (lamp_idle, shooting_star, impact, mina_out, flicker, run_to_lamp, lamp_ignite,
               beam_sweep, absorb_riser, burst, happy_sparkle, ascend, settle_ting, constellation,
               title_chime):
        fn(dry, rng)
    # gentle outdoor space reverb on the whole bus
    ir = make_ir(3.0, 2.4, np.random.default_rng(7), lpf=7000, pre=0.02)
    out = reverb(dry, ir, 0.22)
    out = hp(out.T, 25).T
    out *= duck_env()[:, None]
    # final fades: 0 at very start / end
    t = np.arange(N) / SR
    out *= np.clip((DUR - t) / 0.4, 0, 1)[:, None]
    return out


# ------------------------------------------------------------- ambience
def smooth_noise(rng, n, rate_hz):
    """Slow random modulation signal ~N(0,1) band-limited to rate_hz."""
    m = int(DUR * rate_hz * 4) + 4
    pts = rng.standard_normal(m)
    x = np.linspace(0, m - 1, n)
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(np.arange(m), pts)
    return cs(x)


def build_ambience():
    rng = np.random.default_rng(424242)
    t = np.arange(N) / SR
    gust = 0.7 + 0.18 * np.tanh(smooth_noise(rng, N, 0.08)) + 0.12 * np.tanh(smooth_noise(rng, N, 0.25))
    gust = np.clip(gust, 0.35, 1.1)
    out = np.zeros((N, 2))
    bands = [(80, 220, 1.0, 0.0), (180, 420, 0.8, 0.5), (350, 750, 0.55, 1.0),
             (650, 1300, 0.35, 1.6), (1200, 2600, 0.15, 2.4), (2500, 6000, 0.035, 3.0)]
    corr = 0.6
    common = rng.standard_normal(N)
    for lo, hi, a, tilt in bands:
        for c in range(2):
            src = corr * common + np.sqrt(1 - corr ** 2) * rng.standard_normal(N)
            b = bp(src, lo, hi, 2)
            # higher bands swell more with gusts -> moving spectral tilt = "whoosh"
            mod = gust ** (0.8 + 0.8 * tilt) * (1 + 0.15 * np.tanh(smooth_noise(rng, N, 0.4)))
            out[:, c] += a * b * mod
    out /= np.sqrt(np.mean(out ** 2))
    out *= db(-34)
    # faint whistling resonance, slowly wandering
    fc = 900 + 250 * np.tanh(smooth_noise(rng, N, 0.1))
    wh = sweep_bp_fast(rng.standard_normal(N), fc, q=14)
    wh = wh / np.sqrt(np.mean(wh ** 2)) * db(-52) * gust ** 2.5
    out[:, 0] += wh * 0.8
    out[:, 1] += wh * 0.6
    # airy starry shimmer: high soft partials with slow AM
    sh = np.zeros((N, 2))
    for k in range(10):
        f = rng.uniform(3500, 8000)
        am = np.clip(0.5 + 0.5 * np.tanh(1.5 * smooth_noise(rng, N, 0.3)), 0, 1) ** 2
        s = np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * am
        gl, gr = panlaw(rng.uniform(-0.8, 0.8))
        sh[:, 0] += s * gl
        sh[:, 1] += s * gr
    sh /= np.max(np.abs(sh))
    out += sh * db(-50)
    # occasional faint wind-chime tinkles
    chimes = [1567.98, 1760.0, 2093.0, 2349.32, 2637.0, 3135.96]
    tt = 3.0
    ch = np.zeros((N, 2))
    while tt < 56:
        cpan = rng.uniform(-0.7, 0.7)
        nstr = rng.integers(3, 7)
        st = tt
        for _ in range(nstr):
            b = bell(rng.choice(chimes), 2.5, rng, ratios=(1.0, 2.76, 5.4), decay=0.9, bright=0.5)
            place(ch, st, b, db(-40) * rng.uniform(0.5, 1.0), pan=cpan + rng.uniform(-0.15, 0.15),
                  fin=0.001, fout=0.2)
            st += rng.uniform(0.08, 0.35)
        cue(tt, st, "ambience: faint wind chime tinkle")
        tt += rng.uniform(6.0, 10.0)
    out += ch
    # level shape: fade-in 0–1.5, lower 17–29, fade out 58–60
    lvl = env_pts(t, [(0, 0), (1.5, 1), (16.5, 1), (17.5, db(-4)), (28.5, db(-4)), (29.5, 1),
                      (58.0, 1), (60.0, 0)])
    out *= lvl[:, None]
    return out


# ------------------------------------------------------------- output
def write_wav(path, x):
    assert x.shape == (N, 2), x.shape
    assert np.all(np.isfinite(x)), "non-finite samples"
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def dbfs(v):
    return 20 * np.log10(max(v, 1e-12))


def report(name, x):
    peak = np.max(np.abs(x))
    print(f"\n== {name}: dur={x.shape[0] / SR:.3f}s peak={dbfs(peak):.2f} dBFS "
          f"rms={dbfs(np.sqrt(np.mean(x ** 2))):.2f} dBFS nan={np.isnan(x).any()}")
    # click check: max sample-to-sample jump
    jump = np.max(np.abs(np.diff(x, axis=0)))
    print(f"   max sample jump = {jump:.4f} ({dbfs(jump):.1f} dBFS); first/last samples "
          f"{np.abs(x[0]).max():.2e}/{np.abs(x[-1]).max():.2e}")
    row = []
    for s in range(60):
        seg = x[s * SR:(s + 1) * SR]
        row.append(f"{s:2d}s {dbfs(np.sqrt(np.mean(seg ** 2))):6.1f}/{dbfs(np.max(np.abs(seg))):6.1f}")
    print("   RMS/peak dBFS per second:")
    for i in range(0, 60, 5):
        print("   " + " | ".join(row[i:i + 5]))


def plot(sfx, amb, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    for ax, x, nm in ((axs[0], sfx, "sfx"), (axs[1], amb, "ambience")):
        m = x.mean(axis=1)
        f, tt, S = signal.spectrogram(m, SR, nperseg=2048, noverlap=1024)
        ax.pcolormesh(tt, f, 10 * np.log10(S + 1e-14), vmin=-150, vmax=-50, shading="auto", cmap="magma")
        ax.set_yscale("symlog", linthresh=500)
        ax.set_ylim(30, 20000)
        ax.set_title(nm)
        for a, b in DIALOGUE:
            ax.axvspan(a, b, color="cyan", alpha=0.06)
    axs[1].set_xticks(range(0, 61, 2))
    fig.tight_layout()
    fig.savefig(path, dpi=80)


def main():
    os.makedirs(AUDIO, exist_ok=True)
    sfx = build_sfx()
    peak = np.max(np.abs(sfx))
    if peak > db(-3.2):
        sfx *= db(-3.2) / peak
        print(f"sfx limited by {dbfs(db(-3.2) / peak):.2f} dB")
    amb = build_ambience()
    if np.max(np.abs(amb)) > db(-6):
        amb *= db(-6) / np.max(np.abs(amb))
    for nm, x in (("sfx.wav", sfx), ("ambience.wav", amb)):
        write_wav(os.path.join(AUDIO, nm), x)
        report(nm, x)
    with open(os.path.join(AUDIO, "sfx_cues.txt"), "w") as f:
        f.write("# ほしのとうだい SFX cue list (seconds). sfx.wav unless marked 'ambience'.\n")
        f.write("# ambience.wav: whole film wind bed, -4 dB 17-29 s, fade-in 0-1.5, fade-out 58-60\n")
        f.write("# sfx bus ducked -5 dB under dialogue windows; light outdoor reverb on bus.\n")
        for a, b, nm in sorted(CUES):
            f.write(f"{a:7.3f} - {b:7.3f}  {nm}\n")
    if "--plot" in sys.argv:
        p = sys.argv[sys.argv.index("--plot") + 1]
        plot(sfx, amb, p)
        print("spectrogram ->", p)


if __name__ == "__main__":
    main()
