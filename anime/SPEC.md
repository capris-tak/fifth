# 「ほしのとうだい」(The Star Lighthouse) — production bible

A 60-second, 1920x1080, 30 fps 2D animated short, rendered **entirely from code** (HTML5 Canvas 2D,
deterministic, frame = pure function of time `t` in seconds). Voices are VOICEVOX (already rendered,
see `src/timeline.js`). Music and SFX are synthesized in Python/numpy.

## Story
A tiny floating island in a sea of clouds at night. A lone lighthouse-keeper girl, **ミナ (Mina)**, lives there.
A shooting star crashes beside the lighthouse: it is **ホシ (Hoshi)**, a small star creature whose light is fading —
it cannot return to the sky. Mina turns the great lighthouse lamp on it; Hoshi absorbs the warm light, blazes back
to life, thanks her, and flies home. From that night on, one especially bright star shines above the lighthouse,
and the surrounding stars connect into a constellation shaped like a lighthouse.

## Master timeline (seconds) — everyone syncs to this
| t | event |
|---|---|
| 0.0–1.5 | fade in from black; music starts (soft celesta/music-box) |
| 0–10.5 | **S1 Opening**: camera starts high in the starry sky + aurora, slow crane down to the island floating on the cloud sea; lighthouse lamp turning slowly (gentle). Narration n1 1.0–10.06 |
| 10.5–17.0 | **S2 Fall**: shooting star streaks across sky 10.8→12.4, **impact at 12.5** near the lighthouse (flash + sparkle splash + dust of light). Lighthouse door opens 12.9, Mina runs out 13.0–13.8. Mina m1 13.4–16.08 「わっ！流れ星が、落ちてきた！？」 |
| 17.0–28.6 | **S3 Meeting**: closer shot. Hoshi lies on the grass, dim & flickering, sits up weakly. h1 17.2–22.43, m2 22.9–24.95, h2 25.4–28.1. Sad/tender. |
| 28.6–39.0 | **S4 The Lamp**: m3 28.8–30.72 「それなら、とっておきがあるよ！」 (determined). Mina runs into the lighthouse 30.8–31.8; lamp mechanism clunks & rotates 32.0–33.5; beam sweeps down onto Hoshi 33.5–35.0; Hoshi absorbs light 35.0–37.6 (building shimmer); **BURST at 37.6** (big radiant flash, rays, particle shower); Hoshi bright & joyful 37.6–39 |
| 39.0–50.5 | **S5 Farewell**: h3 39.0–42.25 (joyful), m4 43.0–44.65 「また、会える？」 (wistful), h4 45.3–48.93 (warm promise). Hoshi rises into the sky 48.6–51.5 with a sparkling trail; Mina waves. |
| 50.5–60.0 | **S6 Constellation**: wide shot, Hoshi settles as the brightest star above the lighthouse at ~51.5; constellation lines draw 52.5–56 forming a lighthouse shape; n2 51.0–58.54; title 「ほしのとうだい」 fades in 56.5; credits 57.5; fade to black 59.2–60.0 |

## Art direction
Painterly-flat 2D, in the spirit of Ghibli night scenes × Makoto Shinkai skies: rich gradients, soft glows,
crisp silhouettes, gentle secondary motion everywhere (nothing is ever perfectly still). Use `globalCompositeOperation='lighter'`
and radial gradients for glows; `ctx.filter='blur(..)'` is allowed but sparingly (cost).

Palette (hex):
- Night sky: `#070a24` (zenith) → `#141a4a` → `#2b2366` → `#5a3b7a` (horizon violet), horizon haze `#8a5a8c`
- Aurora: `#5ef2c9`, `#3fb8e0`, `#b48cff`
- Cloud sea: tops `#cfd6f5` moonlit, mid `#7f86c4`, shadows `#3a3f7d`, deep `#1d2156`
- Island: grass `#2f6b5a` / lit `#4f9a74`, rock `#3b3560` / `#2a2548`
- Lighthouse: white walls `#e9e4f2` (night-shaded `#b9b3d6`), red bands `#c8434f`, roof `#7a2e3f`, metal `#2e2a44`
- Lamp / warm light: `#fff3c4`, `#ffd27a`, `#ffb347`, `#ff8a3d`
- Mina: dark-chocolate bob hair `#3a2430` (highlight `#6b4256`), skin `#ffe0cf` (shadow `#f2b8a8`), cheeks `#ff9a9a`,
  navy pea-coat `#2b3a6b` (shade `#1f2a52`), long **red scarf** `#e2474f` that flutters, brown boots `#5a3a2e`,
  eyes large, dark `#2a1830` with 2 white catch-lights.
- Hoshi: rounded 5-point star. Bright: core `#fffbe6`, body `#ffe27a`, rim `#ffb84d`, glow `#fff2a8`.
  Dim: body `#9aa3c7`, rim `#6c7399`, almost no glow. Big glossy eyes, tiny mouth, small blush.

## World coordinates
- World units = pixels at zoom 1. y grows downward. Ground level of island ≈ y = 0.
- Island top surface spans x ≈ −700 … +700; `World.groundY(x)` returns exact ground height (gentle bumps, within ±25).
  Under the surface: rocky underside tapering to a point around y ≈ +520, with hanging roots/vines, tiny drifting rocks.
- Lighthouse: base centre at x = +260, y = groundY(260). Tower height ≈ 760 (tapering), gallery+lamp room above:
  lamp centre at about (260, −860). Door (arched, wooden, warm light inside when open) at its base, facing camera.
- Hoshi crash site: x = −140 on the ground. Mina's main marks: x ≈ −20 … +60.
- Cloud sea: top surface around y ≈ +250 … +350 (island floats above it, underside dips into it).
- Sky: drawn screen-space with parallax; moon upper-left.

## Module API contracts (globals; plain `<script>` files, no imports)
All draw functions take `ctx` already transformed into **world space** by the caller (via `U.camApply`) unless stated.
Never call `ctx.setTransform` with absolute values inside a draw function — use `save()/restore()`.
Randomness only via `U.rng(seed)` / `U.hash` / `U.noise`. No state between frames.

### `src/world.js` → `window.World`
- `World.groundY(x)`
- `World.sky(ctx, cam, t, o)` — draws in **screen space** (it sets its own transform from `cam` with parallax).
  `o = {aurora:0..1, starBoost:0..1, meteorHole?}`: gradient, ~800 twinkling stars (multiple sizes, some coloured), moon with halo,
  aurora curtains (animated), faint milky way band.
- `World.cloudsBack(ctx, cam, t)` / `World.cloudsFront(ctx, cam, t)` — layered cloud sea (screen-space parallax layers, slow drift),
  moonlit tops; front layer passes in front of the island underside.
- `World.island(ctx, t, o)` — world space: underside rock, roots, grass top with wind-swaying grass blades, a few small flowers
  (`o.bloom 0..1` makes flowers glow/open — used after the burst), small fence, stone path to the door.
- `World.lighthouse(ctx, t, o)` — world space. `o = {doorOpen:0..1, lampOn:0..1, lampAngle:rad, windowsLit:0..1}`.
  Lamp lens glows; draw the rotating light visually inside the lamp room.
- `World.beam(ctx, t, o)` — world space volumetric light beam from the lamp: `o = {angle:rad (0 = pointing +x, π/2 = straight down), length, width, intensity:0..1}`.
  Additive, soft edges, dust motes drifting inside.
- `World.lightPool(ctx, x, y, r, intensity, color)` — soft additive ground light.

### `src/mina.js` → `window.Mina`
`Mina.draw(ctx, t, o)` — world space, (o.x, o.y) = point between feet on the ground. Height ≈ 300 px at o.scale = 1.
`o = {x, y, scale=1, facing: 1|-1, pose, poseT (0..1 progress within pose if useful), expr, mouth, lookX, lookY, blink?: auto, rim?: {x,y,color,strength}, wind=1}`
- poses: `'stand'`, `'run'` (cycle driven by t), `'surprised'` (arms up, small jump), `'crouch'` (kneeling to look at something low),
  `'reach'` (one hand extended forward), `'point'` (point upward/forward with determination), `'wave'` (big wave, cycle), `'clasp'` (hands clasped at chest, hopeful), `'lookup'` (head tilted up).
- expr: `'neutral'`, `'happy'`, `'surprised'`, `'worried'`, `'determined'`, `'wistful'`, `'joy'`.
- `mouth`: viseme from `TL.mouth('mina', t)` (`a i u e o n c x`) — distinct mouth shapes per viseme.
- Auto blink (deterministic from t), breathing idle bob, hair & scarf secondary motion (scarf flutters in wind).
- `lookX/lookY` −1..1 eye direction. `rim`: warm rim light from a world point (for the lamp/beam/Hoshi glow).
- Transitions between poses: caller may pass `o.pose2` and `o.mix` (0..1) to blend; at minimum support it for arm angles.

### `src/hoshi.js` → `window.Hoshi`
`Hoshi.draw(ctx, t, o)` — world space, (o.x, o.y) = centre of the star body. Diameter ≈ 110 px at scale 1.
`o = {x, y, scale=1, rot=0, bright:0..1 (0 = dim grey-blue, 1 = brilliant gold), flicker:0..1 (amount of sputtering when dim),
      expr: 'hurt'|'sad'|'hope'|'surprised'|'joy'|'gentle', mouth (viseme from TL.mouth('hoshi',t)), lookX, lookY, squash:0..1, glow=1}`
- Soft, rounded 5-point star, slight jelly wobble, tiny stubby arms (two lower points act as feet, two side points as arms —
  side points can wave). Halo glow & radiating soft rays scale with `bright`. When dim, occasional weak sparks flicker.

### `src/fx.js` → `window.FX` (world space unless noted)
- `FX.shootingStar(ctx, t, {x0,y0,x1,y1,t0,t1})` — streak with glowing head and long fading tail + shed sparks.
- `FX.impact(ctx, t, {x,y,t0})` — flash, expanding ring, sparkle splash of light particles that arc and fall, lingering motes.
- `FX.absorb(ctx, t, {x,y,t0,t1})` — light particles spiralling INTO a point (charging up).
- `FX.burst(ctx, t, {x,y,t0})` — huge radiant burst: god-rays, shockwave ring, hundreds of sparkles, slow afterglow.
- `FX.trail(ctx, t, {path:(tt)=>[x,y], t0, t1})` — sparkling trail behind a moving object.
- `FX.motes(ctx, t, {x,y,w,h,count,seed,color})` — ambient floating fireflies/light motes.
- `FX.constellation(ctx, t, {points:[[x,y]...], lines:[[i,j]...], t0, t1, cam})` — screen space: stars pop in, lines draw progressively, glow.
- Screen-space post (operate on `ctx.canvas`, call with identity transform):
  `FX.bloom(ctx, strength)`, `FX.vignette(ctx, amount)`, `FX.grain(ctx, t, amount)`, `FX.fade(ctx, alpha)`,
  `FX.title(ctx, t, {t0})` (「ほしのとうだい」 elegant title with sparkle), `FX.credits(ctx, t, {t0})`
  (small text: `VOICEVOX:冥鳴ひまり　VOICEVOX:雨晴はう　VOICEVOX:No.7` and `Animation, music & sound: generated with code`).
  Fonts available: `IPAGothic`, `IPAPGothic` (also serif fallbacks). Draw text nicely (letter-spacing, glow).

### `src/scenes.js` → `window.Scenes.render(ctx, t)` — the director (camera, staging, cuts). Written last.

## Tools
- `source tools/env.sh` (sets NODE_PATH for playwright).
- Stills: `node tools/still.js <page.html> <outdir> 1.0 2.5 ...` → PNGs (look at them with the Read tool!). Page must define `window.renderFrame(t)` and set `window.READY=true`.
- For testing a module in isolation, create your own `tests/<name>.html` that loads `src/util.js`, `src/timeline.js`
  and your module (and others if present), and defines `renderFrame`.
- Performance budget: a full frame (everything) must render in < 120 ms. Cache static artwork in offscreen canvases (`U.buf(name,w,h)`) where possible —
  but caches must be deterministic (built from seeds, never from t).
