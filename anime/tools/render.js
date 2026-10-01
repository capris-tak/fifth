// Usage: node tools/render.js <page.html> <startFrame> <endFrame> <out.mp4>   (30 fps, 1920x1080)
const { chromium } = require('playwright');
const path = require('path'); const { spawn } = require('child_process');
(async () => {
  const [page_, s, e, out] = process.argv.slice(2);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', '30', '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const b = await chromium.launch({ args: ['--allow-file-access-from-files'] });
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  p.on('pageerror', e => console.error('PAGEERROR', e.message));
  await p.goto('file://' + path.resolve(page_));
  await p.waitForFunction(() => window.READY === true);
  const cv = p.locator('canvas').first();
  for (let f = +s; f < +e; f++) {
    await p.evaluate(t => window.renderFrame(t), f / 30);
    const buf = await cv.screenshot({ type: 'png' });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (f % 60 === 0) console.log(out, 'frame', f);
  }
  ff.stdin.end(); await new Promise(r => ff.on('close', r)); await b.close();
})();
