// Usage: node tools/still.js <page.html> <outdir> t1 t2 ...   -> <outdir>/f_<t>.png
// The page must define window.renderFrame(t). Prints any page errors.
const { chromium } = require('playwright');
const path = require('path'), fs = require('fs');
(async () => {
  const [page_, outdir, ...ts] = process.argv.slice(2);
  fs.mkdirSync(outdir, { recursive: true });
  const b = await chromium.launch({ args: ['--allow-file-access-from-files'] });
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  p.on('pageerror', e => console.error('PAGEERROR', e.message));
  p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.error('CONSOLE', m.text()); });
  await p.goto('file://' + path.resolve(page_));
  await p.waitForFunction(() => window.READY === true);
  for (const t of ts) {
    const t0 = Date.now();
    await p.evaluate(t => window.renderFrame(t), parseFloat(t));
    const f = path.join(outdir, `f_${(+t).toFixed(2)}.png`);
    await p.locator('canvas').first().screenshot({ path: f });
    console.log(f, (Date.now() - t0) + 'ms');
  }
  await b.close();
})();
