#!/bin/bash
# Parallel render of the full 60 s film in N chunks, then concat → out/video.mp4
cd "$(dirname "$0")/.." && source tools/env.sh
N=${N:-4}; TOTAL=1800; STEP=$(( (TOTAL+N-1)/N )); rm -f out/part_*.mp4
for i in $(seq 0 $((N-1))); do s=$((i*STEP)); e=$(( s+STEP<TOTAL ? s+STEP : TOTAL ));
  node tools/render.js index.html $s $e out/part_$i.mp4 > out/render_$i.log 2>&1 & done; wait
grep -h PAGEERROR out/render_*.log | head
ls out/part_*.mp4 | sort -V | sed "s/^out\//file '/;s/$/'/" > out/parts.txt
ffmpeg -y -loglevel error -f concat -safe 0 -i out/parts.txt -c copy out/video.mp4 && echo VIDEO_DONE
