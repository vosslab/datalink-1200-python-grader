#!/bin/sh

rm -f output/*;

source source_me.sh && ./run_pipeline.py -i scantrons/ \
  -k scantrons/804D5A50-key.jpg \
  -o output/ -d -r ncc --ncc-no-mask

sleep 0.1
open output/*_debug.png
