#!/bin/sh

rm -f output/*;

source source_me.sh && ./run_pipeline.py -i scantrons/ \
  -k scantrons/43F257A7-key.jpg \
  -o output/ -d -r ncc

sleep 0.1
open output/*_debug.png
