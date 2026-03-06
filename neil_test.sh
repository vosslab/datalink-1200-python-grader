#!/bin/sh

rm -f output/*; source source_me.sh && ./run_pipeline.py -i scantrons/ -k scantrons/43F257A7-A03D-4CB2-8D7B-3EE057B41FAC_result.jpg -o output/ -d; open output/*_debug.png
