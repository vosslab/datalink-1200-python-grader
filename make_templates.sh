#!/usr/bin/env bash
# Build per-letter bubble templates from scantron images.
# Runs the two-pass pipeline: ROI extraction then template construction.
# Final templates are saved to config/bubble_templates/

set -e

source source_me.sh

rm -fr config/bubble_templates/* output_bubble_templates/*

python tools/build_bubble_templates.py \
	-i scantrons/ \
	-o output_bubble_templates/

sleep 0.1
open output_bubble_templates/qc_darkness_filter/*_norm_filtered_montage.png
sleep 0.1
open config/bubble_templates/?.png
