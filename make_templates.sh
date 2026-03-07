#!/usr/bin/env bash
# Build per-letter bubble templates from scantron images.
# Runs the two-pass pipeline: ROI extraction then template construction.
# Final templates are saved to config/bubble_templates/

set -e

source source_me.sh

python tools/build_bubble_templates.py \
	-i scantrons/ \
	-o output_bubble_templates/

open config/bubble_templates/?.png
