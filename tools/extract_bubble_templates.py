#!/usr/bin/env python3
"""Extract pixel templates of printed bubble letters from a registered scan.

Produces averaged A-E letter templates at 5X oversize resolution and saves
them as grayscale PNG files. These templates are used for NCC-based bubble
localization refinement.
"""

# Standard Library
import os
import argparse

# PIP3 modules
import cv2

# local repo modules
import omr_utils.template_loader
import omr_utils.image_registration
import omr_utils.bubble_reader
import omr_utils.bubble_template_extractor


#============================================
def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Extract bubble letter templates from a scantron image")
	parser.add_argument(
		'-i', '--input', dest='input_file', required=True,
		help="Input scantron image (blank or lightly marked form)")
	parser.add_argument(
		'-o', '--output', dest='output_dir',
		default=os.path.join("config", "bubble_templates"),
		help="Output directory for template PNGs (default: config/bubble_templates)")
	parser.add_argument(
		'-t', '--template', dest='template_file',
		default=os.path.join("config", "dl1200_template.yaml"),
		help="YAML template file (default: config/dl1200_template.yaml)")
	parser.add_argument(
		'-r', '--registered', dest='pre_registered',
		action='store_true',
		help="Input image is already registered (skip registration)")
	parser.set_defaults(pre_registered=False)
	args = parser.parse_args()
	return args


#============================================
def main() -> None:
	"""Main entry point for template extraction."""
	args = parse_args()
	# load template
	template = omr_utils.template_loader.load_template(args.template_file)
	canon_w = int(template["canonical"]["width_px"])
	canon_h = int(template["canonical"]["height_px"])
	# load and register image
	print(f"Loading: {args.input_file}")
	image = omr_utils.image_registration.load_image(args.input_file)
	if args.pre_registered:
		registered = image
		# resize to canonical if needed
		if registered.shape[1] != canon_w or registered.shape[0] != canon_h:
			registered = cv2.resize(
				registered, (canon_w, canon_h),
				interpolation=cv2.INTER_AREA)
	else:
		print("Registering image...")
		registered = omr_utils.image_registration.register_image(
			image, canon_w, canon_h)
	# convert to grayscale and run bubble detection
	gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	print("Running bubble detection...")
	results = omr_utils.bubble_reader.read_answers(registered, template)
	# extract templates
	print("Extracting letter templates...")
	templates = omr_utils.bubble_template_extractor.extract_letter_templates(
		gray, template, results)
	if not templates:
		print("ERROR: no templates extracted (insufficient empty bubbles)")
		raise RuntimeError("template extraction failed")
	# report extraction results
	for letter, tmpl in sorted(templates.items()):
		th, tw = tmpl.shape
		print(f"  {letter}: {tw}x{th} pixels")
	# save templates
	saved = omr_utils.bubble_template_extractor.save_templates(
		templates, args.output_dir)
	print(f"Saved {len(saved)} templates to {args.output_dir}/")
	for path in saved:
		print(f"  {path}")


#============================================
if __name__ == '__main__':
	main()
