#!/usr/bin/env python3
"""Register a scantron photo or scan: detect page boundary, warp to canonical rectangle."""

# Standard Library
import os
import argparse

# PIP3 modules
import cv2

# local repo modules
import omr_utils.template_loader
import omr_utils.image_registration


#============================================
def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments.

	Returns:
		parsed argument namespace
	"""
	parser = argparse.ArgumentParser(
		description="Register a scantron image: perspective warp to canonical rectangle"
	)
	parser.add_argument(
		'-i', '--input', dest='input_file', required=True,
		help="Path to raw scan or phone photo"
	)
	parser.add_argument(
		'-o', '--output', dest='output_file', required=True,
		help="Path for registered output image"
	)
	parser.add_argument(
		'-t', '--template', dest='template_file',
		default=None,
		help="Path to template YAML (default: config/dl1200_template.yaml)"
	)
	parser.add_argument(
		'-d', '--debug', dest='debug', action='store_true',
		help="Save debug overlay image alongside output"
	)
	args = parser.parse_args()
	return args


#============================================
def get_default_template_path() -> str:
	"""Return the default template path relative to this script.

	Returns:
		absolute path to config/dl1200_template.yaml
	"""
	# find repo root from this script location
	script_dir = os.path.dirname(os.path.abspath(__file__))
	template_path = os.path.join(script_dir, "config", "dl1200_template.yaml")
	return template_path


#============================================
def main() -> None:
	"""Main entry point for scan registration."""
	args = parse_args()
	# resolve template path
	template_path = args.template_file
	if template_path is None:
		template_path = get_default_template_path()
	# load template for canonical dimensions
	template = omr_utils.template_loader.load_template(template_path)
	canon_w = template["canonical"]["width_px"]
	canon_h = template["canonical"]["height_px"]
	# load input image
	print(f"loading: {args.input_file}")
	image = omr_utils.image_registration.load_image(args.input_file)
	print(f"  image size: {image.shape[1]}x{image.shape[0]}")
	# detect page contour
	corners = omr_utils.image_registration.find_page_contour(image)
	print("  page corners detected")
	# save debug overlay of the raw image with detected contour
	if args.debug:
		debug_raw = omr_utils.image_registration.draw_debug_overlay(image, corners)
		debug_raw_path = args.output_file.replace(".", "_contour_debug.")
		cv2.imwrite(debug_raw_path, debug_raw)
		print(f"  debug contour: {debug_raw_path}")
	# warp to natural rectangle
	nat_w, nat_h = omr_utils.image_registration.compute_output_dimensions(corners)
	warped = omr_utils.image_registration.warp_perspective(image, corners, nat_w, nat_h)
	# detect and fix orientation
	rotation = omr_utils.image_registration.detect_orientation(warped)
	if rotation != 0:
		print(f"  correcting orientation: {rotation} degrees")
		warped = omr_utils.image_registration.rotate_image_90(warped, rotation)
	# resize to canonical dimensions
	registered = cv2.resize(warped, (canon_w, canon_h), interpolation=cv2.INTER_AREA)
	# ensure output directory exists
	output_dir = os.path.dirname(args.output_file)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	# save registered image
	cv2.imwrite(args.output_file, registered)
	print(f"  registered: {args.output_file} ({canon_w}x{canon_h})")
	# save debug overlay on registered image
	if args.debug:
		debug_reg = omr_utils.image_registration.draw_registered_debug(registered, template)
		debug_path = args.output_file.replace(".", "_grid_debug.")
		cv2.imwrite(debug_path, debug_reg)
		print(f"  debug grid: {debug_path}")


#============================================
if __name__ == '__main__':
	main()
