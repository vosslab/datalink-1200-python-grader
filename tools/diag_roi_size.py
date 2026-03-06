#!/usr/bin/env python3
"""Diagnostic: print ROI dimensions from SlotMap lattice geometry."""

# PIP3 modules
import cv2

# local repo modules
import omr_utils.bubble_reader
import omr_utils.bubble_template_extractor
import omr_utils.slot_map
import omr_utils.template_loader
import omr_utils.timing_mark_anchors


#============================================
def main() -> None:
	"""Print ROI dimensions for one scan."""
	image = cv2.imread("scantrons/43F257A7-key.jpg")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	template = omr_utils.template_loader.load_template(
		"config/dl1200_template.yaml")
	raw_t = omr_utils.timing_mark_anchors.estimate_anchor_transform(
		gray, template)
	sm = omr_utils.slot_map.SlotMap(raw_t, template)
	geom = sm.geom()
	results = omr_utils.bubble_reader.read_answers(image, template)
	choices = template["answers"]["choices"]
	# pick a few rows: first, middle, last
	test_rows = [results[0], results[24], results[49]]
	for entry in test_rows:
		q_num = entry["question"]
		positions = entry.get("positions", {})
		# get lattice ROI bounds for this row
		top_y, bot_y, _, _ = sm.roi_bounds(q_num, "A")
		cy = sm.row_center(q_num)
		hh_top = cy - top_y
		hh_bot = bot_y - cy
		print(f"Q{q_num}: hh_top={hh_top:.1f} hh_bot={hh_bot:.1f} "
			f"total_h={hh_top + hh_bot:.1f}")
		for c in choices:
			if c in positions:
				px, py = positions[c]
				roi = omr_utils.bubble_template_extractor.extract_roi_1x(
					gray, px, py, geom)
				roi_str = "None"
				if roi is not None:
					roi_str = f"{roi.shape[1]}x{roi.shape[0]}"
				# get lattice bounds for this choice
				_, _, lx, rx = sm.roi_bounds(q_num, c)
				cx = sm.choice_center(q_num, c)
				lhw = cx - lx
				rhw = rx - cx
				print(f"  {c}: lhw={lhw:.1f} rhw={rhw:.1f} ROI={roi_str}")


#============================================
if __name__ == "__main__":
	main()
