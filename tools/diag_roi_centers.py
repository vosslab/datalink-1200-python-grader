#!/usr/bin/env python3
"""Diagnostic: draw ROI centers and boundaries on a scan to check alignment."""

# PIP3 modules
import cv2

# local repo modules
import omr_utils.bubble_reader
import omr_utils.template_loader
import omr_utils.timing_mark_anchors


#============================================
def main() -> None:
	"""Draw choice centers and midpoint boundaries on one scan."""
	image = cv2.imread("scantrons/43F257A7-key.jpg")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	template = omr_utils.template_loader.load_template(
		"config/dl1200_template.yaml")
	raw_t = omr_utils.timing_mark_anchors.estimate_anchor_transform(
		gray, template)
	results = omr_utils.bubble_reader.read_answers(image, template)
	choices = template["answers"]["choices"]
	debug = image.copy()
	# draw for rows 10 and 60 (left and right column)
	for q_idx in [9, 59]:
		entry = results[q_idx]
		positions = entry.get("positions", {})
		q_num = entry["question"]
		# draw choice centers as green vertical lines
		for c in choices:
			if c not in positions:
				continue
			px, py = positions[c]
			cx = int(px)
			cy = int(py)
			# green vertical line at choice center
			cv2.line(debug, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)
			# small circle at center
			cv2.circle(debug, (cx, cy), 3, (0, 255, 0), -1)
			# label
			cv2.putText(debug, f"{c}", (cx - 3, cy - 35),
				cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
		# draw midpoint boundaries as red vertical lines
		avail = [(c, positions[c][0]) for c in choices if c in positions]
		for i in range(1, len(avail)):
			mid_x = int((avail[i - 1][1] + avail[i][1]) / 2.0)
			cy = int(positions[avail[i][0]][1])
			cv2.line(debug, (mid_x, cy - 30), (mid_x, cy + 30),
				(0, 0, 255), 1)
		# print positions
		print(f"Q{q_num}:")
		for c in choices:
			if c in positions:
				print(f"  {c}: x={positions[c][0]:.1f} y={positions[c][1]:.1f}")
	cv2.imwrite("output_smoke/diag_roi_centers.png", debug)
	print("\nSaved to output_smoke/diag_roi_centers.png")


#============================================
if __name__ == "__main__":
	main()
