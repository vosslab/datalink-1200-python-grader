"""Read student ID from the bubble grid on a registered scantron image."""

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.slot_map


#============================================
def _compute_sid_bracket_edge_mean(gray: numpy.ndarray,
	top_y: int, bot_y: int, left_x: int, right_x: int,
	measure_cfg: dict) -> float:
	"""Compute mean brightness of bracket edge strips for a student ID bubble.

	Measures the dark printed bracket borders at the top and bottom of
	the bubble box, using left and right side strips excluding the
	center letter zone. Bounds are the primary geometry input.

	Args:
		gray: grayscale image (0=black, 255=white)
		top_y: top edge y position
		bot_y: bottom edge y position
		left_x: left edge x position
		right_x: right edge x position
		measure_cfg: measurement constants from SlotMap.measure_cfg()

	Returns:
		average brightness of bracket edge strips (0-255 scale),
		or -1.0 if out of bounds
	"""
	h, w = gray.shape
	# derive center x from bounds midpoint
	cx = (left_x + right_x) // 2
	ce = measure_cfg["center_exclusion"]
	bb_v = measure_cfg["bracket_bar_v"]
	bb_h = measure_cfg["bracket_bar_h"]
	# horizontal bounds: bracket inner face to center exclusion
	bi = measure_cfg["bracket_inner_half"]
	lx1 = max(0, cx - bi)
	lx2 = max(0, cx - ce)
	rx1 = min(w, cx + ce)
	rx2 = min(w, cx + bi)
	# top bracket bar strip (positioned at bracket_bar_v from top edge)
	top_y1 = max(0, top_y + bb_v)
	top_y2 = max(0, top_y + bb_v + bb_h)
	# bottom bracket bar strip (mirrored from bottom edge)
	bot_y1 = min(h, bot_y - bb_v - bb_h)
	bot_y2 = min(h, bot_y - bb_v)
	# extract pixel strips
	tl = gray[int(top_y1):int(top_y2), int(lx1):int(lx2)]
	tr = gray[int(top_y1):int(top_y2), int(rx1):int(rx2)]
	bl = gray[int(bot_y1):int(bot_y2), int(lx1):int(lx2)]
	br = gray[int(bot_y1):int(bot_y2), int(rx1):int(rx2)]
	# combine all bracket edge pixels
	all_pixels = numpy.concatenate([
		tl.ravel(), tr.ravel(), bl.ravel(), br.ravel()
	])
	if all_pixels.size == 0:
		return -1.0
	result = float(numpy.mean(all_pixels))
	return result


#============================================
def _compute_sid_measurement_mean(gray: numpy.ndarray,
	top_y: int, bot_y: int, left_x: int, right_x: int,
	measure_cfg: dict) -> float:
	"""Compute mean brightness of left/right measurement zones.

	Measures the fill level inside the bubble by sampling left and
	right strips (excluding center letter zone and insets).
	Bounds are the primary geometry input.

	Args:
		gray: grayscale image (0=black, 255=white)
		top_y: top edge y position
		bot_y: bottom edge y position
		left_x: left edge x position
		right_x: right edge x position
		measure_cfg: measurement constants from SlotMap.measure_cfg()

	Returns:
		average brightness of measurement zones (0-255),
		or -1.0 if out of bounds
	"""
	h, w = gray.shape
	# derive center x from bounds midpoint
	cx = (left_x + right_x) // 2
	ce = measure_cfg["center_exclusion"]
	fi_v = measure_cfg["fill_inset_v"]
	bi = measure_cfg["bracket_inner_half"]
	# left fill zone: bracket inner face to center exclusion
	lx1 = max(0, cx - bi)
	lx2 = max(0, cx - ce)
	# right fill zone: center exclusion to bracket inner face
	rx1 = min(w, cx + ce)
	rx2 = min(w, cx + bi)
	# vertical bounds: large inset below/above bracket bars
	y1 = max(0, top_y + fi_v)
	y2 = min(h, bot_y - fi_v)
	# extract pixel strips
	left_strip = gray[int(y1):int(y2), int(lx1):int(lx2)]
	right_strip = gray[int(y1):int(y2), int(rx1):int(rx2)]
	if left_strip.size == 0 or right_strip.size == 0:
		return -1.0
	left_mean = float(numpy.mean(left_strip))
	right_mean = float(numpy.mean(right_strip))
	# average both zones
	result = (left_mean + right_mean) / 2.0
	return result


#============================================
def _score_sid_bubble(gray: numpy.ndarray,
	top_y: int, bot_y: int, left_x: int, right_x: int,
	measure_cfg: dict) -> float:
	"""Score a single student ID bubble using bracket-edge dark reference.

	Takes explicit ROI bounds from SlotMap.sid_roi_bounds(). No center-
	plus-box reconstruction. Bounds are the primary geometry input.

	Scoring logic ported from score_bubble_fast(): uses the dark printed
	bracket edges as a reference. Unfilled bubbles have bright measurement
	zones and dark bracket edges (high contrast). Filled bubbles have
	pencil marks covering both zones (low contrast, high score).

	Args:
		gray: grayscale image (0=black, 255=white)
		top_y: top edge y position from sid_roi_bounds()
		bot_y: bottom edge y position from sid_roi_bounds()
		left_x: left edge x position from sid_roi_bounds()
		right_x: right edge x position from sid_roi_bounds()
		measure_cfg: measurement constants from SlotMap.measure_cfg()

	Returns:
		fill score (higher = more likely filled, range ~0.0 to 1.0).
		Returns -1.0 for invalid geometry.
	"""
	# bracket edges provide a dark reference (printed border)
	bracket_mean = _compute_sid_bracket_edge_mean(
		gray, top_y, bot_y, left_x, right_x, measure_cfg)
	if bracket_mean < 0:
		return -1.0
	# measurement zone fill level
	measurement_mean = _compute_sid_measurement_mean(
		gray, top_y, bot_y, left_x, right_x, measure_cfg)
	if measurement_mean < 0:
		return -1.0
	# if both zones are bright, no bracket edges found -- misalignment
	if bracket_mean > 200.0 and measurement_mean > 200.0:
		score = 0.0
		return score
	# when bracket edges are very dark (near zero), use 255 as reference
	if bracket_mean <= 1.0:
		score = (255.0 - measurement_mean) / 255.0
		return score
	# pencil covers bracket edges too: measurement <= bracket = filled
	if measurement_mean <= bracket_mean:
		score = 1.0
		return score
	# unfilled: measurement is brighter than bracket edges
	score = 1.0 - (measurement_mean - bracket_mean) / (255.0 - bracket_mean)
	# clamp to valid range
	score = max(0.0, score)
	return score


#============================================
def read_student_id(image, template: dict,
	slot_map: "omr_utils.slot_map.SlotMap",
	threshold: float = 0.05) -> str:
	"""Read the student ID from the bubble grid.

	Each of 9 digit positions has 10 bubbles (0-9).
	The digit with the highest score above threshold is selected.
	If no bubble is filled for a digit, '0' is used as default.

	Args:
		image: BGR registered image (numpy array)
		template: loaded template dictionary
		slot_map: SlotMap instance (single geometry authority)
		threshold: minimum score to consider a bubble filled

	Returns:
		student ID as a string of 9 digits
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	sid_config = template["student_id"]
	num_digits = sid_config["num_digits"]
	measure_cfg = slot_map.measure_cfg()
	digits = []
	for d in range(num_digits):
		best_value = 0
		best_score = -1.0
		for v in range(10):
			# get ROI bounds from SlotMap lattice
			top_y, bot_y, left_x, right_x = slot_map.sid_roi_bounds(d, v)
			score = _score_sid_bubble(
				gray, top_y, bot_y, left_x, right_x, measure_cfg)
			if score > best_score:
				best_score = score
				best_value = v
		# only use the digit if it stands out enough
		if best_score >= threshold:
			digits.append(str(best_value))
		else:
			digits.append("0")
	student_id = "".join(digits)
	return student_id


#============================================
def read_student_id_detailed(image, template: dict,
	slot_map: "omr_utils.slot_map.SlotMap",
	threshold: float = 0.05) -> dict:
	"""Read student ID with detailed per-digit scoring information.

	Args:
		image: BGR registered image (numpy array)
		template: loaded template dictionary
		slot_map: SlotMap instance (single geometry authority)
		threshold: minimum score to consider a bubble filled

	Returns:
		dict with keys: student_id (str), digits (list of detail dicts)
		each detail dict has: position, selected_value, scores (dict of value->score)
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	sid_config = template["student_id"]
	num_digits = sid_config["num_digits"]
	measure_cfg = slot_map.measure_cfg()
	digit_details = []
	id_string = ""
	for d in range(num_digits):
		scores = {}
		for v in range(10):
			# get ROI bounds from SlotMap lattice
			top_y, bot_y, left_x, right_x = slot_map.sid_roi_bounds(d, v)
			score = _score_sid_bubble(
				gray, top_y, bot_y, left_x, right_x, measure_cfg)
			scores[v] = score
		# find best
		best_value = max(scores, key=scores.get)
		best_score = scores[best_value]
		selected = best_value if best_score >= threshold else 0
		id_string += str(selected)
		detail = {
			"position": d,
			"selected_value": selected,
			"scores": scores,
		}
		digit_details.append(detail)
	result = {
		"student_id": id_string,
		"digits": digit_details,
	}
	return result
