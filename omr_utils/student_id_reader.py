"""Read student ID from the bubble grid on a registered scantron image."""

# PIP3 modules
import cv2

# local repo modules
import omr_utils.template_loader
import omr_utils.bubble_reader


#============================================
def read_student_id(image, template: dict, threshold: float = 0.05) -> str:
	"""Read the student ID from the bubble grid.

	Each of 9 digit positions has 10 bubbles (0-9).
	The digit with the highest score above threshold is selected.
	If no bubble is filled for a digit, '0' is used as default.

	Args:
		image: BGR registered image (numpy array)
		template: loaded template dictionary
		threshold: minimum score to consider a bubble filled

	Returns:
		student ID as a string of 9 digits
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	h, w = image.shape[:2]
	sid_config = template["student_id"]
	num_digits = sid_config["num_digits"]
	radius = omr_utils.template_loader.get_bubble_radius_px(template, w, h)
	# get bubble geometry scaled to this image size
	geom = omr_utils.template_loader.get_bubble_geometry_px(template, w, h)
	digits = []
	for d in range(num_digits):
		best_value = 0
		best_score = -1.0
		for v in range(10):
			norm_x, norm_y = omr_utils.template_loader.get_student_id_coords(
				template, d, v)
			px, py = omr_utils.template_loader.to_pixels(norm_x, norm_y, w, h)
			score = omr_utils.bubble_reader.score_bubble_fast(
				gray, px, py, radius, geom)
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
	threshold: float = 0.05) -> dict:
	"""Read student ID with detailed per-digit scoring information.

	Args:
		image: BGR registered image (numpy array)
		template: loaded template dictionary
		threshold: minimum score to consider a bubble filled

	Returns:
		dict with keys: student_id (str), digits (list of detail dicts)
		each detail dict has: position, selected_value, scores (dict of value->score)
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	h, w = image.shape[:2]
	sid_config = template["student_id"]
	num_digits = sid_config["num_digits"]
	radius = omr_utils.template_loader.get_bubble_radius_px(template, w, h)
	# get bubble geometry scaled to this image size
	geom = omr_utils.template_loader.get_bubble_geometry_px(template, w, h)
	digit_details = []
	id_string = ""
	for d in range(num_digits):
		scores = {}
		for v in range(10):
			norm_x, norm_y = omr_utils.template_loader.get_student_id_coords(
				template, d, v)
			px, py = omr_utils.template_loader.to_pixels(norm_x, norm_y, w, h)
			score = omr_utils.bubble_reader.score_bubble_fast(
				gray, px, py, radius, geom)
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
