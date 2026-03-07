"""Detect scantron boundary in a photo, correct perspective, and warp to rectangle."""

# Standard Library
import math

# PIP3 modules
import cv2
import numpy


#============================================
def load_image(image_path: str) -> numpy.ndarray:
	"""Load an image from disk.

	Args:
		image_path: path to image file (JPEG, PNG, etc.)

	Returns:
		BGR image as numpy array

	Raises:
		FileNotFoundError: if image cannot be loaded
	"""
	img = cv2.imread(image_path)
	if img is None:
		raise FileNotFoundError(f"cannot load image: {image_path}")
	return img


#============================================
def preprocess_for_contours(image: numpy.ndarray) -> numpy.ndarray:
	"""Convert image to binary edge map for contour detection.

	Applies grayscale conversion, blur, and Canny edge detection,
	followed by morphological closing to handle broken edges from
	phone photos.

	Args:
		image: BGR input image

	Returns:
		binary edge image
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# blur to reduce noise
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	# canny edge detection
	edges = cv2.Canny(blurred, 50, 150)
	# morphological close to fill gaps in phone photo edges
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
	return closed


#============================================
def find_page_contour(image: numpy.ndarray, min_area_ratio: float = 0.20) -> numpy.ndarray:
	"""Find the largest quadrilateral contour in the image (the scantron page).

	Args:
		image: BGR input image
		min_area_ratio: minimum contour area as fraction of image area

	Returns:
		numpy array of 4 corner points, shape (4, 2)

	Raises:
		ValueError: if no suitable quadrilateral is found
	"""
	h, w = image.shape[:2]
	image_area = h * w
	min_area = min_area_ratio * image_area
	edges = preprocess_for_contours(image)
	# find external contours
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		# no contours found: treat entire image as the page
		corners = _full_image_corners(w, h)
		return order_corners(corners)
	# sort by area descending
	contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
	# try to find a quadrilateral among large contours
	for contour in contours_sorted:
		area = cv2.contourArea(contour)
		if area < min_area:
			break
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
		if len(approx) == 4:
			corners = approx.reshape(4, 2)
			return order_corners(corners)
	# fallback: try largest contour with convex hull
	largest = contours_sorted[0]
	largest_area = cv2.contourArea(largest)
	if largest_area < min_area:
		# contours too small -- likely a flatbed scan where page fills image
		# use full image boundary with a small inset to avoid edge artifacts
		corners = _full_image_corners(w, h)
		return order_corners(corners)
	hull = cv2.convexHull(largest)
	peri = cv2.arcLength(hull, True)
	# try progressively looser approximation
	for epsilon_mult in [0.02, 0.04, 0.06, 0.08, 0.10]:
		approx = cv2.approxPolyDP(hull, epsilon_mult * peri, True)
		if len(approx) == 4:
			corners = approx.reshape(4, 2)
			return order_corners(corners)
	# last resort: use bounding rect corners of the largest contour
	rect = cv2.minAreaRect(largest)
	box = cv2.boxPoints(rect)
	corners = numpy.int32(box)
	return order_corners(corners)


#============================================
def _full_image_corners(width: int, height: int, inset: int = 2) -> numpy.ndarray:
	"""Return corner points for the full image boundary with a small inset.

	Used when the scan fills the entire image (e.g. flatbed scans).

	Args:
		width: image width in pixels
		height: image height in pixels
		inset: pixel inset from edges to avoid artifacts

	Returns:
		array of shape (4, 2) with corner coordinates
	"""
	corners = numpy.array([
		[inset, inset],
		[width - inset, inset],
		[width - inset, height - inset],
		[inset, height - inset],
	], dtype=numpy.float32)
	return corners


#============================================
def order_corners(pts: numpy.ndarray) -> numpy.ndarray:
	"""Order 4 corner points as: top-left, top-right, bottom-right, bottom-left.

	Args:
		pts: array of shape (4, 2) with corner coordinates

	Returns:
		ordered array of shape (4, 2)
	"""
	pts = pts.astype(numpy.float32)
	# sort by sum (x+y): smallest = top-left, largest = bottom-right
	s = pts.sum(axis=1)
	# sort by difference (y-x): smallest = top-right, largest = bottom-left
	d = numpy.diff(pts, axis=1).flatten()
	ordered = numpy.zeros((4, 2), dtype=numpy.float32)
	ordered[0] = pts[numpy.argmin(s)]   # top-left
	ordered[1] = pts[numpy.argmin(d)]   # top-right
	ordered[2] = pts[numpy.argmax(s)]   # bottom-right
	ordered[3] = pts[numpy.argmax(d)]   # bottom-left
	return ordered


#============================================
def compute_output_dimensions(corners: numpy.ndarray) -> tuple:
	"""Compute output width and height from ordered corners.

	Uses the maximum of top/bottom widths and left/right heights
	to determine the warped output size.

	Args:
		corners: ordered corner array (4, 2)

	Returns:
		tuple of (width, height) in pixels
	"""
	tl, tr, br, bl = corners
	# top and bottom widths
	width_top = math.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
	width_bot = math.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
	max_width = int(max(width_top, width_bot))
	# left and right heights
	height_left = math.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
	height_right = math.sqrt((br[0] - tr[0])**2 + (br[1] - tr[1])**2)
	max_height = int(max(height_left, height_right))
	return (max_width, max_height)


#============================================
def warp_perspective(image: numpy.ndarray, corners: numpy.ndarray,
	output_width: int, output_height: int) -> numpy.ndarray:
	"""Apply perspective transform to warp the page to a rectangle.

	Args:
		image: BGR input image
		corners: ordered corner points (4, 2) from find_page_contour
		output_width: desired output width
		output_height: desired output height

	Returns:
		warped image of shape (output_height, output_width, 3)
	"""
	src = corners.astype(numpy.float32)
	dst = numpy.array([
		[0, 0],
		[output_width - 1, 0],
		[output_width - 1, output_height - 1],
		[0, output_height - 1],
	], dtype=numpy.float32)
	matrix = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(image, matrix, (output_width, output_height))
	return warped


#============================================
def detect_orientation(warped: numpy.ndarray) -> int:
	"""Detect the correct orientation of a warped scantron image.

	Checks for the DataLink 1200 form features to determine rotation.
	The correct orientation has:
	- Timing marks along the top and left edges
	- Dense content (answer bubbles) in the center-left area
	- The DataLink logo/header in the upper-right area

	Args:
		warped: perspective-corrected image

	Returns:
		rotation angle in degrees (0, 90, 180, or 270)
	"""
	gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	h, w = gray.shape
	best_rotation = 0
	best_score = -1
	for rotation in [0, 90, 180, 270]:
		if rotation == 0:
			test_img = gray
		else:
			# rotate the grayscale image
			test_img = rotate_image_90(gray, rotation)
		score = _score_orientation(test_img)
		if score > best_score:
			best_score = score
			best_rotation = rotation
	return best_rotation


#============================================
def _score_orientation(gray: numpy.ndarray) -> float:
	"""Score how well a grayscale image matches expected scantron orientation.

	Expected layout in correct orientation (portrait):
	- Top edge: row of timing marks (dark rectangles in a thin strip)
	- Left edge: column of timing marks
	- Upper-left quadrant: student ID grid (many small dark spots)
	- Center area: answer bubble grid

	Args:
		gray: grayscale image

	Returns:
		orientation confidence score (higher is better)
	"""
	h, w = gray.shape
	# threshold to binary
	_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	score = 0.0
	# check for timing marks along the top edge (narrow horizontal strip)
	top_strip = binary[0:int(h * 0.03), int(w * 0.05):int(w * 0.95)]
	if top_strip.size > 0:
		top_density = numpy.sum(top_strip > 0) / top_strip.size
		# timing marks should create moderate density (not solid, not empty)
		if 0.05 < top_density < 0.60:
			score += top_density * 10
	# check for timing marks along the left edge (narrow vertical strip)
	left_strip = binary[int(h * 0.05):int(h * 0.95), 0:int(w * 0.03)]
	if left_strip.size > 0:
		left_density = numpy.sum(left_strip > 0) / left_strip.size
		if 0.05 < left_density < 0.60:
			score += left_density * 10
	# check for content density in the answer area (middle-left portion)
	answer_area = binary[int(h * 0.25):int(h * 0.95), int(w * 0.05):int(w * 0.45)]
	if answer_area.size > 0:
		answer_density = numpy.sum(answer_area > 0) / answer_area.size
		# answer bubbles area should have moderate density
		if 0.02 < answer_density < 0.30:
			score += answer_density * 5
	# portrait orientation: width < height
	if w < h:
		score += 2.0
	return score


#============================================
def rotate_image_90(image: numpy.ndarray, degrees: int) -> numpy.ndarray:
	"""Rotate an image by a multiple of 90 degrees.

	Args:
		image: input image (grayscale or color)
		degrees: rotation angle (90, 180, or 270)

	Returns:
		rotated image
	"""
	if degrees == 90:
		return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
	elif degrees == 180:
		return cv2.rotate(image, cv2.ROTATE_180)
	elif degrees == 270:
		return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
	return image


#============================================
def register_image(image: numpy.ndarray) -> numpy.ndarray:
	"""Full registration pipeline: detect page, warp, and orient.

	Args:
		image: BGR input image (raw photo or scan)

	Returns:
		registered BGR image (perspective-corrected and oriented)
	"""
	# detect page boundary
	corners = find_page_contour(image)
	# compute natural output size from corners
	nat_width, nat_height = compute_output_dimensions(corners)
	# warp to rectangle at natural resolution
	warped = warp_perspective(image, corners, nat_width, nat_height)
	# detect and correct orientation
	rotation = detect_orientation(warped)
	if rotation != 0:
		warped = rotate_image_90(warped, rotation)
	registered = warped
	return registered


#============================================
def draw_debug_overlay(image: numpy.ndarray, corners: numpy.ndarray,
	template: dict = None) -> numpy.ndarray:
	"""Draw debug information on a copy of the image.

	Draws the detected page contour corners and optionally the
	template grid overlay if a template is provided.

	Args:
		image: original BGR image
		corners: ordered corner points (4, 2)
		template: optional template dict for grid overlay

	Returns:
		annotated copy of the image
	"""
	debug = image.copy()
	h, w = debug.shape[:2]
	# draw page contour
	pts = corners.astype(numpy.int32).reshape((-1, 1, 2))
	cv2.polylines(debug, [pts], True, (0, 255, 0), 3)
	# draw corner labels
	labels = ["TL", "TR", "BR", "BL"]
	for i, label in enumerate(labels):
		pt = tuple(corners[i].astype(int))
		cv2.circle(debug, pt, 10, (0, 0, 255), -1)
		cv2.putText(debug, label, (pt[0] + 15, pt[1] + 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	return debug


