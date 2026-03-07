"""Offline template construction pipeline for bubble letter templates.

Builds per-letter (A-E) reference templates from extracted ROIs using a
cryoEM-inspired class averaging approach: upscale to canonical resolution,
align via NCC, reject outliers by mirror symmetry, and produce a sharp
trimmed-mean template.

Coordinate transitions in this module:
  native-resolution ROI -> canonical 480x88 -> NCC-aligned canonical
  -> symmetry-regularized -> trimmed-mean template

This module is used only by tools/build_bubble_templates.py (offline).
Runtime grading code uses omr_utils/bubble_template_extractor.py instead.
"""

# Standard Library
import os
import sys
import time

# PIP3 modules
import cv2
import numpy
import scipy.stats

# local repo modules
import omr_utils.bubble_template_extractor

# Canonical template resolution: all offline-built templates are stored
# at this fixed size so runtime scaling always goes DOWN.
# Chosen to be ~8x the typical runtime bubble size (target ~60x11 px),
# keeping detail without being absurdly oversized.
CANONICAL_TEMPLATE_WIDTH = 480
CANONICAL_TEMPLATE_HEIGHT = 88

# ANSI color codes for stderr diagnostics
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

# symmetry axis rules per letter
_SYMMETRY_AXIS = {
	"A": "lr",  # left-right mirror
	"B": "tb",  # top-bottom mirror
	"C": "tb",
	"D": "tb",
	"E": "tb",
}


#============================================
def _log_image_saved(filepath: str) -> None:
	"""Print a colored diagnostic to stderr when an image is saved.

	Args:
		filepath: path to the saved image file
	"""
	sys.stderr.write(
		f"  {_GREEN}SAVED{_RESET} {_CYAN}{filepath}{_RESET}\n")


#============================================
def _filter_dark_rois(rois: list,
	reject_fraction: float = 0.2) -> tuple:
	"""Discard the darkest ROIs by mean grayscale intensity.

	Filled bubbles are darker than empty printed brackets. This
	physically interpretable first-pass filter removes the darkest
	fraction of ROIs before any NCC-based processing, preventing
	filled bubbles from contaminating medoid selection and alignment.

	Applied per image, per letter -- never globally.

	Args:
		rois: list of grayscale numpy arrays
		reject_fraction: fraction of darkest ROIs to discard (0.2 = 20%)

	Returns:
		tuple of (kept_rois, rejected_rois, means, cutoff_value)
		where means is a list of per-ROI mean intensities and
		cutoff_value is the threshold used for filtering.
		If fewer than 3 would survive, returns all (skip filtering).
	"""
	if len(rois) < 3:
		# not enough to filter; return all as kept
		means = [float(numpy.mean(r)) for r in rois]
		return (list(rois), [], means, 0.0)
	# compute mean grayscale intensity for each ROI
	means = [float(numpy.mean(r)) for r in rois]
	# sort indices by mean ascending (darkest first)
	sorted_indices = sorted(range(len(rois)), key=lambda k: means[k])
	# number to reject
	n_reject = int(len(rois) * reject_fraction)
	# safety: ensure at least 3 survive
	n_kept = len(rois) - n_reject
	if n_kept < 3:
		return (list(rois), [], means, 0.0)
	# split into rejected (darkest) and kept (brighter)
	reject_set = set(sorted_indices[:n_reject])
	# cutoff is the mean intensity of the last rejected ROI
	cutoff_value = means[sorted_indices[n_reject - 1]] if n_reject > 0 else 0.0
	kept_rois = []
	rejected_rois = []
	for i, roi in enumerate(rois):
		if i in reject_set:
			rejected_rois.append(roi)
		else:
			kept_rois.append(roi)
	return (kept_rois, rejected_rois, means, cutoff_value)


#============================================
def _save_filter_qc(rois_before: list, rois_after: list,
	image_id: str, letter: str, output_dir: str,
	norm_rois_before: list = None,
	norm_rois_after: list = None) -> None:
	"""Save pre/post darkness filter QC images for one image-letter group.

	Saves six raw images per group, plus two normalized montages when
	normalized ROIs are provided:
	- Raw ROI montage before filtering
	- Simple mean ROI before filtering
	- Simple median ROI before filtering
	- ROI montage after darkest-20% reject
	- Mean ROI after filtering
	- Median ROI after filtering
	- Normalized montage before filtering (if norm_rois_before given)
	- Normalized montage after filtering (if norm_rois_after given)

	Args:
		rois_before: list of raw ROIs before darkness filtering
		rois_after: list of raw ROIs after darkness filtering
		image_id: scan identifier string
		letter: bubble letter (A-E)
		output_dir: directory for QC output
		norm_rois_before: optional normalized ROIs before filtering
		norm_rois_after: optional normalized ROIs after filtering
	"""
	qc_subdir = os.path.join(output_dir, "qc_darkness_filter")
	os.makedirs(qc_subdir, exist_ok=True)
	prefix = f"{image_id}_{letter}"
	# helper: build a montage from a list of ROIs
	def _build_montage(roi_list: list) -> numpy.ndarray:
		if not roi_list:
			return numpy.full((20, 20), 200, dtype=numpy.uint8)
		# find common size (smallest dimensions)
		min_h = min(r.shape[0] for r in roi_list)
		min_w = min(r.shape[1] for r in roi_list)
		# show up to 50 ROIs
		show = roi_list[:50]
		cols = min(len(show), 10)
		rows = (len(show) + cols - 1) // cols
		montage = numpy.full((min_h * rows, min_w * cols), 200,
			dtype=numpy.uint8)
		for idx, roi in enumerate(show):
			r = idx // cols
			c = idx % cols
			# resize to common cell
			cell = cv2.resize(roi, (min_w, min_h),
				interpolation=cv2.INTER_AREA)
			montage[r * min_h:(r + 1) * min_h,
				c * min_w:(c + 1) * min_w] = cell
		return montage

	# helper: compute mean/median from ROI list
	def _compute_avg(roi_list: list, mode: str) -> numpy.ndarray:
		if not roi_list:
			return numpy.full((20, 20), 200, dtype=numpy.uint8)
		min_h = min(r.shape[0] for r in roi_list)
		min_w = min(r.shape[1] for r in roi_list)
		resized = []
		for r in roi_list:
			resized.append(cv2.resize(r, (min_w, min_h),
				interpolation=cv2.INTER_AREA).astype(numpy.float32))
		stack = numpy.stack(resized, axis=0)
		if mode == "mean":
			avg = numpy.mean(stack, axis=0)
		else:
			avg = numpy.median(stack, axis=0)
		result = numpy.clip(avg, 0, 255).astype(numpy.uint8)
		return result

	# save before-filter images
	montage_before = _build_montage(rois_before)
	path = os.path.join(qc_subdir, f"{prefix}_1_raw_montage.png")
	cv2.imwrite(path, montage_before)
	_log_image_saved(path)
	mean_before = _compute_avg(rois_before, "mean")
	path = os.path.join(qc_subdir, f"{prefix}_2_raw_mean.png")
	cv2.imwrite(path, mean_before)
	_log_image_saved(path)
	median_before = _compute_avg(rois_before, "median")
	path = os.path.join(qc_subdir, f"{prefix}_3_raw_median.png")
	cv2.imwrite(path, median_before)
	_log_image_saved(path)
	# save after-filter images
	montage_after = _build_montage(rois_after)
	path = os.path.join(qc_subdir, f"{prefix}_4_filtered_montage.png")
	cv2.imwrite(path, montage_after)
	_log_image_saved(path)
	mean_after = _compute_avg(rois_after, "mean")
	path = os.path.join(qc_subdir, f"{prefix}_5_filtered_mean.png")
	cv2.imwrite(path, mean_after)
	_log_image_saved(path)
	median_after = _compute_avg(rois_after, "median")
	path = os.path.join(qc_subdir, f"{prefix}_6_filtered_median.png")
	cv2.imwrite(path, median_after)
	_log_image_saved(path)
	# save normalized montages if provided
	if norm_rois_before is not None:
		norm_montage_before = _build_montage(norm_rois_before)
		path = os.path.join(qc_subdir,
			f"{prefix}_1b_norm_montage.png")
		cv2.imwrite(path, norm_montage_before)
		_log_image_saved(path)
	if norm_rois_after is not None:
		norm_montage_after = _build_montage(norm_rois_after)
		path = os.path.join(qc_subdir,
			f"{prefix}_4b_norm_filtered_montage.png")
		cv2.imwrite(path, norm_montage_after)
		_log_image_saved(path)


#============================================
def _find_medoid_roi(rois: list) -> int:
	"""Select the ROI with highest average NCC to all others as reference.

	The medoid is the most representative sample: the one that best
	matches the rest of the collection.

	Args:
		rois: list of grayscale numpy arrays (all same shape)

	Returns:
		index of the medoid ROI in the input list
	"""
	n = len(rois)
	if n <= 1:
		return 0
	# medoid selection assumes all ROIs are already shape-normalized;
	# fail fast instead of silently blurring via resize.
	ref_shape = rois[0].shape
	for idx, roi in enumerate(rois):
		if roi.shape != ref_shape:
			raise ValueError(
				f"_find_medoid_roi expects same-shape ROIs; "
				f"idx0={ref_shape}, idx{idx}={roi.shape}")
	# compute pairwise NCC scores
	avg_scores = numpy.zeros(n, dtype=numpy.float64)
	for i in range(n):
		# print progress every 200 candidates
		if i > 0 and i % 200 == 0:
			print(f"      medoid: {i}/{n} candidates scored")
		total = 0.0
		count = 0
		for j in range(n):
			if i == j:
				continue
			# direct pixel correlation (no template matching needed,
			# since they are the same size)
			ri = rois[i].astype(numpy.float64)
			rj = rois[j].astype(numpy.float64)
			ri_norm = ri - ri.mean()
			rj_norm = rj - rj.mean()
			denom = numpy.sqrt(numpy.sum(ri_norm ** 2) * numpy.sum(rj_norm ** 2))
			if denom < 1e-6:
				continue
			ncc = float(numpy.sum(ri_norm * rj_norm) / denom)
			total += ncc
			count += 1
		if count > 0:
			avg_scores[i] = total / count
	medoid_idx = int(numpy.argmax(avg_scores))
	return medoid_idx


#============================================
def _align_roi_to_reference(roi: numpy.ndarray,
	reference: numpy.ndarray,
	search_margin: int = 15) -> tuple:
	"""Align an ROI to a reference via translation-only NCC.

	Pads the ROI with white (255) border to create a search window,
	then slides the reference across it via cv2.matchTemplate.
	The search_margin controls how many pixels of shift can be
	detected in each direction.

	Args:
		roi: grayscale ROI to align
		reference: grayscale reference template
		search_margin: pixels of padding on each side for the
			search window (default 15)

	Returns:
		tuple of (aligned_roi, dx, dy, score) where aligned_roi is
		cropped to match reference dimensions, dx/dy are the shift,
		and score is the NCC peak value. Returns (center_crop,
		0, 0, 0.0) if matching fails.
	"""
	rh, rw = roi.shape
	th, tw = reference.shape
	# build an explicit white canvas and place ROI centered inside it;
	# this guarantees a translation search even when roi.shape == reference.shape.
	canvas_h = max(rh + (2 * search_margin), th)
	canvas_w = max(rw + (2 * search_margin), tw)
	canvas = numpy.full((canvas_h, canvas_w), 255, dtype=numpy.uint8)
	roi_y = (canvas_h - rh) // 2
	roi_x = (canvas_w - rw) // 2
	canvas[roi_y:roi_y + rh, roi_x:roi_x + rw] = roi
	# run NCC template matching on the padded canvas
	result = cv2.matchTemplate(canvas, reference, cv2.TM_CCOEFF_NORMED)
	if result is None or result.size == 0:
		fallback = cv2.resize(roi, (tw, th), interpolation=cv2.INTER_AREA)
		return (fallback, 0, 0, 0.0)
	_, max_val, _, max_loc = cv2.minMaxLoc(result)
	score = float(max_val)
	# max_loc is top-left of best match in canvas coordinates
	best_x, best_y = max_loc
	# no-shift reference top-left when centered is the ROI insertion point
	dx = best_x - roi_x
	dy = best_y - roi_y
	# crop aligned region from canvas at the matched position
	aligned = canvas[best_y:best_y + th, best_x:best_x + tw].copy()
	return (aligned, dx, dy, score)


#============================================
def _upscale_rois_to_canonical(rois: list) -> list:
	"""Upscale all ROIs to canonical high resolution for template construction.

	Eliminates size variation across scans (+/-1-2 px becomes irrelevant)
	and gives NCC alignment more pixels to work with. This is a
	template-construction canvas normalization step only; it does not
	change SlotMap geometry or ROI ownership.

	Args:
		rois: list of grayscale numpy arrays at native resolution

	Returns:
		list of grayscale numpy arrays all at CANONICAL_TEMPLATE_WIDTH x
		CANONICAL_TEMPLATE_HEIGHT
	"""
	canonical_w = CANONICAL_TEMPLATE_WIDTH
	canonical_h = CANONICAL_TEMPLATE_HEIGHT
	upscaled = []
	for roi in rois:
		resized = cv2.resize(roi, (canonical_w, canonical_h),
			interpolation=cv2.INTER_CUBIC)
		upscaled.append(resized)
	return upscaled


#============================================
def _enforce_symmetry_image(roi: numpy.ndarray, letter: str) -> numpy.ndarray:
	"""Average a single ROI with its own mirror for symmetry regularization.

	Applied after alignment to sharpen features by exploiting known
	letter symmetry. A has left-right symmetry; B/C/D/E have
	top-bottom symmetry.

	Args:
		roi: grayscale uint8 array
		letter: single letter string (A-E)

	Returns:
		symmetry-regularized uint8 array, same shape as input
	"""
	axis = _SYMMETRY_AXIS.get(letter, "tb")
	if axis == "lr":
		mirrored = numpy.fliplr(roi)
	else:
		mirrored = numpy.flipud(roi)
	# average original and mirror, cast back to uint8
	blended = (roi.astype(numpy.float32) + mirrored.astype(numpy.float32)) / 2.0
	result = numpy.clip(blended, 0, 255).astype(numpy.uint8)
	return result


#============================================
def _enforce_symmetry_list(rois: list, letter: str) -> list:
	"""Apply symmetry regularization to each ROI in a list.

	Calls _enforce_symmetry_image() on each ROI individually.
	Applied after alignment, not before.

	Args:
		rois: list of grayscale numpy arrays
		letter: single letter string (A-E)

	Returns:
		same-length list of symmetry-regularized uint8 arrays
	"""
	result = []
	for roi in rois:
		result.append(_enforce_symmetry_image(roi, letter))
	return result


#============================================
def _mirror_ncc_score(roi: numpy.ndarray, letter: str) -> float:
	"""Compute NCC between an ROI and its own mirror.

	Measures how symmetric a single ROI is along the letter's known
	symmetry axis. High score means the ROI already looks like its
	mirror; low score means it is asymmetric (likely misaligned,
	damaged, or filled).

	Args:
		roi: grayscale uint8 array
		letter: single letter string (A-E)

	Returns:
		NCC score between roi and its mirror (float, -1 to 1)
	"""
	axis = _SYMMETRY_AXIS.get(letter, "tb")
	if axis == "lr":
		mirrored = numpy.fliplr(roi)
	else:
		mirrored = numpy.flipud(roi)
	# compute NCC between original and mirror
	r = roi.astype(numpy.float64)
	m = mirrored.astype(numpy.float64)
	r_norm = r - r.mean()
	m_norm = m - m.mean()
	denom = numpy.sqrt(numpy.sum(r_norm ** 2) * numpy.sum(m_norm ** 2))
	if denom < 1e-6:
		return 0.0
	ncc = float(numpy.sum(r_norm * m_norm) / denom)
	return ncc


#============================================
def _reject_asymmetric_rois(rois: list, letter: str,
	reject_fraction: float = 0.1) -> tuple:
	"""Reject the least symmetric ROIs by mirror NCC score.

	After alignment, each ROI should be fairly symmetric along the
	letter's known axis. ROIs that score poorly are likely misaligned
	or contain filled bubbles that slipped through earlier filters.

	Args:
		rois: list of grayscale numpy arrays (all same shape)
		letter: single letter string (A-E)
		reject_fraction: fraction of worst-scoring ROIs to reject

	Returns:
		tuple of (kept_rois, kept_indices, scores) where scores is
		the full list of per-ROI mirror NCC values. If fewer than 3
		would survive, returns all (skip rejection).
	"""
	# compute mirror NCC for each ROI
	scores = [_mirror_ncc_score(r, letter) for r in rois]
	n_reject = int(len(rois) * reject_fraction)
	n_kept = len(rois) - n_reject
	# safety: need at least 3 survivors
	if n_kept < 3:
		kept_indices = list(range(len(rois)))
		return (list(rois), kept_indices, scores)
	# sort indices by score ascending (least symmetric first)
	sorted_indices = sorted(range(len(rois)), key=lambda k: scores[k])
	reject_set = set(sorted_indices[:n_reject])
	kept_rois = []
	kept_indices = []
	for i, roi in enumerate(rois):
		if i not in reject_set:
			kept_rois.append(roi)
			kept_indices.append(i)
	return (kept_rois, kept_indices, scores)


#============================================
def _build_small_montage(rois: list, max_count: int = 100) -> numpy.ndarray:
	"""Build a montage grid from a list of same-sized ROIs.

	Args:
		rois: list of grayscale numpy arrays (should all be same size)
		max_count: maximum number of ROIs to include

	Returns:
		grayscale montage image
	"""
	if not rois:
		return numpy.full((20, 20), 200, dtype=numpy.uint8)
	show = rois[:max_count]
	cell_h, cell_w = show[0].shape[:2]
	cols = min(len(show), 10)
	rows = (len(show) + cols - 1) // cols
	montage = numpy.full((cell_h * rows, cell_w * cols), 200,
		dtype=numpy.uint8)
	for idx, roi in enumerate(show):
		r = idx // cols
		c = idx % cols
		# resize to cell size if needed
		if roi.shape[0] != cell_h or roi.shape[1] != cell_w:
			cell = cv2.resize(roi, (cell_w, cell_h),
				interpolation=cv2.INTER_AREA)
		else:
			cell = roi
		montage[r * cell_h:(r + 1) * cell_h,
			c * cell_w:(c + 1) * cell_w] = cell
	return montage


#============================================
def _build_letter_template(rois: list, letter: str,
	reject_threshold: float = 0.5,
	output_dir: str = None,
	base_reference: numpy.ndarray = None) -> tuple:
	"""Build a sharp per-letter template via two-pass alignment.

	Pipeline:
	  upscale_to_canonical -> reference -> pass1_align -> interim_avg
	  -> symmetry_image(interim) -> pass2_align -> mirror_reject
	  -> symmetry_list -> trim_mean

	When base_reference is provided (480x88 grayscale), pass 1
	aligns to the base reference instead of computing a medoid.
	This anchors bracket positions to a common reference across
	all letters. Pass 2 re-aligns original canonical ROIs to the
	symmetry-regularized interim average. After pass 2, the worst
	10% by mirror NCC are rejected, then symmetry regularization
	and trimmed mean produce the final template.

	Args:
		rois: list of native-resolution grayscale ROI arrays
		letter: bubble letter (A-E)
		reject_threshold: minimum NCC score to keep an aligned ROI
		output_dir: optional directory for per-pass QC images
		base_reference: optional 480x88 grayscale image to use as
			pass-1 alignment target instead of medoid

	Returns:
		tuple of (template, alignment_table) where:
		- template: uint8 grayscale array at canonical resolution
		- alignment_table: list of dicts with dx, dy, score, kept, pass
		Returns (None, []) if fewer than 3 ROIs survive
	"""
	if len(rois) < 3:
		return (None, [])
	# --- upscale all ROIs to canonical resolution up front ---
	canonical_rois = _upscale_rois_to_canonical(rois)
	# --- re-normalize after upscale (1% black / 25% white stretch) ---
	# cubic interpolation can shift intensities; re-normalizing ensures
	# consistent contrast for NCC alignment and mirror scoring
	canonical_rois = [
		omr_utils.bubble_template_extractor.normalize_roi_percentile(r)
		for r in canonical_rois
	]
	n_rois = len(canonical_rois)
	# --- select pass-1 alignment reference ---
	if base_reference is not None:
		# validate base reference dimensions match canonical size
		expected_h = CANONICAL_TEMPLATE_HEIGHT
		expected_w = CANONICAL_TEMPLATE_WIDTH
		if (base_reference.shape[0] != expected_h
			or base_reference.shape[1] != expected_w):
			raise ValueError(
				f"base_reference must be {expected_w}x{expected_h}, "
				f"got {base_reference.shape[1]}x{base_reference.shape[0]}")
		# use base reference as pass-1 target (skip medoid)
		pass1_ref = base_reference
		print(f"    using base reference as pass-1 target"
			f" ({expected_w}x{expected_h})")
	else:
		# --- medoid selection on canonical-sized ROIs ---
		print(f"    finding medoid in {n_rois} canonical ROIs...")
		t0 = time.time()
		medoid_idx = _find_medoid_roi(canonical_rois)
		medoid_elapsed = time.time() - t0
		print(f"    medoid found (idx {medoid_idx}, {medoid_elapsed:.1f}s)")
		pass1_ref = canonical_rois[medoid_idx]
	# save pass-1 reference QC image
	if output_dir is not None:
		qc_subdir = os.path.join(output_dir, "qc")
		os.makedirs(qc_subdir, exist_ok=True)
		# label reflects whether it was medoid or base reference
		ref_label = "base_ref" if base_reference is not None else "medoid"
		path = os.path.join(qc_subdir, f"qc_{letter}_{ref_label}.png")
		cv2.imwrite(path, pass1_ref)
		_log_image_saved(path)
	# === PASS 1: align to reference ===
	ref_label = "base reference" if base_reference is not None else "medoid"
	print(f"    pass 1: aligning {n_rois} ROIs to {ref_label}...")
	t1 = time.time()
	pass1_aligned = []
	alignment_table = []
	for i, roi in enumerate(canonical_rois):
		aligned, dx, dy, score = _align_roi_to_reference(roi, pass1_ref)
		entry = {"index": i, "dx": dx, "dy": dy, "score": score,
			"kept": True, "pass": 1}
		alignment_table.append(entry)
		pass1_aligned.append(aligned)
	p1_elapsed = time.time() - t1
	# compute pass-1 median alignment score
	p1_scores = [e["score"] for e in alignment_table]
	p1_median_score = float(numpy.median(p1_scores))
	print(f"    pass 1 done ({p1_elapsed:.1f}s,"
		f" median NCC={p1_median_score:.4f})")
	if len(pass1_aligned) < 3:
		return (None, alignment_table)
	# --- reject worst 20% by mirror symmetry for pass-1 average ---
	# these are excluded from the interim average only; all canonical
	# ROIs still get a second chance in pass 2 against the sharper ref
	p1_kept, p1_kept_indices, p1_mirror_scores = _reject_asymmetric_rois(
		pass1_aligned, letter, reject_fraction=0.5)
	n_p1_mirror_rejected = len(pass1_aligned) - len(p1_kept)
	p1_mirror_arr = numpy.array(p1_mirror_scores)
	print(f"    pass 1 mirror symmetry: min={float(p1_mirror_arr.min()):.4f}"
		f" median={float(numpy.median(p1_mirror_arr)):.4f}"
		f" max={float(p1_mirror_arr.max()):.4f}"
		f" rejected={n_p1_mirror_rejected}")
	if len(p1_kept) < 3:
		return (None, alignment_table)
	# build pass-1 interim average from kept ROIs only
	p1_stack = numpy.stack(
		[a.astype(numpy.float32) for a in p1_kept], axis=0)
	p1_avg = numpy.clip(numpy.mean(p1_stack, axis=0), 0, 255).astype(
		numpy.uint8)
	# apply symmetry regularization to interim average for pass-2 reference
	p1_sym_avg = _enforce_symmetry_image(p1_avg, letter)
	# save pass-1 QC images
	if output_dir is not None:
		qc_subdir = os.path.join(output_dir, "qc")
		# pass-1 average before symmetry
		path = os.path.join(qc_subdir, f"qc_{letter}_pass1_avg.png")
		cv2.imwrite(path, p1_avg)
		_log_image_saved(path)
		# pass-1 average after symmetry (pass-2 reference)
		path = os.path.join(qc_subdir, f"qc_{letter}_pass1_sym_avg.png")
		cv2.imwrite(path, p1_sym_avg)
		_log_image_saved(path)
		# pass-1 montage of aligned ROIs, sorted best-to-worst by mirror NCC
		p1_sorted_indices = sorted(range(len(pass1_aligned)),
			key=lambda k: p1_mirror_scores[k], reverse=True)
		p1_sorted_rois = [pass1_aligned[i] for i in p1_sorted_indices]
		p1_montage = _build_small_montage(p1_sorted_rois, max_count=100)
		path = os.path.join(qc_subdir,
			f"qc_{letter}_pass1_montage.png")
		cv2.imwrite(path, p1_montage)
		_log_image_saved(path)
	# === PASS 2: re-align original canonical ROIs to symmetrized average ===
	print(f"    pass 2: re-aligning {n_rois} ROIs"
		" to symmetrized pass-1 average...")
	t2 = time.time()
	pass2_aligned = []
	pass2_table = []
	for i, roi in enumerate(canonical_rois):
		aligned, dx, dy, score = _align_roi_to_reference(roi, p1_sym_avg)
		entry = {"index": i, "dx": dx, "dy": dy, "score": score,
			"kept": True, "pass": 2}
		pass2_table.append(entry)
		pass2_aligned.append(aligned)
	p2_elapsed = time.time() - t2
	# compute pass-2 median alignment score
	p2_scores = [e["score"] for e in pass2_table]
	p2_median_score = float(numpy.median(p2_scores))
	print(f"    pass 2 done ({p2_elapsed:.1f}s,"
		f" median NCC={p2_median_score:.4f})")
	# append pass-2 entries to alignment table
	alignment_table.extend(pass2_table)
	if len(pass2_aligned) < 3:
		return (None, alignment_table)
	# --- reject least symmetric ROIs by mirror NCC ---
	pass2_kept, kept_indices, mirror_scores = _reject_asymmetric_rois(
		pass2_aligned, letter, reject_fraction=0.1)
	n_mirror_rejected = len(pass2_aligned) - len(pass2_kept)
	# mark rejected entries in pass2_table
	reject_set = set(range(len(pass2_aligned))) - set(kept_indices)
	for idx in reject_set:
		pass2_table[idx]["kept"] = False
	# compute mirror score stats for console output
	mirror_arr = numpy.array(mirror_scores)
	print(f"    mirror symmetry: min={float(mirror_arr.min()):.4f}"
		f" median={float(numpy.median(mirror_arr)):.4f}"
		f" max={float(mirror_arr.max()):.4f}"
		f" rejected={n_mirror_rejected}")
	if len(pass2_kept) < 3:
		return (None, alignment_table)
	# apply symmetry regularization to surviving pass-2 ROIs
	pass2_sym = _enforce_symmetry_list(pass2_kept, letter)
	# build pass-2 average before symmetry (for QC comparison)
	p2_stack_raw = numpy.stack(
		[a.astype(numpy.float32) for a in pass2_kept], axis=0)
	p2_avg_raw = numpy.clip(numpy.mean(p2_stack_raw, axis=0),
		0, 255).astype(numpy.uint8)
	# trimmed mean on symmetrized pass-2 ROIs: reject top/bottom 10%
	p2_stack = numpy.stack(
		[s.astype(numpy.float32) for s in pass2_sym], axis=0)
	template_float = scipy.stats.trim_mean(
		p2_stack, proportiontocut=0.1, axis=0)
	template = numpy.clip(template_float, 0, 255).astype(numpy.uint8)
	# save pass-2 QC images
	if output_dir is not None:
		qc_subdir = os.path.join(output_dir, "qc")
		# pass-2 average before symmetry
		path = os.path.join(qc_subdir, f"qc_{letter}_pass2_avg.png")
		cv2.imwrite(path, p2_avg_raw)
		_log_image_saved(path)
		# pass-2 average after symmetry (final template)
		path = os.path.join(qc_subdir, f"qc_{letter}_pass2_sym_avg.png")
		cv2.imwrite(path, template)
		_log_image_saved(path)
		# pass-2 montage of aligned ROIs, sorted best-to-worst by mirror NCC
		p2_sorted_indices = sorted(range(len(pass2_aligned)),
			key=lambda k: mirror_scores[k], reverse=True)
		p2_sorted_rois = [pass2_aligned[i] for i in p2_sorted_indices]
		p2_montage = _build_small_montage(p2_sorted_rois, max_count=100)
		path = os.path.join(qc_subdir,
			f"qc_{letter}_pass2_montage.png")
		cv2.imwrite(path, p2_montage)
		_log_image_saved(path)
	# print score comparison
	print(f"    NCC improvement: pass1={p1_median_score:.4f}"
		f" -> pass2={p2_median_score:.4f}")
	return (template, alignment_table)
