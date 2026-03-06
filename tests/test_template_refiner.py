"""Unit tests for omr_utils.template_refiner."""

# PIP3 modules
import pytest

# local repo modules
import omr_utils.template_loader
import omr_utils.template_refiner


#============================================
@pytest.fixture()
def mini_template() -> dict:
	"""Minimal template with two short answer columns."""
	return {
		"form": {"name": "mini", "orientation": "portrait"},
		"template_version": 2,
		"canonical": {"width_px": 1000, "height_px": 2000},
		"student_id": {
			"num_digits": 1,
			"grid": {
				"first_digit_x": 0.1,
				"digit_spacing_x": 0.1,
				"digit_values": [0],
				"first_value_y": 0.1,
				"value_spacing_y": 0.1,
			},
			"bubble_radius": 0.006,
		},
		"answers": {
			"num_questions": 6,
			"choices": ["A", "B", "C", "D", "E"],
			"bubble_radius": 0.006,
			"bubble_shape": {
				"aspect_ratio": 5.454545,
				"target_area_px_at_canonical": 660.0,
			},
			"left_column": {
				"question_range": [1, 3],
				"first_question_y": 0.2000,
				"question_spacing_y": 0.0100,
				"choice_x": {
					"A": 0.1000, "B": 0.1500, "C": 0.2000,
					"D": 0.2500, "E": 0.3000,
				},
			},
			"right_column": {
				"question_range": [4, 6],
				"first_question_y": 0.2000,
				"question_spacing_y": 0.0100,
				"choice_x": {
					"A": 0.5000, "B": 0.5500, "C": 0.6000,
					"D": 0.6500, "E": 0.7000,
				},
			},
		},
	}


#============================================
def test_collect_empty_offsets_uses_non_selected_low_scores(
	mini_template: dict) -> None:
	"""Only likely-empty choices should be collected as offset candidates."""
	w = mini_template["canonical"]["width_px"]
	h = mini_template["canonical"]["height_px"]
	results = []
	choices = mini_template["answers"]["choices"]
	for q_num in range(1, 7):
		positions = {}
		scores = {}
		for choice in choices:
			norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
				mini_template, q_num, choice)
			px, py = omr_utils.template_loader.to_pixels(norm_x, norm_y, w, h)
			positions[choice] = (px + 3, py + 2)
			if choice == "A":
				scores[choice] = 0.75
			else:
				scores[choice] = 0.05
		results.append({
			"question": q_num,
			"answer": "A",
			"scores": scores,
			"flags": "",
			"positions": positions,
		})
	offsets = omr_utils.template_refiner.collect_empty_offsets(
		mini_template, results, w, h, empty_score_max=0.12)
	# 6 questions * 4 non-selected choices
	assert len(offsets) == 24
	for entry in offsets:
		assert entry["choice"] != "A"
		assert entry["dx"] == 3.0
		assert entry["dy"] == 2.0


#============================================
def test_apply_refined_offsets_updates_column_model(
	mini_template: dict) -> None:
	"""Per-bubble medians should fit to updated first_y/spacing/choice_x."""
	w = mini_template["canonical"]["width_px"]
	h = mini_template["canonical"]["height_px"]
	# target refined model we expect to recover
	target = {
		"left_column": {
			"first_question_y": 0.2050,
			"question_spacing_y": 0.0105,
			"choice_x_delta": 0.0020,
		},
		"right_column": {
			"first_question_y": 0.1980,
			"question_spacing_y": 0.0095,
			"choice_x_delta": -0.0010,
		},
	}
	aggregated = {}
	for column_key in ["left_column", "right_column"]:
		col = mini_template["answers"][column_key]
		q_start, q_end = col["question_range"]
		for q_num in range(q_start, q_end + 1):
			row_idx = q_num - q_start
			for choice in mini_template["answers"]["choices"]:
				old_x, old_y = omr_utils.template_loader.get_bubble_coords(
					mini_template, q_num, choice)
				new_x = old_x + target[column_key]["choice_x_delta"]
				new_y = (target[column_key]["first_question_y"]
					+ row_idx * target[column_key]["question_spacing_y"])
				aggregated[(q_num, choice)] = {
					"dx": (new_x - old_x) * w,
					"dy": (new_y - old_y) * h,
					"count": 3,
				}
	refined = omr_utils.template_refiner.apply_refined_offsets(
		mini_template, aggregated, w, h)
	left = refined["answers"]["left_column"]
	right = refined["answers"]["right_column"]
	assert left["first_question_y"] == pytest.approx(0.205, abs=1e-6)
	assert left["question_spacing_y"] == pytest.approx(0.0105, abs=1e-6)
	assert right["first_question_y"] == pytest.approx(0.198, abs=1e-6)
	assert right["question_spacing_y"] == pytest.approx(0.0095, abs=1e-6)
	assert left["choice_x"]["A"] == pytest.approx(0.102, abs=1e-6)
	assert left["choice_x"]["E"] == pytest.approx(0.302, abs=1e-6)
	assert right["choice_x"]["A"] == pytest.approx(0.499, abs=1e-6)
	assert right["choice_x"]["E"] == pytest.approx(0.699, abs=1e-6)
