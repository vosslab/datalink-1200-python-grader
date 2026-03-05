"""Tests for omr_utils.image_registration using real scantron images."""

# Standard Library
import os

# PIP3 modules
import numpy
import pytest

# local repo modules
import git_file_utils
import omr_utils.image_registration

REPO_ROOT = git_file_utils.get_repo_root()
SCANTRON_DIR = os.path.join(REPO_ROOT, "scantrons")

# test image paths
BOTELLO = os.path.join(SCANTRON_DIR, "Botello_2.24.26.jpg")
KEY_SHEET = os.path.join(SCANTRON_DIR, "8B5D0C61-395B-4E28-AED1-3E0D9959FAE0_result.jpg")
STUDENT_43F = os.path.join(SCANTRON_DIR, "43F257A7-A03D-4CB2-8D7B-3EE057B41FAC_result.jpg")
WOJTASZEK = os.path.join(SCANTRON_DIR, "Wojtaszek,Jessica_02-24-26.jpg")


#============================================
def _skip_if_no_scantrons() -> None:
	"""Skip test if scantron images are not available."""
	if not os.path.isdir(SCANTRON_DIR):
		pytest.skip("scantrons/ directory not found")


#============================================
class TestLoadImage:
	"""Tests for load_image function."""

	def test_loads_jpeg(self) -> None:
		"""Loads a JPEG image successfully."""
		_skip_if_no_scantrons()
		img = omr_utils.image_registration.load_image(BOTELLO)
		assert img is not None
		assert len(img.shape) == 3
		assert img.shape[2] == 3

	def test_missing_file_raises(self) -> None:
		"""Missing file raises FileNotFoundError."""
		with pytest.raises(FileNotFoundError):
			omr_utils.image_registration.load_image("/nonexistent/image.jpg")


#============================================
class TestFindPageContour:
	"""Tests for find_page_contour function."""

	def test_flatbed_scan(self) -> None:
		"""Flatbed scan returns 4 corners."""
		_skip_if_no_scantrons()
		img = omr_utils.image_registration.load_image(BOTELLO)
		corners = omr_utils.image_registration.find_page_contour(img)
		assert corners.shape == (4, 2)

	def test_phone_photo(self) -> None:
		"""Phone photo returns 4 corners."""
		_skip_if_no_scantrons()
		img = omr_utils.image_registration.load_image(KEY_SHEET)
		corners = omr_utils.image_registration.find_page_contour(img)
		assert corners.shape == (4, 2)


#============================================
class TestOrderCorners:
	"""Tests for order_corners function."""

	def test_already_ordered(self) -> None:
		"""Points already in correct order stay ordered."""
		pts = numpy.array([[0, 0], [100, 0], [100, 200], [0, 200]], dtype=numpy.float32)
		ordered = omr_utils.image_registration.order_corners(pts)
		# top-left should have smallest x+y
		assert ordered[0][0] + ordered[0][1] < ordered[2][0] + ordered[2][1]

	def test_shuffled_points(self) -> None:
		"""Shuffled points get correctly ordered."""
		pts = numpy.array([[100, 200], [0, 0], [0, 200], [100, 0]], dtype=numpy.float32)
		ordered = omr_utils.image_registration.order_corners(pts)
		# TL should be (0,0), TR should be (100,0)
		assert ordered[0][0] < ordered[1][0]
		assert ordered[0][1] < ordered[3][1]


#============================================
class TestRegisterImage:
	"""Tests for the full register_image pipeline."""

	def test_flatbed_produces_portrait(self) -> None:
		"""Flatbed scan registers to portrait canonical dimensions."""
		_skip_if_no_scantrons()
		img = omr_utils.image_registration.load_image(BOTELLO)
		registered = omr_utils.image_registration.register_image(img, 1700, 2200)
		assert registered.shape == (2200, 1700, 3)

	def test_phone_photo_produces_portrait(self) -> None:
		"""Phone photo registers to portrait canonical dimensions."""
		_skip_if_no_scantrons()
		img = omr_utils.image_registration.load_image(KEY_SHEET)
		registered = omr_utils.image_registration.register_image(img, 1700, 2200)
		assert registered.shape == (2200, 1700, 3)

	def test_all_images_register(self) -> None:
		"""All four test images register without error."""
		_skip_if_no_scantrons()
		paths = [BOTELLO, KEY_SHEET, STUDENT_43F, WOJTASZEK]
		for path in paths:
			if not os.path.isfile(path):
				continue
			img = omr_utils.image_registration.load_image(path)
			registered = omr_utils.image_registration.register_image(img, 1700, 2200)
			assert registered.shape == (2200, 1700, 3)


#============================================
class TestRotateImage90:
	"""Tests for rotate_image_90 function."""

	def test_90_degrees(self) -> None:
		"""90 degree rotation swaps dimensions."""
		img = numpy.zeros((100, 200), dtype=numpy.uint8)
		rotated = omr_utils.image_registration.rotate_image_90(img, 90)
		assert rotated.shape == (200, 100)

	def test_180_degrees(self) -> None:
		"""180 degree rotation preserves dimensions."""
		img = numpy.zeros((100, 200), dtype=numpy.uint8)
		rotated = omr_utils.image_registration.rotate_image_90(img, 180)
		assert rotated.shape == (100, 200)

	def test_0_degrees(self) -> None:
		"""0 degree rotation returns same image."""
		img = numpy.zeros((100, 200), dtype=numpy.uint8)
		rotated = omr_utils.image_registration.rotate_image_90(img, 0)
		assert rotated.shape == (100, 200)
