"""Tests for preprocessing: noise filtering and text cleaning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    _is_noisy_line,
    _strip_gutenberg_boilerplate,
    filter_noisy_lines,
    clean_text,
)


class TestIsNoisyLine:
    """Unit tests for the per-line noise heuristic."""

    def test_blank_line_not_noisy(self):
        assert not _is_noisy_line("")
        assert not _is_noisy_line("   ")

    def test_short_fragments_are_noisy(self):
        assert _is_noisy_line("f")
        assert _is_noisy_line("CK")
        assert _is_noisy_line("3")

    def test_low_alpha_ratio_is_noisy(self):
        assert _is_noisy_line("4 ; y")
        assert _is_noisy_line("~ : =")
        assert _is_noisy_line("_ 3 { }")

    def test_short_one_word_lines_are_noisy(self):
        assert _is_noisy_line("pesite")
        assert _is_noisy_line("Tica")
        assert _is_noisy_line("iter:")

    def test_short_two_word_lines_are_noisy(self):
        assert _is_noisy_line("ere ie")
        assert _is_noisy_line("Sy IFES")

    def test_ocr_fragment_clusters_are_noisy(self):
        assert _is_noisy_line("De ei ue rf")
        assert _is_noisy_line("i eh et See er")

    def test_high_tiny_word_ratio_is_noisy(self):
        assert _is_noisy_line("AE EGE SC AP i Gem er eae scat, owt 24")

    def test_toc_lines_are_noisy(self):
        assert _is_noisy_line("THE MONSTER'S WIFE ............ 108")

    def test_page_ref_lines_are_noisy(self):
        assert _is_noisy_line("01_559168 ffirs.qxd  11/30/04  11:38 PM  Page i")

    def test_normal_prose_survives(self):
        assert not _is_noisy_line("It was a dark, stormy night late in November.")
        assert not _is_noisy_line("Dr. Victor Frankenstein bent over the huge, lifeless figure.")
        assert not _is_noisy_line('"What is it?" he asked.')

    def test_chapter_headings_survive(self):
        assert not _is_noisy_line("The Creation")
        assert not _is_noisy_line("CHAPTER I")
        assert not _is_noisy_line("Frankenstein")

    def test_short_dialogue_survives(self):
        assert not _is_noisy_line("No! he cried.")
        assert not _is_noisy_line('"Yes," she said.')

    def test_reasonable_short_sentences_survive(self):
        assert not _is_noisy_line("He ran away.")
        assert not _is_noisy_line("The door opened.")


class TestFilterNoisyLines:
    """Integration tests for the full filter pipeline."""

    def test_removes_ocr_garbage(self):
        text = "f\nCK\ngood prose line here\n3\n"
        result = filter_noisy_lines(text)
        assert "good prose line here" in result
        assert "\nf\n" not in result
        assert "\nCK\n" not in result

    def test_strips_gutenberg_boilerplate(self):
        text = (
            "License info\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK ***\n"
            "Actual book content here.\n"
            "More real text.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK ***\n"
            "Donate to Gutenberg\n"
        )
        result = _strip_gutenberg_boilerplate(text)
        assert "Actual book content here." in result
        assert "More real text." in result
        assert "License info" not in result
        assert "Donate to Gutenberg" not in result

    def test_handles_multiple_gutenberg_books(self):
        text = (
            "*** START OF THE PROJECT GUTENBERG EBOOK ***\n"
            "Book one content.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK ***\n"
            "\n"
            "*** START OF THIS PROJECT GUTENBERG EBOOK ***\n"
            "Book two content.\n"
            "*** END OF THIS PROJECT GUTENBERG EBOOK ***\n"
        )
        result = _strip_gutenberg_boilerplate(text)
        assert "Book one content." in result
        assert "Book two content." in result

    def test_collapses_excessive_blank_lines(self):
        text = "A decent line here\n\n\n\n\nAnother decent line\n"
        result = filter_noisy_lines(text)
        assert "\n\n\n" not in result
        assert "A decent line here" in result
        assert "Another decent line" in result


class TestCleanTextWithFilter:
    """Test that clean_text integrates the noise filter."""

    def test_filter_noise_enabled_by_default(self):
        text = "f\nGood prose here.\nCK\n"
        result = clean_text(text, filter_noise=True)
        assert "Good prose here." in result
        # "f" and "CK" should be removed
        lines = result.strip().split("\n")
        assert all("Good" in l or not l.strip() for l in lines)

    def test_filter_noise_can_be_disabled(self):
        text = "f\nGood prose here.\nCK\n"
        result = clean_text(text, filter_noise=False)
        assert "f" in result
