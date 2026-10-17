"""Unit tests for Controlled Repair pipeline (B4 Generalization Test & Transition to B5)."""

import csv
import json
import logging
from pathlib import Path

import pytest

from scripts.controlled_repair_b4_to_b5 import (
    CounterfactualMetrics,
    clean_and_split_dataset,
    verify_scientific_go_gate,
)


def test_clean_and_split_dataset_filters_noise_and_ensures_zero_overlap(tmp_path: Path):
    manifest = tmp_path / "raw_pseudo_edits.csv"
    fieldnames = ["reference", "positive", "cf_a", "cf_b", "modification", "attribute_type"]

    rows = [
        # Valid tuple 1
        {"reference": "img1.jpg", "positive": "img2.jpg", "cf_a": "img3.jpg", "cf_b": "img4.jpg", "modification": "make red", "attribute_type": "color"},
        # Valid tuple 2
        {"reference": "img5.jpg", "positive": "img6.jpg", "cf_a": "img7.jpg", "cf_b": "img8.jpg", "modification": "add stripes", "attribute_type": "pattern"},
        # Valid tuple 3
        {"reference": "img9.jpg", "positive": "img10.jpg", "cf_a": "img11.jpg", "cf_b": "img12.jpg", "modification": "change to silk", "attribute_type": "material"},
        # Valid tuple 4
        {"reference": "img13.jpg", "positive": "img14.jpg", "cf_a": "img15.jpg", "cf_b": "img16.jpg", "modification": "make longer", "attribute_type": "size"},
        # Corrupt: reference == positive
        {"reference": "img17.jpg", "positive": "img17.jpg", "cf_a": "img18.jpg", "cf_b": "img19.jpg", "modification": "invalid", "attribute_type": "color"},
        # Corrupt: positive == cf_a
        {"reference": "img20.jpg", "positive": "img21.jpg", "cf_a": "img21.jpg", "cf_b": "img22.jpg", "modification": "invalid", "attribute_type": "color"},
        # Corrupt: empty field
        {"reference": "img23.jpg", "positive": "", "cf_a": "img24.jpg", "cf_b": "img25.jpg", "modification": "invalid", "attribute_type": "color"},
    ]

    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    clean_train, clean_val, stats = clean_and_split_dataset(
        manifest, tmp_path, val_ratio=0.25, seed=42, log=logging.getLogger()
    )

    assert stats["initial_samples"] == 7
    assert stats["valid_samples"] == 4
    assert stats["reference_overlap_count"] == 0
    assert stats["is_reference_disjoint"] is True

    assert clean_train.exists()
    assert clean_val.exists()

    # Read back and ensure zero image leakage
    def get_images(csv_path: Path):
        imgs = set()
        with csv_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                imgs.update([r["reference"], r["positive"], r["cf_a"], r["cf_b"]])
        return imgs

    train_imgs = get_images(clean_train)
    val_imgs = get_images(clean_val)
    assert len(train_imgs.intersection(val_imgs)) == 0


def test_verify_scientific_go_gate_passes_when_b4_exceeds_b1():
    b1_baseline = {
        "cf_a_win_percent": 80.04,
        "both_win_percent": 77.54,
        "hard_negative_error_percent": 22.46,
    }

    # B4 improves on all criteria
    b4_passing = CounterfactualMetrics(
        total_queries=1000,
        cf_a_win_percent=82.50,
        cf_b_win_percent=98.00,
        both_win_percent=80.10,
        hard_negative_error_percent=19.90,
    )

    report = verify_scientific_go_gate(b4_passing, b1_baseline, logging.getLogger())
    assert report["overall_passed"] is True
    assert "Approved to proceed to B5" in report["verdict"]


def test_verify_scientific_go_gate_fails_when_cf_a_drops():
    b1_baseline = {
        "cf_a_win_percent": 80.04,
        "both_win_percent": 77.54,
        "hard_negative_error_percent": 22.46,
    }

    # Historical failure case: CF-A drops to 77.98%
    b4_failing = CounterfactualMetrics(
        total_queries=1000,
        cf_a_win_percent=77.98,
        cf_b_win_percent=97.68,
        both_win_percent=76.58,
        hard_negative_error_percent=23.42,
    )

    report = verify_scientific_go_gate(b4_failing, b1_baseline, logging.getLogger())
    assert report["overall_passed"] is False
    assert report["criteria"]["cf_a_win"]["passed"] is False
    assert "Transition to B5 is BLOCKED" in report["verdict"]
