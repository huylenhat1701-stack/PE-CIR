from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pic2word.data.cirr import load_cirr_split
from pic2word.evaluation.metrics import recall_at_k


class CIRRSplitTests(unittest.TestCase):
    def test_loads_official_annotation_shape_and_resolves_paths(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "captions").mkdir()
            (root / "image_splits").mkdir()
            (root / "img_raw" / "dev").mkdir(parents=True)
            image_split = {
                "reference": "./dev/reference.png",
                "target": "./dev/target.png",
                "other": "./dev/other.png",
            }
            captions = [
                {
                    "pairid": 7,
                    "reference": "reference",
                    "target_hard": "target",
                    "caption": "make it red",
                    "img_set": {
                        "members": ["reference", "target", "other"],
                        "reference_rank": 0,
                        "target_rank": 1,
                    },
                }
            ]
            (root / "image_splits" / "split.rc2.val.json").write_text(
                json.dumps(image_split), encoding="utf-8"
            )
            (root / "captions" / "cap.rc2.val.json").write_text(
                json.dumps(captions), encoding="utf-8"
            )
            (root / "img_raw" / "dev" / "reference.png").touch()

            dataset = load_cirr_split(root)

            self.assertEqual(dataset.queries[0].pair_id, 7)
            self.assertEqual(dataset.queries[0].target_id, "target")
            self.assertEqual(dataset.path_for("reference").name, "reference.png")
            self.assertEqual(len(dataset.missing_images()), 2)

    def test_missing_annotation_has_clear_error(self) -> None:
        with (
            TemporaryDirectory() as temporary_directory,
            self.assertRaisesRegex(FileNotFoundError, "annotation file not found"),
        ):
            load_cirr_split(temporary_directory)


class RecallTests(unittest.TestCase):
    def test_recall_at_k_returns_percentages(self) -> None:
        rankings = [["target-a", "x"], ["x", "target-b"]]
        result = recall_at_k(rankings, ["target-a", "target-b"], (1, 2))

        self.assertEqual(result, {1: 50.0, 2: 100.0})

    def test_recall_rejects_mismatched_query_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "counts must match"):
            recall_at_k([["target"]], [], (1,))


if __name__ == "__main__":
    unittest.main()
