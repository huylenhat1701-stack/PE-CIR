from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from pic2word.retrieval.index import CandidateIndex


class CandidateIndexTests(unittest.TestCase):
    def test_search_ranks_by_cosine_and_can_exclude_query(self) -> None:
        paths = [Path("first.jpg"), Path("second.jpg"), Path("third.jpg")]
        features = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.8, 0.2, 0.0],
            ]
        )
        index = CandidateIndex(paths, features)
        results = index.search(
            torch.tensor([1.0, 0.0, 0.0]),
            top_k=2,
            exclude_paths={paths[0]},
        )

        self.assertEqual([result.path.name for result in results], ["third.jpg", "second.jpg"])
        self.assertGreater(results[0].score, results[1].score)

    def test_index_save_and_load(self) -> None:
        index = CandidateIndex([Path("image.jpg")], torch.tensor([[3.0, 4.0]]))
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "index.pt"
            index.save(path)
            restored = CandidateIndex.load(path)

        self.assertEqual(restored.paths, index.paths)
        self.assertTrue(torch.allclose(restored.features, index.features))

    def test_search_can_be_restricted_to_cirr_group(self) -> None:
        paths = [Path("reference.jpg"), Path("global-best.jpg"), Path("group-target.jpg")]
        index = CandidateIndex(
            paths,
            torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.8, 0.2]]),
        )

        results = index.search(
            torch.tensor([1.0, 0.0]),
            top_k=3,
            exclude_paths={paths[0]},
            allowed_paths={paths[0], paths[2]},
        )

        self.assertEqual([result.path.name for result in results], ["group-target.jpg"])


if __name__ == "__main__":
    unittest.main()
