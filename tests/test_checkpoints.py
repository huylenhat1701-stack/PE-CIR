from __future__ import annotations

import unittest

import torch

from pic2word.checkpoints import convert_official_mapping_state


class OfficialCheckpointTests(unittest.TestCase):
    def test_converts_all_three_official_linear_layers(self) -> None:
        official = {
            "module.layers.0.0.weight": torch.zeros(512, 768),
            "module.layers.0.0.bias": torch.zeros(512),
            "module.layers.1.0.weight": torch.zeros(512, 512),
            "module.layers.1.0.bias": torch.zeros(512),
            "module.fc_out.weight": torch.zeros(768, 512),
            "module.fc_out.bias": torch.zeros(768),
        }

        converted = convert_official_mapping_state(official)

        self.assertEqual(
            set(converted),
            {
                "hidden.0.weight",
                "hidden.0.bias",
                "hidden.3.weight",
                "hidden.3.bias",
                "output.weight",
                "output.bias",
            },
        )

    def test_rejects_incomplete_official_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected official Mapping Network keys"):
            convert_official_mapping_state({})
