from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from pic2word.retrieval import CandidateIndex, search_cfpe


@pytest.mark.parametrize('verifier', [False, True])
def test_full_gallery_retains_group_members_outside_shortlist(tmp_path, verifier):
    reference = tmp_path / 'reference.png'
    Image.new('RGB', (2, 2)).save(reference)
    paths = [reference] + [tmp_path / f'{i}.png' for i in range(4)]
    index = CandidateIndex(paths, torch.tensor([[1., 0.], [.99, .1], [.9, .3], [.6, .8], [.1, .99]]))
    backbone = SimpleNamespace(device='cpu', preprocess=lambda im: torch.zeros(3, 2, 2),
                               encode_image=lambda *a, **k: torch.tensor([[1., 0.]]),
                               encode_text=lambda *a, **k: torch.tensor([[1., 0.]]))

    class Model:
        uses_verifier = verifier

        def __call__(self, *args):
            return SimpleNamespace(query=torch.tensor([[1., 0.]]))

        def verify(self, candidates, *args):
            assert candidates.shape[1] == 2
            return SimpleNamespace(preserve=torch.zeros(1, 2), edit=torch.zeros(1, 2), violation=torch.zeros(1, 2))

        def rerank_scores(self, scores, *args, **kwargs):
            return -scores

    results = search_cfpe(Model(), backbone, index, reference, 'red', top_k=2, return_k=5)
    names = [r.path.name for r in results]
    assert names == (['1.png', '0.png', '2.png', '3.png'] if verifier else ['0.png', '1.png', '2.png', '3.png'])
    assert [r.path.name for r in results if r.path in {paths[3], paths[4]}] == ['2.png', '3.png']
