from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from cre_eval.checkpoint import load_cre_checkpoint


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4))


def test_sparse_checkpoint_loader(tmp_path):
    checkpoint = tmp_path / "tiny.safetensors"
    save_file(
        {
            "t5.weight.indices": torch.tensor([1, 3], dtype=torch.int32),
            "t5.weight.values": torch.tensor([2.0, -1.0]),
            "clip.weight.indices": torch.tensor([0], dtype=torch.int32),
            "clip.weight.values": torch.tensor([0.5]),
        },
        str(checkpoint),
        metadata={"format": "rce_v3_sparse_parameter_delta", "components": '["clip", "t5"]'},
    )
    pipe = SimpleNamespace(text_encoder_2=Tiny(), text_encoder=Tiny())
    loaded = load_cre_checkpoint(pipe, checkpoint)
    assert loaded == {"t5": ["weight"], "clip": ["weight"]}
    assert torch.equal(pipe.text_encoder_2.weight, torch.tensor([0.0, 2.0, 0.0, -1.0]))
    assert torch.equal(pipe.text_encoder.weight, torch.tensor([0.5, 0.0, 0.0, 0.0]))

