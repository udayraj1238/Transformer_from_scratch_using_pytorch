"""CPU checks: shapes, causal mask, greedy and beam-search decoding."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Transformer_from_scratch_using_pytorch"))

from dataset import causal_mask  # noqa: E402
from decoding import beam_search_decode, greedy_decode  # noqa: E402
from model import build_transformer  # noqa: E402

VOCAB, SEQ = 50, 12
SOS, EOS = 1, 2


class Tok:
    def token_to_id(self, t):
        return {"[SOS]": SOS, "[EOS]": EOS, "[PAD]": 0}[t]


def tiny_model():
    torch.manual_seed(0)
    return build_transformer(VOCAB, VOCAB, SEQ, SEQ, d_model=32, N=2, h=4, d_ff=64).eval()


def test_causal_mask_is_lower_triangular():
    m = causal_mask(4)[0]
    assert torch.equal(m, torch.tril(torch.ones(4, 4)).bool())


def test_forward_shapes():
    model = tiny_model()
    src = torch.randint(3, VOCAB, (2, SEQ))
    src_mask = torch.ones(2, 1, 1, SEQ, dtype=torch.int)
    tgt = torch.randint(3, VOCAB, (2, SEQ))
    enc = model.encode(src, src_mask)
    dec = model.decode(enc, src_mask, tgt, causal_mask(SEQ))
    assert enc.shape == (2, SEQ, 32)
    assert model.project(dec).shape == (2, SEQ, VOCAB)


def test_beam_width_one_matches_greedy():
    model = tiny_model()
    src = torch.randint(3, VOCAB, (1, SEQ))
    mask = torch.ones(1, 1, 1, SEQ, dtype=torch.int)
    with torch.no_grad():
        g = greedy_decode(model, src, mask, Tok(), Tok(), SEQ, "cpu")
        b = beam_search_decode(model, src, mask, Tok(), SEQ, "cpu", beam_size=1)
    assert torch.equal(g, b)


def test_beam_search_returns_valid_sequence():
    model = tiny_model()
    src = torch.randint(3, VOCAB, (1, SEQ))
    mask = torch.ones(1, 1, 1, SEQ, dtype=torch.int)
    with torch.no_grad():
        out = beam_search_decode(model, src, mask, Tok(), SEQ, "cpu", beam_size=4)
    assert out[0].item() == SOS and 1 < out.numel() <= SEQ
