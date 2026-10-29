"""Greedy and beam-search decoding for the from-scratch Transformer."""
import torch

from dataset import causal_mask


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def beam_search_decode(model, source, source_mask, tokenizer_tgt, max_len, device, beam_size=3, length_penalty=0.6):
    """Beam search over log-probabilities with GNMT length normalisation.

    Keeps `beam_size` partial hypotheses, expands each by its top-k next tokens,
    and returns the best finished (or longest-surviving) sequence as a 1-D tensor.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    encoder_output = model.encode(source, source_mask)

    start = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    beams = [(start, 0.0)]          # (tokens [1, t], summed log-prob)
    finished = []

    def score(tokens, logp):
        length = tokens.size(1)
        return logp / (((5 + length) / 6) ** length_penalty)

    for _ in range(max_len - 1):
        candidates = []
        for tokens, logp in beams:
            mask = causal_mask(tokens.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, tokens, mask)
            log_probs = torch.log_softmax(model.project(out[:, -1]), dim=-1).squeeze(0)
            top_lp, top_ix = log_probs.topk(beam_size)
            for lp, ix in zip(top_lp.tolist(), top_ix.tolist()):
                new = torch.cat([tokens, torch.tensor([[ix]], device=device).type_as(source)], dim=1)
                candidates.append((new, logp + lp))
        candidates.sort(key=lambda c: score(*c), reverse=True)
        beams = []
        for tokens, logp in candidates:
            if tokens[0, -1].item() == eos_idx:
                finished.append((tokens, logp))
            else:
                beams.append((tokens, logp))
            if len(beams) == beam_size:
                break
        if not beams or len(finished) >= beam_size:
            break

    pool = finished or beams
    best, _ = max(pool, key=lambda c: score(*c))
    return best.squeeze(0)
