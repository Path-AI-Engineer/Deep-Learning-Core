from __future__ import annotations

from dataclasses import dataclass

import torch

from transformer_lab.models import SequenceTransformer
from transformer_lab.tokenization import BOS, EOS, PAD


@dataclass(frozen=True)
class DecodingResult:
    token_ids: tuple[int, ...]
    stopped_by: str
    steps: tuple[dict[str, object], ...]
    trace: dict[str, object] | None


def greedy_decode(
    model: SequenceTransformer,
    source_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    trace: bool = False,
) -> DecodingResult:
    if source_ids.ndim != 2 or source_ids.shape[0] != 1:
        raise ValueError("Greedy decoding supports one batch item at a time.")
    if not 1 <= max_new_tokens <= model.config.max_length - 1:
        raise ValueError("max_new_tokens exceeds the approved decoding contract.")
    model.eval()
    source_padding = source_ids.eq(PAD)
    generated = torch.tensor([[BOS]], dtype=torch.long, device=source_ids.device)
    steps: list[dict[str, object]] = []
    final_trace: dict[str, object] | None = None
    with torch.inference_mode():
        memory, encoder_weights = model.encode(
            source_ids, source_padding, trace=trace
        )
        for index in range(max_new_tokens):
            logits, decoder_trace = model.decode(
                generated,
                memory,
                target_padding_mask=generated.eq(PAD),
                source_padding_mask=source_padding,
                trace=trace,
            )
            scores = torch.softmax(logits[:, -1, :], dim=-1)
            top_values, top_ids = scores.topk(3, dim=-1)
            next_id = int(top_ids[0, 0])
            steps.append(
                {
                    "step": index + 1,
                    "selected_token_id": next_id,
                    "top_k": [
                        {"token_id": int(token_id), "probability": float(probability)}
                        for token_id, probability in zip(
                            top_ids[0], top_values[0], strict=True
                        )
                    ],
                }
            )
            generated = torch.cat(
                [
                    generated,
                    torch.tensor([[next_id]], device=generated.device),
                ],
                dim=1,
            )
            if trace:
                final_trace = {
                    "encoder_self": encoder_weights,
                    **decoder_trace,
                }
            if next_id == EOS:
                return DecodingResult(
                    tuple(int(value) for value in generated[0, 1:]),
                    "eos",
                    tuple(steps),
                    final_trace,
                )
    return DecodingResult(
        tuple(int(value) for value in generated[0, 1:]),
        "max_new_tokens",
        tuple(steps),
        final_trace,
    )

