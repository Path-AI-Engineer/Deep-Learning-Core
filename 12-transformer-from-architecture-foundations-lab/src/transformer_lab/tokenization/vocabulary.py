from __future__ import annotations

from dataclasses import dataclass

PAD, BOS, EOS, SEP, COPY, REVERSE, RECALL = range(7)
SPECIAL_TOKENS = ("PAD", "BOS", "EOS", "SEP", "COPY", "REVERSE", "RECALL")
SYMBOLS = tuple(f"SYMBOL_{index:02d}" for index in range(32))
TOKENS = SPECIAL_TOKENS + SYMBOLS


@dataclass(frozen=True)
class Vocabulary:
    version: str = "1.0.0"

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(TOKENS)}

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for index, token in enumerate(TOKENS)}

    def encode(self, tokens: list[str] | tuple[str, ...]) -> list[int]:
        mapping = self.token_to_id
        unknown = [token for token in tokens if token not in mapping]
        if unknown:
            raise ValueError(f"Unknown tokens: {sorted(set(unknown))}.")
        return [mapping[token] for token in tokens]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> list[str]:
        mapping = self.id_to_token
        unknown = [token_id for token_id in token_ids if token_id not in mapping]
        if unknown:
            raise ValueError(f"Unknown token IDs: {sorted(set(unknown))}.")
        return [mapping[token_id] for token_id in token_ids]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "reserved": {token: index for index, token in enumerate(SPECIAL_TOKENS)},
            "symbols": {
                token: index + len(SPECIAL_TOKENS)
                for index, token in enumerate(SYMBOLS)
            },
        }
