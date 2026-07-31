from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_high_level_transformer_modules_are_not_used() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src").rglob("*.py")
    )
    forbidden = ("nn.MultiheadAttention", "nn.Transformer(", "nn.TransformerEncoder")
    for symbol in forbidden:
        assert symbol not in source


def test_required_product_surfaces_exist() -> None:
    expected = [
        "frontend/src/pages/AttentionMathPage.tsx",
        "frontend/src/pages/MasksPositionsPage.tsx",
        "frontend/src/pages/TracePage.tsx",
        "frontend/src/pages/TransductionPage.tsx",
        "frontend/src/pages/ExperimentsPage.tsx",
        "frontend/src/pages/PaperPage.tsx",
        "backend/app/api/v1/router.py",
    ]
    assert all((ROOT / path).is_file() for path in expected)
