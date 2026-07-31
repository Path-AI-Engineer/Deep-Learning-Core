from __future__ import annotations

from _common import PROJECT_ROOT, configure_imports

configure_imports()

from torchvision.datasets import FashionMNIST  # noqa: E402


def main() -> None:
    root = PROJECT_ROOT / "data" / "raw"
    FashionMNIST(root=root, train=True, download=True)
    FashionMNIST(root=root, train=False, download=True)
    print(f"FashionMNIST downloaded to {root}")


if __name__ == "__main__":
    main()
