# Transformer Architecture Lab

An inspectable encoder-decoder Transformer built from PyTorch primitives, trained on
controlled sequence-transduction tasks and delivered as a research-oriented web
instrument.

Project 12 closes the **Deep Learning Core** plan by connecting the mathematics of
attention to a real inference path. The product exposes scaled dot-product attention,
masking, positional encoding, encoder/decoder traces, greedy decoding, controlled
generalization evidence and the limitations attached to the approved model bundle.

## Product surface

- **Overview** — architecture, tasks and ID/OOD validation boundary.
- **Attention Math** — compute `softmax(QKᵀ / √dₖ + M)V` and inspect every matrix.
- **Masks & Positions** — causal visibility and sinusoidal position signals.
- **Architecture Trace** — bounded encoder self-, decoder self- and cross-attention.
- **Sequence Transduction** — real greedy inference on copy, reverse and recall.
- **Attention Explorer** — layer/head heatmaps with token axes and entropy.
- **Experiments** — disaggregated validation metrics, latency and error evidence.
- **Paper & Limits** — research questions, protocol, model card and threats.

The interface does not manufacture demo outcomes. Pages that depend on inference are
enabled only when the local, hash-verified Transformer bundle is available.

## Architecture

```text
React + TypeScript
        │
        ▼
FastAPI /api/v1
        │
        ├── model registry + bundle integrity
        ├── controlled data catalog
        ├── bounded inference and trace services
        └── evaluation/research evidence
                │
                ▼
Manual PyTorch Transformer
embeddings → positions → encoder → masked decoder → projection
```

The implementation deliberately avoids `nn.MultiheadAttention`, `nn.Transformer` and
the high-level encoder stack. Multi-head projection, scaled attention, masks,
educational LayerNorm, Pre/Post-LN blocks and the encoder-decoder path are local code.

See [Architecture](docs/architecture.md), [API contracts](docs/api-contract.md) and
[research protocol](docs/research-protocol.md).

## Controlled Sequence Transduction Suite

| Task | Source contract | Oracle target | Pressure |
|---|---|---|---|
| Copy | `COPY symbols EOS` | Same symbols + `EOS` | alignment and termination |
| Reverse | `REVERSE symbols EOS` | reversed symbols + `EOS` | order and position |
| Associative recall | `RECALL key value ... SEP query EOS` | queried value + `EOS` | content-addressed retrieval |

The fixed vocabulary reserves `PAD=0`, `BOS=1`, `EOS=2`, `SEP=3`, task tokens `4–6`
and 32 discrete symbols. Train, validation-ID and validation-OOD examples are
deterministic and checked for canonical-hash overlap.

## Local setup

Requirements: Python 3.12, Node.js 20+ and PowerShell.

```powershell
Set-Location "C:\JeanLoa\Path-AI-Engineer\Deep-Learning-Core\12-transformer-from-architecture-foundations-lab"

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

Set-Location frontend
npm install
Set-Location ..
```

Build the honest reference-validation bundle:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\backend"
python scripts\bootstrap_reference_bundle.py `
  --destination artifacts/models/transformer/v1.0.0-reference
python scripts\build_paper_assets.py
```

Run the integrated application:

```powershell
.\scripts\run.ps1 -Port 8012 -Reload
```

Open [http://127.0.0.1:8012](http://127.0.0.1:8012). During frontend development,
run `npm run dev` inside `frontend` and open
[http://127.0.0.1:5182](http://127.0.0.1:5182).

## Quality gate

```powershell
.\scripts\check.ps1
```

The gate runs Ruff, strict mypy, pytest, TypeScript checking and the production Vite
build. Tests cover math parity, masks, normalization, task oracles, deterministic
splits, target shifting, model shapes, artifact hashes, API behavior and the
no-high-level-Transformer architecture contract.

## Docker

The production image compiles React and serves the SPA and API from one Cloud Run
service:

```powershell
docker build -t transformer-architecture-lab:v1.0.0 .
docker run --rm -p 8080:8080 transformer-architecture-lab:v1.0.0
```

GCP planning is non-mutating unless `-Apply` is passed:

```powershell
.\infra\gcp\release.ps1 `
  -ProjectId "jeanloa-ai-engineer" `
  -ImageTag "v1.0.0"
```

The intended Cloud Run service is `ai-02-p12-transformer-architecture-lab`.

## Evidence status

The packaged quick reference run is intentionally labeled `reference_validation`.
It proves that the complete data → training → bundle → API → UI path is executable.
It does **not** replace the full multi-seed ablation matrix and it does not open the
frozen test split. Final claims must be produced only through the registered protocol.

## Scope

This is an educational and research artifact for small symbolic sequence tasks. It is
not a language model, does not claim natural-language understanding and treats
attention weights as descriptive tensors rather than causal explanations.
