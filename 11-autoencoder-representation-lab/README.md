# Latent Representation Lab

An inspectable reconstruction and representation-learning laboratory built around
deterministic autoencoders. It reconstructs and denoises images, exposes a dedicated
two-dimensional latent space, interpolates between samples, compares model families
and makes the largest errors inspectable.

The official learning target is FashionMNIST. The committed `v1.0.0` release uses a
deterministic clothing-shaped fixture to validate the software without a download.
Every fixture result is labelled and must not be presented as a FashionMNIST benchmark.

## Product capabilities

- Mean-image and PCA baselines.
- Dense, convolutional, denoising and 2D-latent autoencoders.
- MSE, MAE, PSNR and SSIM reconstruction evidence.
- Frozen-embedding linear-probe evidence.
- Gaussian and masking corruption with clean targets.
- Reconstruction, upload inference, denoising, latent decoding and interpolation.
- Model comparison, highest-error review and hash-validated bundles.
- React laboratory delivered by a versioned FastAPI application.

Labels are excluded from autoencoder optimization. They are used only after training
for stratification, visual interpretation and linear-probe evaluation. Reconstruction
quality, probe performance and latent geometry remain separate evidence families.

## Architecture

```text
frontend/                 React + TypeScript laboratory
backend/app/              FastAPI delivery and bundle registry
src/autoencoder_lab/      Data, models, training, evaluation and inference
scripts/                  Reproducible pipeline and validation entry points
configs/                  Dataset, model, experiment and runtime contracts
artifacts/                Versioned fixture bundles and comparison evidence
tests/                    Unit, integration and API contract tests
docs/                     Architecture, model card and operating guidance
```

Training stays outside the request path. The frontend consumes `/api/v1` contracts and
does not contain model logic or embedded result data.

## Run locally

```powershell
Set-Location "C:\JeanLoa\Path-AI-Engineer\Deep-Learning-Core\11-autoencoder-representation-lab"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Set-Location frontend
npm install
npm run build
Set-Location ..

python scripts\bootstrap_fixture_bundles.py
$env:PYTHONPATH = "$PWD\src;$PWD\backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8011
```

Open `http://127.0.0.1:8011`; API documentation is at
`http://127.0.0.1:8011/docs`.

For Vite development, set `VITE_API_URL=http://127.0.0.1:8011/api/v1` and run
`npm run dev` from `frontend`.

## Official FashionMNIST workflow

```powershell
python scripts\download_data.py
python scripts\prepare_data.py
python scripts\train_model.py --model conv-ae
python scripts\train_model.py --model denoising-ae
python scripts\train_model.py --model latent-2d
python scripts\evaluate_models.py
```

Official-data outputs are generated separately from the committed fixture release.

## Quality gate

```powershell
python scripts\validate_project.py
```

## Cloud Run release package

```powershell
.\infra\gcp\release.ps1 -ProjectId "jeanloa-ai-engineer"
```

Without `-Apply`, the command only reports the intended resources. An approved
release builds the React/FastAPI image in Cloud Build, publishes it to
Artifact Registry `plan-02`, deploys
`ai-02-p11-latent-representation-lab` with a dedicated identity and verifies
`/api/v1/health`. The frontend lockfile and versioned fixture artifacts are
mandatory preflight inputs.

## Scope limits

This release does not implement variational autoencoders, GANs, diffusion models,
anomaly detection, t-SNE, UMAP, authentication, persistent user storage or
browser-triggered training. Uploads are processed in memory and are not retained.

## Documentation

- [Architecture](docs/architecture.md)
- [Data contract](docs/data-contract.md)
- [Model card](docs/model-card.md)
- [Evaluation protocol](docs/evaluation.md)
- [API reference](docs/api.md)
- [Operations](docs/operations.md)
- [Limitations](docs/limitations.md)

## License

MIT. See [LICENSE](LICENSE).

<details>
<summary>Original academic brief (superseded by the implemented scope)</summary>

## Original description

Lab técnico para construir **autoencoders** y entender representación, compresión y reconstrucción de datos.

Este proyecto continúa la base del:

```txt id="hg8fpq"
10-sequence-models-rnn-lstm-lab
```

pero cambia el enfoque:

```txt id="ud90bk"
Antes:
modelos que procesan secuencias

Ahora:
modelos que aprenden representaciones internas
```

Este proyecto pertenece al:

```txt id="mb9kwf"
Plan 2 — Deep Learning Core
```

y forma parte del conjunto:

```txt id="rwxa5b"
Núcleo de Redes Neuronales Profundas
```

La idea no es solo reconstruir datos.

La idea es entender cómo una red puede aprender una versión comprimida y útil de la información.

---

## 🎯 Objetivo

Construir autoencoders básicos para entender cómo una red aprende representaciones internas mediante un proceso de compresión y reconstrucción.

El objetivo técnico es comprender:

* Encoder.
* Latent space.
* Decoder.
* Reconstruction loss.
* Compresión.
* Reconstrucción.
* Representación aprendida.
* Uso básico para reducción dimensional o detección de anomalías.

---

## 👤 Usuario objetivo

* AI Engineer en formación.
* Estudiante de Deep Learning.
* Persona que quiere entender representation learning.
* Futuro constructor de modelos generativos, embeddings, diffusion models, multimodal AI y anomaly detection.
* Reclutador técnico interesado en fundamentos de Deep Learning.

---

## 🧱 Arquitectura esperada

```txt id="49k0ft"
Input data
      ↓
Encoder
      ↓
Latent representation
      ↓
Decoder
      ↓
Reconstruction
      ↓
Reconstruction loss
      ↓
Evaluation
      ↓
Representation analysis
```

---

## 🔁 Flujo técnico

```txt id="84ej8q"
data
   ↓
preprocessing
   ↓
tensor preparation
   ↓
encoder
   ↓
latent space
   ↓
decoder
   ↓
reconstructed output
   ↓
loss(input, reconstruction)
   ↓
training loop
   ↓
analysis
```

---

## 🧩 Módulos

### Módulo 1 — Basic Autoencoder

Construir un autoencoder simple.

Incluye:

* Encoder.
* Bottleneck.
* Decoder.
* Reconstruction output.
* Training loop.
* Reconstruction loss.

Pregunta central:

```txt id="sz842k"
¿Cómo una red aprende a reconstruir su propia entrada?
```

---

### Módulo 2 — Latent Representation

Analizar la representación interna aprendida.

Incluye:

* Latent vector.
* Dimensión reducida.
* Bottleneck.
* Visualización si aplica.
* Comparación input vs latent representation.

Pregunta central:

```txt id="r9zn8x"
¿Qué información conserva la red cuando comprime los datos?
```

---

### Módulo 3 — Reconstruction Evaluation

Evaluar la calidad de reconstrucción.

Incluye:

* Reconstruction loss.
* Comparación input vs output.
* Errores de reconstrucción.
* Casos bien reconstruidos.
* Casos mal reconstruidos.

Pregunta central:

```txt id="kupd3b"
¿Qué tan bien reconstruye el modelo y dónde falla?
```

---

### Módulo 4 — Denoising Concept

Explorar el concepto de autoencoder denoising.

Incluye:

* Ruido artificial.
* Entrada con ruido.
* Reconstrucción limpia.
* Comparación original / noisy / reconstructed.
* Limitaciones.

Pregunta central:

```txt id="csn1be"
¿Puede una red aprender a recuperar información útil desde datos ruidosos?
```

---

### Módulo 5 — Anomaly Detection with Reconstruction

Usar error de reconstrucción para detectar anomalías.

Incluye:

* Error alto como señal de rareza.
* Casos normales.
* Casos anómalos.
* Umbral simple.
* Limitaciones de este enfoque.

Pregunta central:

```txt id="6l47f6"
¿Puedo usar reconstrucción pobre como señal de anomalía?
```

---

### Módulo 6 — Representation Visualization

Visualizar o documentar la representación aprendida.

Incluye:

* Latent space.
* Comparación de muestras.
* Gráficos si aplica.
* Interpretación conceptual.
* Conexión con embeddings.

Pregunta central:

```txt id="cjvoyz"
¿Cómo se conecta un autoencoder con la idea de embeddings y representación?
```

---

## 🧪 Labs

### tec-labs

* `tec-basic-autoencoder-lab`
* `tec-latent-space-lab`
* `tec-reconstruction-loss-lab`
* `tec-denoising-autoencoder-concept-lab`
* `tec-autoencoder-anomaly-lab`
* `tec-representation-visualization-lab`

---

## 📊 Métricas / señales de aprendizaje

Este proyecto se evalúa principalmente por reconstrucción y comprensión de representación.

Métricas posibles:

* Reconstruction loss.
* MSE.
* MAE.
* Error de reconstrucción por muestra.
* Comparación input vs output.
* Error normal vs error anómalo.

Señales de aprendizaje:

* El autoencoder reconstruye entradas.
* El latent space tiene menor dimensión.
* El bottleneck obliga a comprimir.
* El error de reconstrucción se interpreta correctamente.
* Se entiende que reconstruir no es lo mismo que clasificar.
* Se conecta representación interna con embeddings y modelos generativos.

---

## 📌 Próximos pasos

* Elegir dataset simple.
* Preparar datos como tensores.
* Definir encoder.
* Definir decoder.
* Crear autoencoder básico.
* Crear training loop.
* Calcular reconstruction loss.
* Comparar input vs output.
* Analizar latent representation.
* Probar ruido artificial si aplica.
* Explorar detección de anomalías con error de reconstrucción.
* Visualizar o documentar latent space.
* Escribir conclusiones técnicas.
* Grabar demo o explicación corta.
* Actualizar LinkedIn y CV.

---

## ✅ Entregable final

Al terminar este proyecto debe existir:

* Autoencoder básico entrenado.
* Encoder y decoder definidos.
* Training loop funcional.
* Reconstruction loss registrada.
* Comparación input vs reconstruction.
* Análisis de latent representation.
* Experimento denoising conceptual.
* Experimento básico de anomalías si aplica.
* README técnico.
* Labs documentados.
* Conclusión sobre representación aprendida.

---

## 🧭 Regla final

```txt id="cdv56c"
Un autoencoder no aprende una etiqueta.
Aprende a comprimir y reconstruir.

La representación interna es el verdadero aprendizaje.
```

Este proyecto no busca crear un modelo generativo avanzado todavía.

Busca entender cómo una red puede aprender representaciones internas útiles.

</details>
