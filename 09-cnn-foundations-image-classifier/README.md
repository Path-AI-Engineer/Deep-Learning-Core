# CNN Vision Lab

An inspectable FashionMNIST product that connects convolution mechanics,
controlled CNN inference, intermediate activations and held-out evaluation in
one professional web laboratory.

Project 09 of **AI Engineer · Deep Learning Core**.

## Product

CNN Vision Lab makes the image-to-tensor pipeline observable:

- calculate two-dimensional cross-correlation and verify PyTorch parity;
- compare a compact MLP and CNN under one fair protocol;
- classify official test samples or memory-only PNG/JPEG uploads;
- inspect selected feature maps without causal claims;
- review accuracy, macro F1, per-class evidence, confusion and errors;
- load one immutable CPU bundle with reconstruction metadata and hashes.

The interface exposes six real routes:

```text
/              Overview and interactive 3D tensor topology
/classify      Controlled sample and upload inference
/convolution   Manual cross-correlation laboratory
/feature-maps  Whitelisted intermediate activations
/evaluation    Test metrics and MLP comparison
/about         Model card, architecture and limitations
```

Without a complete artifact, model-dependent endpoints return HTTP 503 and the
interface displays **Degraded mode**. The convolution lab remains usable because
it does not depend on trained weights.

See [architecture](docs/architecture.md), [contracts](docs/contracts.md) and
[engineering decisions](docs/decisions.md).

## Local setup

```powershell
Set-Location "C:\JeanLoa\Path-AI-Engineer\Deep-Learning-Core\09-cnn-foundations-image-classifier"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .

Set-Location frontend
$env:npm_config_offline = "false"
npm install
Set-Location ..
```

## Prepare, train and approve

```powershell
python scripts/prepare_data.py
python scripts/train_mlp_baseline.py
python scripts/train_cnn.py
```

The training commands print immutable run paths. Build the serving artifact:

```powershell
python scripts/build_model_bundle.py `
  --mlp-run artifacts/runs/mlp-baseline/<run-id> `
  --cnn-run artifacts/runs/cnn-base/<run-id> `
  --version v1.0.0
```

No numeric performance claim belongs here until those commands run against the
official data. See the [reproducibility contract](docs/reproducibility.md).

## Run the product

Terminal 1:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8009 --reload
```

Terminal 2:

```powershell
Set-Location frontend
npm run dev
```

Open `http://127.0.0.1:5179`. API documentation is at
`http://127.0.0.1:8009/docs`.

## Validation

```powershell
python -m pytest -q
python -m ruff check .
python -m mypy src backend scripts
python scripts/validate_project.py

Set-Location frontend
npm run typecheck
npm run build
```

With the approved assets present, run `python scripts/smoke_test.py`. The
production Dockerfile intentionally refuses to build without the official
gallery and versioned bundle.

## Boundary

This is not a general computer-vision platform. It has no online training,
database, authentication, remote persistence or arbitrary dataset ingestion.
Uploaded images are processed in memory and are not retained.

See the [model card](docs/model-card.md), [demo guide](docs/demo-guide.md) and
[closeout evidence](docs/closeout.md).

<!-- Legacy planning draft intentionally hidden from rendered documentation.

## 🧠 Descripción

Proyecto aplicado para construir una **CNN básica** orientada a clasificación de imágenes.

Este proyecto continúa la base del:

```txt
08-pytorch-regression-classification-api
```

pero ahora cambia el tipo de datos:

```txt
Antes:
datos tabulares

Ahora:
imágenes como tensores
```

Este proyecto pertenece al:

```txt
Plan 2 — Deep Learning Core
```

y forma parte del conjunto:

```txt
Núcleo de Redes Neuronales Profundas
```

La idea no es construir todavía un sistema avanzado de Computer Vision.

La idea es entender cómo una red neuronal puede aprender patrones visuales usando convoluciones, filtros, pooling y capas finales de clasificación.

---

## 🎯 Objetivo

Construir una CNN básica para clasificar imágenes, entrenarla, evaluarla y documentar sus resultados.

El objetivo técnico es entender cómo una imagen se convierte en tensor y cómo una CNN aprende patrones visuales como bordes, formas, texturas y combinaciones más complejas.

---

## 👤 Usuario objetivo

* AI Engineer en formación.
* Estudiante de Deep Learning.
* Persona que quiere entrar a Computer Vision.
* Reclutador técnico interesado en fundamentos de CNNs.
* Futuro constructor de modelos de visión, multimodales, robótica y sistemas de percepción.

---

## 🧱 Arquitectura esperada

```txt
Dataset de imágenes
      ↓
Preprocesamiento
      ↓
Image Tensors
      ↓
CNN Model
      ↓
Training Loop
      ↓
Evaluation
      ↓
Confusion Matrix
      ↓
Prediction Demo
      ↓
Reporte técnico
```

---

## 🔁 Flujo técnico

```txt
data/images
   ↓
image loading
   ↓
resize / normalization
   ↓
torch Dataset
   ↓
DataLoader
   ↓
convolution layers
   ↓
pooling
   ↓
flatten
   ↓
fully connected layers
   ↓
class prediction
   ↓
metrics
```

---

## 🧩 Módulos

### Módulo 1 — Image Tensor Preparation

Preparar imágenes para entrenamiento con PyTorch.

Incluye:

* Carga de imágenes.
* Redimensionamiento.
* Normalización.
* Conversión a tensores.
* Organización por clases.
* Uso de `Dataset` y `DataLoader`.

Pregunta central:

```txt
¿Cómo convierto imágenes en datos que una red puede procesar?
```

---

### Módulo 2 — CNN Architecture

Construir la arquitectura base de una CNN.

Incluye:

* Capas convolucionales.
* Kernels.
* Filtros.
* Activaciones.
* Pooling.
* Flatten.
* Capas fully connected.
* Salida por clases.

Pregunta central:

```txt
¿Cómo una CNN extrae patrones visuales de una imagen?
```

---

### Módulo 3 — Training and Evaluation

Entrenar y evaluar la CNN.

Incluye:

* Training loop.
* Evaluation loop.
* Loss de clasificación.
* Optimizer.
* Accuracy.
* Comparación train vs test.
* Detección básica de overfitting.

Pregunta central:

```txt
¿La CNN está aprendiendo patrones visuales o solo memorizando imágenes?
```

---

### Módulo 4 — Data Augmentation

Aplicar transformaciones básicas para mejorar generalización.

Incluye:

* Rotaciones simples.
* Flips.
* Crops.
* Cambios de escala.
* Normalización.
* Comparación con y sin augmentation.

Pregunta central:

```txt
¿Cómo ayudo al modelo a generalizar mejor con imágenes nuevas?
```

---

### Módulo 5 — Error Analysis

Analizar dónde falla la CNN.

Incluye:

* Clases con más errores.
* Predicciones correctas.
* Predicciones incorrectas.
* Confusion Matrix.
* Imágenes difíciles.
* Posibles causas del error.

Pregunta central:

```txt
¿Qué tipo de imágenes confunden más al modelo?
```

---

### Módulo 6 — Prediction Demo

Crear una demostración simple de predicción.

Incluye:

* Cargar una imagen nueva.
* Aplicar preprocessing.
* Ejecutar inferencia.
* Devolver clase predicha.
* Mostrar probabilidad o score si aplica.

Pregunta central:

```txt
¿Puedo usar la CNN entrenada para clasificar una imagen nueva?
```

---

## 🧪 Labs

### tec-labs

* `tec-image-tensor-lab`
* `tec-convolution-pooling-lab`
* `tec-cnn-training-lab`
* `tec-image-overfitting-lab`
* `tec-data-augmentation-lab`
* `tec-confusion-matrix-vision-lab`

---

## 📊 Métricas

Este proyecto usa métricas de clasificación:

* Accuracy.
* Precision si aplica.
* Recall si aplica.
* F1-score si aplica.
* Confusion Matrix.
* Train loss.
* Validation loss.
* Train accuracy.
* Validation accuracy.

---

## 📌 Próximos pasos

* Elegir dataset pequeño de imágenes.
* Organizar imágenes por clases.
* Cargar imágenes con PyTorch.
* Convertir imágenes a tensores.
* Crear `Dataset` y `DataLoader`.
* Diseñar CNN básica.
* Implementar training loop.
* Implementar evaluation loop.
* Calcular accuracy.
* Crear Confusion Matrix.
* Revisar errores por clase.
* Probar data augmentation básica.
* Guardar modelo entrenado.
* Crear demo de predicción con una imagen nueva.
* Documentar resultados.
* Grabar demo o explicación corta.
* Actualizar LinkedIn y CV.

---

## ✅ Entregable final

Al terminar este proyecto debe existir:

* Dataset de imágenes preparado.
* Pipeline de carga de imágenes.
* CNN básica entrenada.
* Training loop funcional.
* Evaluation loop funcional.
* Métricas de clasificación.
* Confusion Matrix.
* Análisis de errores.
* Data augmentation básica.
* Modelo guardado.
* Demo de predicción.
* README técnico.
* Labs documentados.
* Conclusión sobre límites de una CNN básica.

---

## 🧭 Regla final

```txt
Una imagen no entra al modelo como imagen.
Entra como tensor.

Una CNN no ve como humano.
Aprende filtros, patrones y representaciones visuales.
```

Este proyecto no busca dominar toda la visión computacional.

Busca construir la primera base sólida para entender cómo una red neuronal aprende desde imágenes.
