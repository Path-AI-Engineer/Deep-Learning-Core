# Neural Network Foundations Lab

> Plan 2 · Deep Learning Core · Global days 190–210

An inspectable neural-network engine built from first principles with NumPy.
The project owns the mathematical engine, validation evidence and serializable
trace contract. Its standalone **Neural Network Foundations Lab** interface
lives in `frontend/app.py` and directly consumes this repository's public
Python package; it does not depend on another project or repository.

The current map supersedes older scope notes in this document. Closure requires
finite-difference gradient checks, deterministic XOR learning, checkpoint
restoration, PyTorch parity and a working visual integration—not merely a
decreasing loss.

### Delivery milestones

| Milestone | Global days | Status |
|---|---:|---|
| Foundations and trace contract | 190–196 | Completed |
| Backpropagation and XOR training | 197–203 | Completed |
| Stability and PyTorch parity | 204–207 | Completed |
| Standalone visual interface | 208–210 | Completed |

Architecture and contract references:
[learning contract](docs/learning-contract.md) ·
[interface contract](docs/interface-contract.md) ·
[decisions](docs/decisions.md).

### Run locally

```powershell
Set-Location "C:\JeanLoa\Path-AI-Engineer\Deep-Learning-Core\07-neural-network-foundations-lab"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path "src").Path
python -m pytest -q
python scripts\validate_project.py
python scripts\run_experiment.py configs\experiments\xor.yaml
python -m streamlit run frontend\app.py
```

Generated evidence includes a full trace, gradient-check report, reproducible
checkpoint, NumPy/PyTorch parity report and SVG decision boundaries. See the
[demo guide](docs/demo-guide.md) and [delivery evidence](docs/worklog.md).

Open `http://localhost:8501` to configure a deterministic experiment, apply a
single SGD update or bounded training run, inspect a per-neuron forward/backward
trace, compare decision boundaries and execute NumPy/PyTorch parity.

### Cloud Run release package

The non-root `Dockerfile` exposes the Streamlit health contract on the Cloud Run
`PORT`. Review the deployment plan without changing GCP:

```powershell
.\infra\gcp\release.ps1 -ProjectId "jeanloa-ai-engineer"
```

Use `-Apply` only for an intentional public release. Cloud Build creates the
versioned image in `plan-02`; Cloud Run uses the semantic service name
`ai-02-p07-neural-foundations-lab`, a dedicated service account, scale-to-zero
limits and a `/_stcore/health` smoke check.

<details>
<summary>Historical initial scope</summary>

The following notes are retained as the original learning brief. The status,
commands, architecture and acceptance evidence above are authoritative.

## 🧠 Initial description

Lab técnico para entender los fundamentos internos de una red neuronal y conectarlos con una implementación profesional usando PyTorch.

Este proyecto inicia el:

```txt
Plan 2 — Deep Learning Core
```

y forma parte del conjunto:

```txt
Núcleo de Redes Neuronales Profundas
```

La idea no es crear una red neuronal “de juguete” ni quedarse en una implementación académica.

La idea es entender primero la mecánica interna:

```txt
entrada → pesos → predicción → pérdida → gradiente → actualización
```

y luego replicar ese flujo con PyTorch para comprender qué hace el framework por dentro.

---

## 🎯 Objetivo

Construir una base firme de redes neuronales entendiendo:

* Forward pass.
* Pesos y bias.
* Funciones de activación.
* Predicción.
* Función de pérdida.
* Gradiente.
* Gradient descent.
* Training loop.
* Evaluación.
* Implementación equivalente con PyTorch.

El objetivo técnico es dejar de ver una red neuronal como una caja negra.

---

## 👤 Usuario objetivo

* AI Engineer en formación.
* Estudiante de Deep Learning.
* Persona que quiere entender redes neuronales antes de usar modelos grandes.
* Reclutador técnico interesado en fundamentos.
* Futuro constructor de CNNs, Transformers, Diffusion Models, modelos RL y sistemas multimodales.

---

## 🧱 Arquitectura esperada

```txt
Dataset simple
      ↓
Implementación manual mínima
      ↓
Forward pass
      ↓
Loss
      ↓
Gradient descent
      ↓
Training loop
      ↓
Visualización de pérdida
      ↓
Implementación equivalente con PyTorch
      ↓
Comparación manual vs framework
      ↓
Conclusión técnica
```

---

## 🔁 Flujo técnico

```txt
input X
   ↓
weights + bias
   ↓
linear combination
   ↓
activation function
   ↓
y_pred
   ↓
loss(y_true, y_pred)
   ↓
gradient
   ↓
weight update
   ↓
repeat
   ↓
PyTorch equivalent
```

---

## 🧩 Módulos

### Módulo 1 — Neural Network Mechanics

Entender las piezas internas de una red neuronal.

Incluye:

* Entrada.
* Pesos.
* Bias.
* Activación.
* Salida.
* Predicción.

Pregunta central:

```txt
¿Qué ocurre dentro de una red antes de que aprenda?
```

---

### Módulo 2 — Manual Forward Pass

Implementar manualmente el paso de predicción.

Incluye:

* Combinación lineal.
* Activación.
* Predicción inicial.
* Comparación con el valor real.

Pregunta central:

```txt
¿Cómo una red convierte una entrada en una predicción?
```

---

### Módulo 3 — Loss and Error

Calcular qué tan equivocada está la red.

Incluye:

* Error.
* MSE.
* Binary Cross Entropy si aplica.
* Interpretación de la pérdida.
* Comparación entre `y_true` y `y_pred`.

Pregunta central:

```txt
¿Cómo sabe la red que se está equivocando?
```

---

### Módulo 4 — Gradient Descent

Actualizar pesos para reducir el error.

Incluye:

* Gradiente.
* Learning rate.
* Dirección de ajuste.
* Actualización de pesos.
* Relación entre error y aprendizaje.

Pregunta central:

```txt
¿Cómo cambia la red para equivocarse menos?
```

---

### Módulo 5 — Training Loop

Construir el ciclo completo de entrenamiento.

Incluye:

* Forward pass.
* Loss.
* Backward step conceptual.
* Update.
* Epochs.
* Historial de pérdida.

Pregunta central:

```txt
¿Qué ocurre dentro de cada epoch de entrenamiento?
```

---

### Módulo 6 — PyTorch Equivalent

Replicar el mismo flujo usando PyTorch.

Incluye:

* `torch.Tensor`.
* `nn.Module`.
* Loss function.
* Optimizer.
* Training loop.
* Evaluation loop.

Pregunta central:

```txt
¿Qué automatiza PyTorch y qué debo seguir entendiendo yo?
```

---

### Módulo 7 — Manual vs PyTorch Comparison

Comparar la implementación manual con la versión profesional.

Incluye:

* Diferencias de código.
* Diferencias de robustez.
* Qué hace PyTorch internamente.
* Qué partes ya no conviene implementar a mano.
* Qué conocimiento sí debo conservar.

Pregunta central:

```txt
¿Por qué construir una versión manual me ayuda a usar PyTorch mejor?
```

---

## 🧪 Labs

### tec-labs

* `tec-neural-network-mechanics-lab`
* `tec-forward-pass-lab`
* `tec-loss-function-lab`
* `tec-gradient-descent-lab`
* `tec-training-loop-lab`
* `tec-pytorch-equivalent-lab`
* `tec-manual-vs-pytorch-comparison-lab`

---

## 📊 Métricas / señales de aprendizaje

Este proyecto no se mide solo por accuracy o error final.

Se mide por comprensión del proceso.

Señales principales:

* La red produce predicciones.
* La función de pérdida se calcula correctamente.
* La pérdida baja durante el entrenamiento.
* Los pesos cambian durante el entrenamiento.
* El learning rate afecta el aprendizaje.
* El training loop se entiende paso a paso.
* La versión PyTorch replica el flujo manual.
* El README explica el proceso sin magia.

Métricas posibles:

* MSE.
* Binary Cross Entropy si aplica.
* Loss por epoch.
* Error antes vs después del entrenamiento.
* Comparación manual vs PyTorch.

---

## 📌 Próximos pasos

* Definir un dataset pequeño y simple.
* Crear entradas `X` y target `y`.
* Inicializar pesos y bias.
* Implementar forward pass manual.
* Implementar función de pérdida.
* Calcular gradientes simples.
* Actualizar pesos con gradient descent.
* Crear training loop.
* Guardar historial de pérdida.
* Graficar loss.
* Probar distintos learning rates.
* Replicar el flujo con PyTorch.
* Comparar implementación manual vs PyTorch.
* Documentar qué hace el framework por dentro.
* Documentar limitaciones.
* Grabar demo o explicación corta.
* Actualizar LinkedIn y CV.

---

## ✅ Entregable final

Al terminar este proyecto debe existir:

* Implementación manual mínima de una red neuronal.
* Forward pass funcionando.
* Función de pérdida.
* Gradient descent básico.
* Training loop.
* Gráfico de pérdida.
* Comparación antes/después del entrenamiento.
* Implementación equivalente con PyTorch.
* Comparación manual vs PyTorch.
* README técnico.
* Labs documentados.
* Explicación clara de cómo aprende una red.
* Nota sobre por qué se usan frameworks profesionales.

---

## 🧭 Regla final

```txt
Primero entiendo la mecánica interna.
Luego uso PyTorch.
No uso frameworks para esconder lo que no entiendo.
Los uso para construir mejor.
```

Este proyecto no busca demostrar que puedo hacer una red neuronal enorme.

Busca demostrar que entiendo el núcleo del aprendizaje profundo.

</details>
