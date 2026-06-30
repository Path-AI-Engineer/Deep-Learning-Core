# 10-sequence-models-rnn-lstm-lab

## 🧠 Descripción

Lab técnico para entender modelos secuenciales como **RNN, LSTM y GRU**.

Este proyecto continúa la base del:

```txt
09-cnn-foundations-image-classifier
```

pero cambia el tipo de problema:

```txt
Antes:
imágenes como tensores

Ahora:
datos ordenados en secuencia
```

Este proyecto pertenece al:

```txt
Plan 2 — Deep Learning Core
```

y forma parte del conjunto:

```txt
Núcleo de Redes Neuronales Profundas
```

La idea no es construir todavía un modelo de lenguaje avanzado.

La idea es entender cómo una red puede procesar información paso a paso en el tiempo o en una secuencia.

---

## 🎯 Objetivo

Construir y comparar modelos secuenciales básicos usando RNN, LSTM y GRU.

El objetivo técnico es entender cómo una red procesa datos donde el orden importa, como texto, ventas temporales, sensores, eventos o secuencias de comportamiento.

---

## 👤 Usuario objetivo

* AI Engineer en formación.
* Estudiante de Deep Learning.
* Persona que quiere entender modelos secuenciales.
* Futuro constructor de LLMs, Transformers, agentes, forecasting avanzado y sistemas de robótica.
* Reclutador técnico interesado en fundamentos de redes secuenciales.

---

## 🧱 Arquitectura esperada

```txt
Dataset secuencial
      ↓
Preprocesamiento de secuencia
      ↓
Sequence Dataset
      ↓
RNN / LSTM / GRU
      ↓
Training Loop
      ↓
Evaluation
      ↓
Comparación de modelos
      ↓
Reporte técnico
```

---

## 🔁 Flujo técnico

```txt
sequence data
   ↓
tokens / timesteps
   ↓
padding si aplica
   ↓
Dataset / DataLoader
   ↓
RNN baseline
   ↓
LSTM model
   ↓
GRU model
   ↓
metrics
   ↓
comparison notes
```

---

## 🧩 Módulos

### Módulo 1 — Sequence Dataset

Preparar datos secuenciales para entrenamiento.

Incluye:

* Secuencias.
* Timesteps.
* Features por paso.
* Padding básico si aplica.
* Batches de secuencias.
* Separación train/test.

Pregunta central:

```txt
¿Cómo preparo datos donde el orden importa?
```

---

### Módulo 2 — RNN Baseline

Construir una RNN simple como primera referencia.

Incluye:

* Hidden state.
* Procesamiento paso a paso.
* Salida final.
* Predicción secuencial o clasificación de secuencia.
* Limitaciones de una RNN básica.

Pregunta central:

```txt
¿Cómo una RNN recuerda información de pasos anteriores?
```

---

### Módulo 3 — LSTM Model

Construir un modelo LSTM para manejar mejor memoria secuencial.

Incluye:

* Cell state.
* Hidden state.
* Gates.
* Memoria a largo plazo.
* Comparación contra RNN.
* Problema de vanishing gradients.

Pregunta central:

```txt
¿Por qué LSTM puede recordar mejor que una RNN simple?
```

---

### Módulo 4 — GRU Model

Construir un modelo GRU como alternativa más compacta.

Incluye:

* Update gate.
* Reset gate.
* Hidden state.
* Comparación contra LSTM.
* Ventajas y limitaciones.

Pregunta central:

```txt
¿Cuándo una GRU puede ser suficiente frente a una LSTM?
```

---

### Módulo 5 — Sequence Evaluation

Evaluar modelos secuenciales.

Incluye:

* Loss.
* Accuracy si es clasificación.
* MAE/RMSE si es predicción numérica.
* Comparación train vs test.
* Errores por longitud de secuencia.
* Overfitting en secuencias.

Pregunta central:

```txt
¿Cómo sé si el modelo entiende la secuencia o solo memorizó patrones?
```

---

### Módulo 6 — Comparison Notes

Comparar RNN, LSTM y GRU con criterio.

Incluye:

* Simplicidad.
* Capacidad de memoria.
* Costo computacional.
* Rendimiento.
* Interpretación.
* Limitaciones frente a Transformers.

Pregunta central:

```txt
¿Qué modelo secuencial elegiría y por qué?
```

---

## 🧪 Labs

### tec-labs

* `tec-sequence-dataset-lab`
* `tec-rnn-hidden-state-lab`
* `tec-lstm-memory-lab`
* `tec-gru-comparison-lab`
* `tec-sequence-evaluation-lab`
* `tec-sequence-model-limitations-lab`

---

## 📊 Métricas

Dependiendo del tipo de problema:

### Clasificación secuencial

* Accuracy.
* Precision.
* Recall.
* F1-score.
* Confusion Matrix.
* Train loss.
* Validation loss.

### Predicción numérica secuencial

* MAE.
* RMSE.
* MSE.
* Train loss.
* Validation loss.

### Señales de comparación

* RNN vs LSTM.
* LSTM vs GRU.
* Error por longitud de secuencia.
* Diferencia entre train y test.
* Tiempo de entrenamiento si aplica.

---

## 📌 Próximos pasos

* Elegir un dataset secuencial pequeño.
* Definir si el problema será clasificación o predicción.
* Preparar secuencias.
* Crear `Dataset` y `DataLoader`.
* Implementar RNN simple.
* Implementar LSTM.
* Implementar GRU.
* Crear training loop.
* Crear evaluation loop.
* Calcular métricas.
* Comparar resultados.
* Analizar limitaciones.
* Documentar diferencias entre RNN, LSTM y GRU.
* Explicar por qué Transformers superan varias limitaciones.
* Grabar demo o explicación corta.
* Actualizar LinkedIn y CV.

---

## ✅ Entregable final

Al terminar este proyecto debe existir:

* Dataset secuencial preparado.
* RNN básica entrenada.
* LSTM entrenada.
* GRU entrenada.
* Training loop funcional.
* Evaluation loop funcional.
* Métricas registradas.
* Tabla comparativa.
* Análisis de limitaciones.
* README técnico.
* Labs documentados.
* Nota clara sobre la transición hacia Transformers.

---

## 🧭 Regla final

```txt
Una secuencia no es solo una lista de datos.
El orden cambia el significado.

RNN, LSTM y GRU enseñan cómo una red intenta recordar.
Transformers enseñarán otra forma más potente de usar contexto.
```

Este proyecto no busca construir un LLM.

Busca entender la base histórica y conceptual de los modelos que procesan secuencias.
