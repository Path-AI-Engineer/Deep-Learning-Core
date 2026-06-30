# 08-pytorch-regression-classification-api

## 🧠 Descripción

Proyecto aplicado para construir modelos de **regresión** y **clasificación** usando PyTorch, y exponer al menos uno mediante una API simple.

Este proyecto continúa el aprendizaje del:

```txt
07-neural-network-foundations-lab
```

pero ahora pasa de entender la mecánica interna de una red neuronal a usar PyTorch de forma profesional.

Este proyecto pertenece al:

```txt
Plan 2 — Deep Learning Core
```

y forma parte del conjunto:

```txt
Núcleo de Redes Neuronales Profundas
```

La idea no es usar PyTorch como magia.

La idea es entender cómo PyTorch organiza el flujo de Deep Learning:

```txt
tensores
→ dataset
→ modelo
→ loss
→ optimizer
→ training loop
→ evaluation
→ inference
→ API
```

---

## 🎯 Objetivo

Crear modelos simples de regresión y clasificación usando PyTorch, entrenarlos, evaluarlos, guardarlos y exponer al menos uno mediante una API de inferencia.

El objetivo técnico es conectar fundamentos de redes neuronales con una implementación más profesional y reutilizable.

---

## 👤 Usuario objetivo

* AI Engineer en formación.
* Estudiante de Deep Learning.
* Desarrollador que quiere usar PyTorch con criterio.
* Reclutador técnico interesado en modelos PyTorch aplicados.
* Futuro constructor de CNNs, Transformers, Diffusion Models y sistemas RL.

---

## 🧱 Arquitectura esperada

```txt
Dataset
   ↓
Preprocesamiento
   ↓
Tensores
   ↓
Dataset / DataLoader
   ↓
Modelo PyTorch
   ↓
Training Loop
   ↓
Evaluation Loop
   ↓
Model Saving
   ↓
Inference Service
   ↓
FastAPI
```

---

## 🔁 Flujo técnico

```txt
data/raw
   ↓
data cleaning
   ↓
feature preparation
   ↓
X / y
   ↓
torch.Tensor
   ↓
Dataset / DataLoader
   ↓
nn.Module
   ↓
loss function
   ↓
optimizer
   ↓
training loop
   ↓
evaluation
   ↓
save model
   ↓
predict endpoint
```

---

## 🧩 Módulos

### Módulo 1 — Tensor Preparation

Convertir datos en tensores aptos para PyTorch.

Incluye:

* Separación de `X` e `y`.
* Conversión a `torch.Tensor`.
* Tipos de datos correctos.
* Shapes correctos.
* Preparación de batches.

Pregunta central:

```txt
¿Cómo convierto datos normales en datos que PyTorch puede entrenar?
```

---

### Módulo 2 — PyTorch Regression Model

Crear un modelo de regresión con PyTorch.

Incluye:

* `nn.Module`.
* Capas lineales.
* Activación si aplica.
* Predicción numérica.
* Loss de regresión.
* Métricas de regresión.

Pregunta central:

```txt
¿Cómo uso una red neuronal para predecir un valor numérico?
```

---

### Módulo 3 — PyTorch Classification Model

Crear un modelo de clasificación con PyTorch.

Incluye:

* `nn.Module`.
* Capas lineales.
* Salida para clases.
* Binary classification o multiclass classification.
* Loss de clasificación.
* Métricas de clasificación.

Pregunta central:

```txt
¿Cómo uso una red neuronal para predecir una clase?
```

---

### Módulo 4 — Training Loop

Construir el ciclo de entrenamiento.

Incluye:

* Forward pass.
* Loss.
* `loss.backward()`.
* `optimizer.step()`.
* `optimizer.zero_grad()`.
* Epochs.
* Registro de pérdida.

Pregunta central:

```txt
¿Qué ocurre dentro de cada epoch cuando entreno con PyTorch?
```

---

### Módulo 5 — Evaluation Loop

Evaluar modelos sin actualizar pesos.

Incluye:

* Modo evaluación.
* Desactivación de gradientes.
* Predicciones sobre test.
* Cálculo de métricas.
* Comparación train vs test.
* Detección básica de overfitting.

Pregunta central:

```txt
¿Cómo sé si el modelo aprendió o solo memorizó?
```

---

### Módulo 6 — Model Saving and Loading

Guardar y cargar el modelo entrenado.

Incluye:

* Guardado de pesos.
* Carga del modelo.
* Separación entre entrenamiento e inferencia.
* Validación de que el modelo cargado predice correctamente.

Pregunta central:

```txt
¿Cómo convierto el modelo entrenado en un artefacto reutilizable?
```

---

### Módulo 7 — Model Inference API

Exponer el modelo mediante una API simple.

Incluye:

* Request schema.
* Response schema.
* Service de inferencia.
* Endpoint `/predict`.
* Manejo básico de errores.
* Prueba con Swagger o cliente HTTP.

Pregunta central:

```txt
¿Cómo hago que un modelo PyTorch pueda ser usado fuera del notebook?
```

---

## 🧪 Labs

### tec-labs

* `tec-pytorch-tensors-lab`
* `tec-pytorch-dataloader-lab`
* `tec-pytorch-training-loop-lab`
* `tec-regression-vs-classification-loss-lab`
* `tec-pytorch-model-saving-lab`
* `tec-pytorch-inference-api-lab`

---

## 📊 Métricas

### Regresión

* MAE.
* MSE.
* RMSE.
* R² si aplica.
* Loss por epoch.

### Clasificación

* Accuracy.
* Precision.
* Recall.
* F1-score.
* Confusion Matrix.
* Loss por epoch.

---

## 📌 Próximos pasos

* Elegir dataset simple para regresión.
* Elegir dataset simple para clasificación.
* Preparar `X` e `y`.
* Convertir datos a tensores.
* Crear `Dataset` y `DataLoader`.
* Crear modelo de regresión con `nn.Module`.
* Crear modelo de clasificación con `nn.Module`.
* Implementar training loop.
* Implementar evaluation loop.
* Calcular métricas.
* Guardar modelo entrenado.
* Cargar modelo para inferencia.
* Crear endpoint `/predict`.
* Probar API.
* Documentar diferencias entre regresión y clasificación.
* Grabar demo o explicación corta.
* Actualizar LinkedIn y CV.

---

## ✅ Entregable final

Al terminar este proyecto debe existir:

* Modelo PyTorch de regresión.
* Modelo PyTorch de clasificación.
* Training loop funcional.
* Evaluation loop funcional.
* Métricas registradas.
* Modelo guardado.
* Modelo cargado para inferencia.
* API simple de predicción.
* Tests básicos.
* README profesional.
* Labs documentados.
* Explicación clara de cómo PyTorch organiza el entrenamiento.

---

## 🧭 Regla final

```txt
No uso PyTorch para saltarme los fundamentos.
Uso PyTorch para construir mejor después de entenderlos.
```

Este proyecto no busca entrenar una red enorme.

Busca demostrar que puedo usar PyTorch con criterio para construir modelos entrenables, evaluables, guardables y servibles por API.
