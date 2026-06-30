# AI Engineer Roadmap — Plan 2

## 🧠 Deep Learning Core

Esta organización reúne los proyectos del **Plan 2 — Deep Learning Core**.

Este plan pertenece a la ruta mayor:

```txt
Path-AI-Engineer-2.0
```

El objetivo de este segundo plan es construir una base sólida en **redes neuronales profundas**, entendiendo cómo aprenden los modelos antes de avanzar hacia Computer Vision, LLMs, Diffusion Models, Reinforcement Learning, Quantum AI y Robotics.

Este plan no busca solo usar PyTorch como una caja negra.

Busca entender:

```txt
tensores
→ capas
→ pesos
→ activaciones
→ función de pérdida
→ backpropagation
→ optimización
→ entrenamiento
→ evaluación
→ documentación
```

La idea central es aprender cómo funcionan las redes neuronales desde sus fundamentos y construir pequeños sistemas o labs que demuestren comprensión real.

---

## 🎯 Objetivo general

Convertirme en un AI Engineer capaz de:

* Entender el funcionamiento interno de redes neuronales.
* Trabajar con tensores.
* Implementar training loops.
* Usar PyTorch con criterio.
* Entrenar modelos de regresión y clasificación con redes.
* Entender backpropagation.
* Usar funciones de pérdida.
* Aplicar optimizadores.
* Construir CNNs básicas.
* Entender modelos secuenciales como RNN, LSTM y GRU.
* Trabajar con autoencoders.
* Comprender fundamentos de GANs.
* Implementar un Transformer pequeño desde fundamentos.
* Documentar experimentos de Deep Learning.
* Prepararme para Computer Vision, LLMs, Diffusion y RL.

---

## 🧭 Filosofía de trabajo

Este plan se trabaja con una regla clara:

```txt
Primero entiendo cómo aprende una red.
Luego uso frameworks.
Después aplico modelos a dominios concretos.
```

No se busca correr modelos gigantes desde el inicio.

Se busca entender el núcleo:

```txt
input
↓
modelo
↓
predicción
↓
loss
↓
gradiente
↓
actualización de pesos
↓
nueva predicción
```

Cada proyecto debe dejar evidencia:

* Código ejecutable.
* Experimento reproducible.
* Métrica registrada.
* Gráfico si aplica.
* README claro.
* Conclusión técnica.
* Limitaciones.
* Aprendizaje explicado.

---

## 🧩 Conceptos base

### Red neuronal

Una red neuronal es un modelo compuesto por capas que transforman entradas en salidas mediante pesos aprendibles.

### Tensor

Un tensor es la estructura principal con la que trabajan los modelos de Deep Learning.

Puede representar:

* Datos tabulares.
* Imágenes.
* Secuencias.
* Embeddings.
* Pesos del modelo.
* Gradientes.

### Backpropagation

Backpropagation es el proceso que permite calcular cómo debe ajustarse cada peso del modelo para reducir el error.

### Training loop

Un training loop es el ciclo de entrenamiento:

```txt
forward pass
→ loss
→ backward pass
→ optimizer step
→ repeat
```

### Modelo profundo

Un modelo profundo usa varias capas para aprender representaciones más complejas que un modelo lineal simple.

---

## 🧪 Tipos de proyectos en este plan

### Lab

Proyecto corto para entender un concepto fundamental.

Ejemplos:

* Red neuronal desde cero.
* RNN/LSTM.
* Autoencoders.
* Transformer pequeño.

### Aplicado

Proyecto donde se usa Deep Learning para resolver un problema concreto.

Ejemplo:

* PyTorch para regresión y clasificación.

### Lab avanzado

Proyecto técnico donde se implementa una arquitectura con más profundidad conceptual.

Ejemplo:

* Transformer desde fundamentos.

---

## 🗺️ Cronograma Plan 2

| Semana | Proyecto                                   | Objetivo                                             |
| ------ | ------------------------------------------ | ---------------------------------------------------- |
| 28-30  | `07-neural-network-foundations-lab`        | Entender una red neuronal sin framework pesado       |
| 31-33  | `08-pytorch-regression-classification-api` | Usar PyTorch para regresión y clasificación          |
| 34-37  | `09-cnn-foundations-image-classifier`      | Construir una CNN básica para imágenes               |
| 38-40  | `10-sequence-models-rnn-lstm-lab`          | Entender modelos secuenciales                        |
| 41-44  | `11-autoencoder-representation-lab`        | Aprender representación y reconstrucción             |
| 45-49  | `12-transformer-from-scratch-mini-lab`     | Implementar un Transformer pequeño desde fundamentos |

Duración total aproximada:

```txt
22 semanas
```

---

# 📁 Proyectos del Plan 2

## 07 — neural-network-foundations-lab

### Objetivo

Construir una red neuronal pequeña desde cero para entender sus componentes internos.

No se busca crear un framework.

Se busca entender qué ocurre cuando una red aprende.

---

### Flujo

```txt
datos simples
→ inicialización de pesos
→ forward pass
→ cálculo de loss
→ gradiente
→ actualización de pesos
→ evaluación
```

---

### Aprendizajes principales

* Perceptrón.
* MLP básica.
* Pesos.
* Bias.
* Activaciones.
* Forward pass.
* Loss.
* Gradiente.
* Backpropagation conceptual.
* Gradient Descent.
* Learning rate.
* Overfitting básico.

---

### Módulos

* Forward Pass.
* Loss Function.
* Manual Gradient Update.
* Training Loop.
* Error Visualization.
* Limitations Notes.

---

### Labs

* `tec-forward-pass-lab`
* `tec-loss-function-lab`
* `tec-gradient-descent-lab`
* `tec-training-loop-lab`

---

### Entregable final

* Red neuronal simple desde cero.
* Training loop manual.
* Gráfico de pérdida.
* Explicación de cómo aprende.
* README técnico.
* Conclusión sobre límites de implementar desde cero.

---

## 08 — pytorch-regression-classification-api

### Objetivo

Crear modelos de regresión y clasificación con PyTorch y exponer al menos uno mediante una API simple.

Este proyecto conecta Deep Learning con software aplicado.

---

### Flujo

```txt
dataset
→ tensores
→ modelo PyTorch
→ training loop
→ evaluación
→ inferencia
→ API simple
```

---

### Aprendizajes principales

* PyTorch.
* Tensors.
* Dataset / DataLoader.
* MLP para regresión.
* MLP para clasificación.
* Loss functions.
* Optimizers.
* Training loop.
* Evaluation loop.
* Model saving.
* Inference.
* FastAPI básico aplicado a modelo PyTorch.

---

### Módulos

* Tensor Preparation.
* PyTorch Regression.
* PyTorch Classification.
* Training Loop.
* Evaluation Loop.
* Model Inference API.

---

### Labs

* `tec-pytorch-tensors-lab`
* `tec-pytorch-training-loop-lab`
* `tec-regression-vs-classification-loss-lab`
* `tec-pytorch-model-saving-lab`

---

### Entregable final

* Modelo de regresión en PyTorch.
* Modelo de clasificación en PyTorch.
* Métricas registradas.
* Modelo guardado.
* API simple de inferencia.
* README profesional.
* Demo local.

---

## 09 — cnn-foundations-image-classifier

### Objetivo

Construir una CNN básica para clasificación de imágenes.

Este proyecto introduce la idea de percepción visual desde Deep Learning.

---

### Flujo

```txt
dataset de imágenes
→ preprocessing
→ CNN
→ entrenamiento
→ evaluación
→ predicción
→ reporte
```

---

### Aprendizajes principales

* Imágenes como tensores.
* Canales.
* Convoluciones.
* Kernels.
* Pooling.
* Flatten.
* Fully connected layers.
* CNN training.
* Data augmentation básica.
* Accuracy.
* Confusion Matrix.
* Overfitting en imágenes.

---

### Módulos

* Image Tensor Preparation.
* CNN Architecture.
* Training and Evaluation.
* Data Augmentation.
* Error Analysis.
* Prediction Demo.

---

### Labs

* `tec-image-tensor-lab`
* `tec-convolution-pooling-lab`
* `tec-cnn-training-lab`
* `tec-image-overfitting-lab`

---

### Entregable final

* CNN básica entrenada.
* Dataset de imágenes preparado.
* Métricas de clasificación.
* Confusion Matrix.
* Predicciones de ejemplo.
* README técnico.
* Conclusión sobre limitaciones de CNN básica.

---

## 10 — sequence-models-rnn-lstm-lab

### Objetivo

Entender modelos secuenciales como RNN, LSTM y GRU.

Este proyecto introduce modelos que procesan información ordenada en el tiempo o en secuencia.

---

### Flujo

```txt
secuencia
→ token / timestep
→ RNN / LSTM / GRU
→ hidden state
→ predicción
→ evaluación
```

---

### Aprendizajes principales

* Secuencias.
* Time steps.
* Hidden state.
* RNN.
* Vanishing gradients.
* LSTM.
* GRU.
* Sequence classification.
* Sequence prediction.
* Padding básico.
* Batch de secuencias.
* Limitaciones frente a Transformers.

---

### Módulos

* Sequence Dataset.
* RNN Baseline.
* LSTM Model.
* GRU Model.
* Sequence Evaluation.
* Comparison Notes.

---

### Labs

* `tec-rnn-hidden-state-lab`
* `tec-lstm-memory-lab`
* `tec-gru-comparison-lab`
* `tec-sequence-model-limitations-lab`

---

### Entregable final

* RNN pequeña.
* LSTM pequeña.
* GRU pequeña.
* Comparación básica.
* Métricas.
* README explicativo.
* Nota sobre por qué Transformers superan muchas limitaciones secuenciales.

---

## 11 — autoencoder-representation-lab

### Objetivo

Construir autoencoders para entender representación, compresión y reconstrucción.

Este proyecto ayuda a entender cómo una red puede aprender una representación interna de los datos.

---

### Flujo

```txt
input
→ encoder
→ latent representation
→ decoder
→ reconstruction
→ reconstruction loss
```

---

### Aprendizajes principales

* Encoder.
* Decoder.
* Latent space.
* Reconstruction loss.
* Dimensionality reduction.
* Denoising autoencoder conceptual.
* Anomaly detection básica con reconstrucción.
* Visualización de representaciones.
* Limitaciones de autoencoders.

---

### Módulos

* Basic Autoencoder.
* Latent Representation.
* Reconstruction Evaluation.
* Denoising Concept.
* Anomaly Detection with Reconstruction.
* Representation Visualization.

---

### Labs

* `tec-basic-autoencoder-lab`
* `tec-latent-space-lab`
* `tec-reconstruction-loss-lab`
* `tec-autoencoder-anomaly-lab`

---

### Entregable final

* Autoencoder entrenado.
* Reconstrucciones de ejemplo.
* Métrica de reconstrucción.
* Visualización de latent space si aplica.
* README técnico.
* Conclusión sobre representación aprendida.

---

## 12 — transformer-from-scratch-mini-lab

### Objetivo

Implementar un Transformer pequeño desde fundamentos para entender su arquitectura interna.

No se busca crear un LLM completo.

Se busca entender los bloques que más adelante sostienen modelos de lenguaje, visión y multimodalidad.

---

### Flujo

```txt
tokens
→ embeddings
→ positional encoding
→ self-attention
→ feed-forward
→ residual connections
→ output
```

---

### Aprendizajes principales

* Tokenización básica.
* Embeddings.
* Positional Encoding.
* Self-Attention.
* Query, Key, Value.
* Multi-head Attention conceptual.
* Feed Forward Network.
* Residual Connections.
* Layer Normalization conceptual.
* Transformer Encoder básico.
* Limitaciones de escala.
* Relación con LLMs.

---

### Módulos

* Token and Embedding Layer.
* Positional Encoding.
* Self-Attention.
* Mini Transformer Block.
* Training Mini Task.
* Architecture Notes.

---

### Labs

* `tec-token-embedding-lab`
* `tec-positional-encoding-lab`
* `tec-self-attention-lab`
* `tec-mini-transformer-block-lab`

---

### Entregable final

* Transformer pequeño implementado.
* Ejemplo de tarea mínima.
* Explicación de Q/K/V.
* Diagrama o nota de arquitectura.
* README técnico.
* Conclusión sobre cómo esto se conecta con LLMs, Vision Transformers y modelos multimodales.

---

# 📊 Nivel esperado al terminar Plan 2

| Área                             | Nivel esperado |
| -------------------------------- | -------------: |
| PyTorch                          |           8/10 |
| Tensores                         |           8/10 |
| Backpropagation conceptual       |           8/10 |
| MLP                              |         8.5/10 |
| CNN básica                       |         7.5/10 |
| RNN / LSTM / GRU                 |           7/10 |
| Autoencoders                     |         7.5/10 |
| GANs básicos                     |         6.5/10 |
| Transformers fundamentos         |           7/10 |
| Regularización / Dropout         |           8/10 |
| Batch Normalization              |           7/10 |
| Training loops                   |           8/10 |
| Debugging de modelos DL          |         7.5/10 |
| Documentación de experimentos DL |           8/10 |

---

# 🧠 Resultado esperado del Plan 2

Al completar este plan, podré decir:

```txt
Sé cómo aprende una red neuronal.
Sé trabajar con tensores.
Sé construir training loops.
Sé usar PyTorch con criterio.
Sé entrenar modelos de regresión y clasificación con redes.
Sé construir CNNs básicas.
Sé entender modelos secuenciales.
Sé usar autoencoders para representación.
Sé comprender la base de los Transformers.
Sé documentar experimentos de Deep Learning.
Sé preparar la base para Computer Vision, LLMs, Diffusion, RL y Robotics.
```

---

# 🧭 Regla final

```txt
No corro redes profundas como magia.
Primero entiendo sus piezas.
Luego las entreno.
Después las aplico a problemas reales.
```

---

# 👤 Autor

**Jean Franck Loa Rojas**

AI Engineer Path Builder
Deep Learning • PyTorch • Neural Networks • CNNs • RNNs • LSTMs • Autoencoders • Transformers • Training Loops • Representation Learning
