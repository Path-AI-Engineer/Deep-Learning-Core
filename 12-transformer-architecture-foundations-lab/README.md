# 12-transformer-architecture-foundations-lab

## 🧠 Descripción

Lab técnico avanzado para implementar un **Transformer pequeño** desde fundamentos.

Este proyecto cierra el:

```txt
Plan 2 — Deep Learning Core
```

y forma parte del conjunto:

```txt
Núcleo de Redes Neuronales Profundas
```

Este proyecto continúa la base del:

```txt
11-autoencoder-representation-lab
```

pero ahora cambia el enfoque:

```txt
Antes:
representaciones comprimidas con autoencoders

Ahora:
representaciones contextuales mediante atención
```

La idea no es construir un LLM completo.

La idea es entender los bloques internos que más adelante sostienen:

* LLMs.
* Vision Transformers.
* Modelos multimodales.
* RAG.
* Agentes.
* Diffusion Transformers.
* Sistemas avanzados de IA.

---

## 🎯 Objetivo

Implementar un Transformer pequeño para entender su arquitectura interna.

El objetivo técnico es comprender:

* Tokens.
* Embeddings.
* Positional Encoding.
* Query.
* Key.
* Value.
* Self-Attention.
* Multi-Head Attention conceptual.
* Feed Forward Network.
* Residual Connections.
* Layer Normalization conceptual.
* Transformer Block.
* Relación con LLMs y Vision Transformers.

---

## 👤 Usuario objetivo

* AI Engineer en formación.
* Estudiante de Deep Learning.
* Persona que quiere entender Transformers antes de usar LLMs.
* Futuro constructor de RAG, agentes, Vision Transformers, modelos multimodales y diffusion systems.
* Reclutador técnico interesado en fundamentos de arquitecturas modernas.

---

## 🧱 Arquitectura esperada

```txt
Secuencia de tokens
      ↓
Token Embeddings
      ↓
Positional Encoding
      ↓
Self-Attention
      ↓
Feed Forward Network
      ↓
Residual Connections
      ↓
Layer Normalization
      ↓
Mini Transformer Block
      ↓
Output
      ↓
Reporte técnico
```

---

## 🔁 Flujo técnico

```txt
text / sequence
   ↓
tokenization básica
   ↓
token ids
   ↓
embeddings
   ↓
positional encoding
   ↓
Q / K / V
   ↓
attention scores
   ↓
attention output
   ↓
feed forward
   ↓
transformer block
   ↓
prediction / mini task
```

---

## 🧩 Módulos

### Módulo 1 — Token and Embedding Layer

Convertir una secuencia en vectores.

Incluye:

* Tokenización básica.
* Token IDs.
* Embedding matrix.
* Representación vectorial.
* Diferencia entre token y embedding.

Pregunta central:

```txt
¿Cómo pasa una palabra o símbolo a convertirse en vector?
```

---

### Módulo 2 — Positional Encoding

Agregar información de posición a la secuencia.

Incluye:

* Orden de tokens.
* Posición dentro de la secuencia.
* Positional Encoding.
* Limitación de self-attention sin posición.
* Representación de orden.

Pregunta central:

```txt
¿Cómo sabe el modelo en qué orden aparecen los tokens?
```

---

### Módulo 3 — Self-Attention

Implementar el núcleo del Transformer.

Incluye:

* Query.
* Key.
* Value.
* Attention scores.
* Softmax.
* Weighted sum.
* Contexto.
* Relación entre tokens.

Pregunta central:

```txt
¿Cómo decide un token a qué otros tokens debe prestar atención?
```

---

### Módulo 4 — Multi-Head Attention Concept

Entender por qué se usan varias cabezas de atención.

Incluye:

* Múltiples perspectivas.
* Diferentes relaciones entre tokens.
* Atención paralela.
* Concatenación conceptual.
* Limitaciones de una sola cabeza.

Pregunta central:

```txt
¿Por qué una sola atención puede no ser suficiente?
```

---

### Módulo 5 — Feed Forward Network

Agregar transformación después de la atención.

Incluye:

* Capas lineales.
* Activación.
* Transformación token-wise.
* Proyección interna.
* Salida del bloque.

Pregunta central:

```txt
¿Qué aprende el modelo después de mezclar contexto con atención?
```

---

### Módulo 6 — Residual Connections and Normalization

Entender estabilidad y flujo de información.

Incluye:

* Residual connections.
* Layer normalization conceptual.
* Estabilidad del entrenamiento.
* Flujo de gradientes.
* Arquitectura de bloque moderno.

Pregunta central:

```txt
¿Por qué los Transformers usan conexiones residuales y normalización?
```

---

### Módulo 7 — Mini Transformer Block

Unir las piezas en un bloque pequeño.

Incluye:

* Embeddings.
* Positional Encoding.
* Self-Attention.
* Feed Forward.
* Residuals.
* Normalization.
* Output.

Pregunta central:

```txt
¿Cómo se conectan todas las piezas en un Transformer real?
```

---

### Módulo 8 — Mini Task and Architecture Notes

Probar el bloque en una tarea mínima y documentar arquitectura.

Incluye:

* Tarea pequeña.
* Predicción simple.
* Limitaciones.
* Diagrama conceptual.
* Relación con LLMs.
* Relación con Vision Transformers.

Pregunta central:

```txt
¿Qué aprendí del Transformer antes de usar modelos grandes?
```

---

## 🧪 Labs

### tec-labs

* `tec-token-embedding-lab`
* `tec-positional-encoding-lab`
* `tec-self-attention-lab`
* `tec-multi-head-attention-concept-lab`
* `tec-feed-forward-network-lab`
* `tec-residual-layernorm-lab`
* `tec-mini-transformer-block-lab`
* `tec-transformer-mini-task-lab`

---

## 📊 Métricas / señales de aprendizaje

Este proyecto no se mide por crear un LLM potente.

Se mide por comprensión arquitectónica.

Señales principales:

* Los tokens se convierten en embeddings.
* La posición se incorpora correctamente.
* Self-attention produce scores interpretables.
* Se entiende Q/K/V.
* Se entiende softmax aplicado a atención.
* Se entiende que la atención mezcla información entre tokens.
* Se entiende por qué hay feed forward después de atención.
* Se entiende para qué sirven residual connections y normalization.
* Se puede explicar cómo un Transformer pequeño se conecta con LLMs.
* Se puede explicar cómo un Transformer se conecta con Vision Transformers.

Métricas posibles:

* Loss de una mini tarea.
* Accuracy si la mini tarea es clasificación.
* Comparación de outputs antes/después de atención.
* Visualización de attention scores.
* Verificación de shapes.

---

## 📌 Próximos pasos

* Definir una secuencia pequeña.
* Crear tokenización básica.
* Crear token IDs.
* Crear embeddings.
* Implementar positional encoding.
* Implementar Q/K/V.
* Calcular attention scores.
* Aplicar softmax.
* Calcular attention output.
* Implementar feed forward.
* Agregar residual connections.
* Agregar layer normalization conceptual.
* Construir mini Transformer block.
* Probar una mini tarea.
* Visualizar attention scores si aplica.
* Documentar shapes.
* Documentar limitaciones.
* Conectar el aprendizaje con LLMs, Vision Transformers y modelos multimodales.
* Grabar demo o explicación corta.
* Actualizar LinkedIn y CV.

---

## ✅ Entregable final

Al terminar este proyecto debe existir:

* Tokenización básica.
* Embedding layer.
* Positional Encoding.
* Self-Attention implementado.
* Q/K/V explicado.
* Attention scores calculados.
* Feed Forward Network.
* Mini Transformer Block.
* Mini tarea ejecutable.
* Visualización o explicación de attention scores.
* README técnico.
* Labs documentados.
* Diagrama o nota de arquitectura.
* Conclusión sobre cómo esto prepara para LLMs, Vision Transformers y multimodal AI.

---

## 🧭 Regla final

```txt
No construyo un LLM.
Construyo la comprensión de la arquitectura que hace posibles los LLMs.
```

Este proyecto no busca escala.

Busca entendimiento profundo de la arquitectura Transformer.
