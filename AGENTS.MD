# AGENTS.md

Este archivo da contexto a cualquier agente de IA (Antigravity u otro) que
trabaje en este repositorio. Léelo completo antes de tocar código.

## Qué es este proyecto

Estimación de edad a partir de imágenes faciales. Se comparan de 4 a 8
arquitecturas CNN (transfer learning con `timm`) usando el enfoque DEX
(clasificación por bins de edad + esperanza matemática como predicción
final). El objetivo no es solo entregar un modelo, sino dejar un pipeline
que se pueda seguir reentrenando cuando lleguen datos nuevos.

## Modo de trabajo para este proyecto (obligatorio)

- **Paso a paso, no todo de una vez.** Para este proyecto, ningún agente
  debe tomar una tarea grande (ej. "implementa el entrenamiento de las 8
  arquitecturas") y ejecutarla completa sin pausas. Se divide en pasos
  chicos y verificables, y se confirma que cada paso funciona antes de
  seguir al siguiente. Si una tarea pedida es grande, el primer paso del
  agente es proponer cómo la va a dividir, no empezar a escribir código.
- **Cero hardcodeo, sin excepciones.** Ningún valor que pueda cambiar
  (rutas, hiperparámetros, nombres de columnas, tamaños de imagen, rutas
  de checkpoints, IDs de Drive, umbrales) va escrito directo en el código.
  Todo eso vive en `configs/*.yaml`, variables de entorno (`.env`, nunca
  commiteado) o argumentos de CLI. Si un agente necesita "un número
  rápido para probar", igual va a un config, no al código.
- **Buenas prácticas sí o sí, no son negociables ni opcionales.** Esto
  incluye como mínimo: nombres de variables y funciones descriptivos (no
  `x`, `df2`, `tmp`), funciones con una sola responsabilidad, manejo de
  errores explícito (no `except: pass`), type hints en funciones nuevas
  de `src/`, y docstrings breves en cualquier función pública. Un agente
  no debe sacrificar esto por velocidad, ni siquiera si la tarea parece
  pequeña o desechable.

## Setup

```bash
pip install -r requirements.txt
```

Entrenamiento local/Colab:

```bash
python src/train.py --config configs/<arquitectura>.yaml
```

Reentrenamiento incremental (warm start desde un checkpoint existente):

```bash
python src/retrain.py --base-checkpoint models/best.pt --new-data data/incoming/
```

## Mapa del repo

| Ruta | Qué hay |
|---|---|
| `src/data/` | Datasets, transforms, splits |
| `src/models/` | Wrappers por arquitectura sobre `timm` |
| `src/train.py` | Loop de entrenamiento, recibe un config YAML |
| `src/retrain.py` | Fine-tuning incremental con warm start |
| `src/evaluate.py` | Cálculo de MAE en val/test |
| `configs/*.yaml` | Un archivo por experimento/arquitectura — no hardcodear hiperparámetros en el código |
| `models/registry.json` | Cuál checkpoint es el de "producción" y por qué |
| `experiments/results.csv` | Tabla acumulada de resultados por experimento |
| `docs/weekly_log.md` | Bitácora del proceso (qué se hizo, qué funcionó) |
| `data/` | Nunca se commitea directo — va versionado con DVC |

## Reglas para agentes — leer antes de modificar nada

1. **Nunca commitear datasets ni checkpoints (`.pt`, `.pth`, `.ckpt`) directo a Git.** Van con `dvc add`, no con `git add`. Si un archivo pesa más de unos pocos MB y no es código, es señal de que debería ir por DVC.
2. **Nunca sobrescribir `models/registry.json` a mano.** Solo se actualiza mediante `src/retrain.py`, y solo si el nuevo checkpoint mejora el MAE en el set de validación fijo. Si el agente entrena algo nuevo, reporta el resultado — no decide solo que ese modelo pasa a producción.
3. **Un config YAML nuevo por experimento nuevo**, no editar `configs/resnet50.yaml` para probar otra cosa — copiarlo con otro nombre. Así queda trazabilidad de qué hiperparámetros dieron qué resultado.
4. **Toda corrida de entrenamiento completa se registra** en `experiments/results.csv` (aunque también se loguee en Weights & Biases) y se resume en `docs/weekly_log.md` si fue un hito relevante de la semana.
5. **Commits pequeños y descriptivos**, en español, en tiempo presente: `agrega baseline CNN`, `entrena EfficientNet-B0, MAE val 6.4`, no `update` ni `fix stuff`.
6. **No instalar ni cambiar de framework.** El proyecto usa PyTorch + `timm`. Si un agente considera que TensorFlow sería mejor para algo puntual, debe preguntar antes, no migrar código.
7. **Antes de un PR a `main`**, correr localmente lo mismo que valida `.github/workflows/ci.yml` (que los imports básicos de `torch`/`timm` funcionen) para no descubrir el error en GitHub.
8. **Si el agente va a "hacer todo el trabajo" de un experimento** (entrenar, evaluar, documentar), debe dejar todo esto en el mismo commit o PR: el config usado, la fila en `results.csv`, y una línea en `weekly_log.md` — un experimento sin esos tres elementos no queda trazado y no cuenta como completo.
9. **Nunca subir datos personales/rostros nuevos capturados desde la demo sin consentimiento explícito registrado.** Si se implementa el bucle de retroalimentación (`data/incoming/`), cualquier dato ahí debe tener su origen y consentimiento documentado antes de integrarse al dataset versionado.

## Convención de ramas

- `main`: siempre debe poder correr sin errores
- `feat/<arquitectura>`: un branch por backbone en experimentación (ej. `feat/resnet50`)
- Merge a `main` solo cuando el experimento cierra y está documentado (ver regla 8)

## Métrica de referencia

MAE (Mean Absolute Error) en años, sobre el set de test fijo (no el de
validación usado para early stopping). Cualquier comparación entre
arquitecturas o entre versiones del modelo debe reportarse en esta métrica
para ser comparable.