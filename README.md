# Age Estimation CNN

Proyecto de estimación de edad a partir de imágenes faciales, comparando
múltiples arquitecturas CNN (4 a 8) mediante transfer learning, con un
pipeline pensado para ser reentrenado a medida que llegan nuevos datos.

## Objetivo

A partir de una imagen de un rostro, predecir la edad de la persona con la
mayor precisión posible (medida en MAE — Mean Absolute Error, en años).

## Enfoque

En vez de regresión pura, se usa el enfoque **DEX (Deep EXpectation)**:
clasificación sobre bins de edad + cálculo de la esperanza matemática sobre
las probabilidades como predicción final. Esto suele dar mejor MAE que
predecir la edad directamente con una sola neurona de salida.

## Datasets

| Dataset | Uso | Notas |
|---|---|---|
| UTKFace | Entrenamiento / validación principal | ~23,700 imágenes, edades 0-116 |
| IMDB-WIKI | Pre-entrenamiento del backbone | ~500k imágenes, etiquetas ruidosas |
| APPA-REAL | Test / análisis de error | Edad "aparente" vs "real" |

Los datasets **no se versionan en Git** — se gestionan con DVC apuntando a
un remoto (Google Drive). Ver `data/raw.dvc`.

## Arquitecturas comparadas

1. CNN simple (baseline, desde cero)
2. ResNet50
3. MobileNetV3-Large
4. EfficientNet-B0
5. DenseNet121
6. InceptionV3
7. VGG16-BN
8. ConvNeXt-Tiny

Todas (excepto el baseline) preentrenadas en ImageNet vía `timm`, afinadas
sobre rostros.

## Estructura del repo

```
age-estimation-cnn/
├── data/                 # datasets versionados con DVC (no en Git directamente)
├── notebooks/            # exploración y prototipado
├── src/                  # código de datos, modelos, entrenamiento
├── configs/              # un YAML por arquitectura/experimento
├── models/               # registry.json — qué checkpoint es "producción"
├── experiments/          # respaldo local de métricas (además de W&B)
└── docs/weekly_log.md    # bitácora semanal del proceso
```

## Cómo correr

```bash
pip install -r requirements.txt
python src/train.py --config configs/resnet50.yaml
```

## Estado del proyecto

Ver [`docs/weekly_log.md`](docs/weekly_log.md) para el avance semana a semana
y [`experiments/results.csv`](experiments/results.csv) para la comparación
de resultados entre arquitecturas.

## Reentrenamiento incremental

Cuando hay datos nuevos etiquetados:

```bash
python src/retrain.py --base-checkpoint models/best.pt --new-data data/incoming/
```

El script solo reemplaza el checkpoint de "producción" (`models/registry.json`)
si el MAE en el set de validación fijo mejora respecto al actual.
