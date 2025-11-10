# Capítulo 0 · Instrucciones de reproducción

> **Overview**: Este capítulo resume en 3–5 líneas el propósito y lo que el lector aprenderá.


**Cómo compilar el libro**:

```bash
jupyter-book build .
```

**Dependencias**: ver `requirements.txt` en la raíz de `book/`.

**Dataset**: se espera en `data/ames_housing.csv`.
- Descarga manual desde Kaggle o usa la API:
  ```bash
  kaggle datasets download -d prevek18/ames-housing-dataset -p data/ --unzip
  mv data/AmesHousing.csv data/ames_housing.csv
  ```

**Control de versiones** (estructura sugerida):
```
book/
├── data/
├── notebooks/
├── _build/
└── _config.yml
```

**Semillas reproducibles**: todos los experimentos fijan `random_state`.
**Mapa del libro**: ver la barra lateral y `_toc.yml`.

## Key takeaways

- Punto 1
- Punto 2
- Punto 3

