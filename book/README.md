# Ames Housing Project (Jupyter Book)

Este repositorio contiene un libro reproducible sobre regresión lineal, inferencia, diagnóstico y métodos robustos usando **Ames Housing**.

## Estructura
```
book/
├── data/
│   └── ames_housing.csv
├── notebooks/
│   ├── 00-repro.md
│   ├── 01-introduccion.md
│   ├── 02-exploracion.md
│   ├── 03-modelado.md
│   ├── 04-diagnostico.md
│   ├── 05-metodos-robustos.md
│   └── 06-conclusiones.md
├── _config.yml
├── _toc.yml
└── requirements.txt
```

## Cómo compilar
Desde un entorno en la nube (GitHub Codespaces o GitHub Actions):

```bash
pip install -r requirements.txt
jupyter-book build .
```

Para publicar a GitHub Pages usando `ghp-import`:

```bash
ghp-import -n -p -f _build/html
```

Asegúrate de que GitHub Pages esté configurado para servir desde la rama `gh-pages`.
