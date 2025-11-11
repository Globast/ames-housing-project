---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---

# Capítulo 0 · Instrucciones de reproducción
## Instrucciones de reproducción y Demostraciones
> **Overview**: Este capitulo presenta instrucciones de reproducción. Se muestran las demostraciones solicitadas.


## Cómo compilar el libro:

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

**Control de versiones** 
```
book/
├── data/
├── notebooks/
├── _build/
└── _config.yml
```

**Semillas reproducibles**: todos los experimentos fijan `random_state`.
**Mapa del libro**: ver la barra lateral y `_toc.yml`.



## Demostraciones solicitadas

### Enunciado

1. Sea un modelo de regresión lineal simple; muestra que la suma de cuadrados de los residuos dividida por $ \sigma^2 $ puede escribirse como una **combinación cuadrática** de los errores $ \varepsilon_i $ y, usando ese resultado, que su distribución es $ \chi^2_{n-2} $.

La varianza es $ \sigma^2 $.

La esperanza es \( \mathbb{E}[X] \).

### Paso 1 — Modelo y notación matricial

```{math}
:label: eq:2.11.1-modelo
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\, \sigma^2 \mathbf{I}_n)
```
donde en regresión simple
$\mathbf{X} = [\mathbf{1}\ \ \mathbf{x}] \in \mathbb{R}^{n\times 2}$
(tiene **p = 2** columnas: intercepto y regresor).

El estimador OLS es
```{math}
:label: eq:2.11.1-beta
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
```
El vector de **residuos** es
```{math}
:label: eq:2.11.1-residuos
\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}
= \left(\mathbf{I}_n - \mathbf{H}\right)\mathbf{y},
```
con la **matriz sombrero** (proyección) definida por
```{math}
:label: eq:2.11.1-hat
\mathbf{H} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top, \qquad
\mathbf{M} \equiv \mathbf{I}_n - \mathbf{H}.
```

### Paso 2 — Forma cuadrática

Sustituyendo {eq}`eq:2.11.1-modelo` en {eq}`eq:2.11.1-residuos`,
```{math}
:label: eq:2.11.1-e-M-eps
\mathbf{e} = \mathbf{M}\mathbf{y} = \mathbf{M}(\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon})
= \underbrace{\mathbf{M}\mathbf{X}}_{=\ \mathbf{0}} \boldsymbol{\beta} + \mathbf{M}\boldsymbol{\varepsilon}
= \mathbf{M}\boldsymbol{\varepsilon},
```
pues $\mathbf{M}\mathbf{X} = (\mathbf{I}-\mathbf{H})\mathbf{X} = \mathbf{X} - \mathbf{X} = \mathbf{0}$.

La **suma de cuadrados de residuos** es
```{math}
:label: eq:2.11.1-ssr
SS_{\text{Res}} = \mathbf{e}^\top \mathbf{e} = (\mathbf{M}\boldsymbol{\varepsilon})^\top(\mathbf{M}\boldsymbol{\varepsilon})
= \boldsymbol{\varepsilon}^\top \mathbf{M}^\top \mathbf{M}\,\boldsymbol{\varepsilon}.
```
Como $\mathbf{H}$ es simétrica e idempotente ($\mathbf{H}=\mathbf{H}^\top$, $\mathbf{H}^2=\mathbf{H}$),
entonces $\mathbf{M}=\mathbf{I}-\mathbf{H}$ es también **simétrica** e **idempotente**:
$\mathbf{M}^\top=\mathbf{M}$ y $\mathbf{M}^2=\mathbf{M}$.
Por tanto,
```{math}
:label: eq:2.11.1-ssr-forma
SS_{\text{Res}} = \boldsymbol{\varepsilon}^\top \mathbf{M}\,\boldsymbol{\varepsilon}
\qquad\Longrightarrow\qquad
\frac{SS_{\text{Res}}}{\sigma^2} = \left(\frac{\boldsymbol{\varepsilon}}{\sigma}\right)^\top
\mathbf{M}\left(\frac{\boldsymbol{\varepsilon}}{\sigma}\right).
```
Esta es una **combinación cuadrática** de los errores, como se pedía.

### Paso 3 — Distribución $\chi^2$ y grados de libertad

Sea $\mathbf{z} \equiv \boldsymbol{\varepsilon}/\sigma \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_n)$.
Entonces, por {eq}`eq:2.11.1-ssr-forma`,
```{math}
:label: eq:2.11.1-qf
\frac{SS_{\text{Res}}}{\sigma^2} = \mathbf{z}^\top \mathbf{M}\,\mathbf{z}.
```
Teorema clásico: si $\mathbf{A}$ es simétrica e idempotente con **rango $r$**, y $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$,
entonces $\mathbf{z}^\top \mathbf{A}\,\mathbf{z} \sim \chi^2_r$. Aquí $\mathbf{A}=\mathbf{M}$.

En regresión simple ($p=2$), $\operatorname{rank}(\mathbf{H})=p=2$, por lo que
```{math}
:label: eq:2.11.1-rank
\operatorname{rank}(\mathbf{M}) = \operatorname{rank}(\mathbf{I}-\mathbf{H}) = n - \operatorname{rank}(\mathbf{H}) = n-2.
```
Concluimos que
```{math}
:label: eq:2.11.1-chi2
\frac{SS_{\text{Res}}}{\sigma^2} \sim \chi^2_{\,n-2}.
```

### Comentario final (intuición)

- $\mathbf{H}$ proyecta $\mathbf{y}$ sobre el subespacio generado por las columnas de $\mathbf{X}$ (dimensión $p$).  
- $\mathbf{M}$ proyecta sobre su **complemento ortogonal** (dimensión $n-p$), donde viven los residuos.  
- Al estandarizar los errores por $\sigma$, la energía de la proyección en ese subespacio (forma cuadrática) sigue una $\chi^2$ con $n-p$ grados de libertad, que aquí es $n-2$.

**Referencias internas.** De {eq}`eq:2.11.1-e-M-eps` y {eq}`eq:2.11.1-ssr` se obtiene la forma cuadrática {eq}`eq:2.11.1-ssr-forma`; usando {eq}`eq:2.11.1-rank` se deduce la distribución en {eq}`eq:2.11.1-chi2`.

