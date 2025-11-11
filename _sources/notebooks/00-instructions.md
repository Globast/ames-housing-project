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

## Modelo y notación

El modelo lineal es:

y = Xβ + ε,  ε ~ N(0, σ²Iₙ)

donde:

- X: matriz de diseño n × k  
- β: vector de parámetros  
- ε: vector de errores  


## Estimador OLS y matriz hat

El estimador de mínimos cuadrados es:

β̂ = (X'X)⁻¹ X'y

Las predicciones:

ŷ = Xβ̂ = H y,  H = X(X'X)⁻¹X'

Propiedades de H:
- Simétrica: H' = H  
- Idempotente: H² = H  
- Rango: rank(H) = k  


## Residuos y operador (I - H)

Los residuos:

e = y - ŷ = (I - H)y

Sustituyendo y = Xβ + ε y usando (I - H)X = 0:

e = (I - H)ε

Cada residuo es una combinación lineal de los errores originales.


## Suma de cuadrados residual

SSR = e'e = [(I-H)ε]'[(I-H)ε] = ε'(I-H)'(I-H)ε

Como (I - H) es simétrica e idempotente:

SSR = ε'(I - H)ε

El SSR es una forma cuadrática en los errores.


## Rango y grados de libertad

rank(I - H) = n - rank(H) = n - k

→ El espacio de los residuos tiene dimensión n - k.  
→ Solo n - k residuos son independientes (pues X'e = 0).


## Distribución Chi-cuadrado

Si ε ~ N(0, σ²I) y A es simétrica e idempotente de rango r:

(1/σ²) ε'Aε ~ χ²ᵣ

Aplicando A = I - H:

(1/σ²) ε'(I - H)ε ~ χ²ₙ₋ₖ

Por tanto:

SSR / σ² ~ χ²ₙ₋ₖ


## Por qué no basta con elevar los residuos al cuadrado

- Los errores εᵢ son normales independientes, por lo que (εᵢ / σ)² ~ χ²₁  
- Pero los residuos eᵢ son combinaciones lineales → no independientes  
- Además, Var(eᵢ) = σ²(1 - hᵢᵢ)  
- Por eso, solo la suma total e'e sigue χ², no cada residuo por separado.


## Intuición geométrica

- ε vive en un espacio de dimensión n  
- H proyecta sobre el subespacio de predicciones (dimensión k)  
- (I - H) proyecta sobre el espacio ortogonal de residuos (dimensión n - k)  
- La longitud al cuadrado de esa proyección, dividida por σ², sigue χ²ₙ₋ₖ


## Resultado final

SSR = ε'(I - H)ε  
SSR / σ² ~ χ²ₙ₋ₖ  

En regresión lineal simple (k = 2): n - 2 grados de libertad.


1. Sea un modelo de regresión lineal simple; muestra que la suma de cuadrados de los residuos dividida por $ \sigma^2 $ puede escribirse como una **combinación cuadrática** de los errores $ \varepsilon_i $ y, usando ese resultado, que su distribución es $ \chi^2_{n-2} $.

La varianza es $ \sigma^2 $.

La esperanza es \( \mathbb{E}[X] \).

1. … dividida por $$ \sigma^2 $$ … los errores $ \varepsilon_i $$ … es $ \chi^2_{n-2} $.

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

