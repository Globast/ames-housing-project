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

1. Sea un modelo de regresión lineal simple

```{math}
:label: eq:modelo_simple
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i,\quad i=1,\ldots,n
```

donde los errores aleatorios cumplen $\mathbb{E}(\varepsilon_i)=0$ y $\operatorname{Var}(\varepsilon_i)=\sigma^2$.

**(a)** Demuestra que la suma de cuadrados de los residuos dividida por $\sigma^2$,

```{math}
:label: eq:ssres_def
\frac{SS_{\text{Res}}}{\sigma^2}
\;=\;
\frac{\sum_{i=1}^n e_i^2}{\sigma^2},
```

puede escribirse como una **combinación cuadrática** de los errores $\varepsilon_i$.

**(b)** Usando el resultado anterior, muestra que

```{math}
:label: eq:ssres_chi2
\frac{SS_{\text{Res}}}{\sigma^2}\;\sim\; \chi^2_{\,n-2},
```

y explica por qué se restan **dos grados de libertad** en el modelo de regresión simple (los asociados a $\hat\beta_0$ y $\hat\beta_1$).

---

### Solución 

Sea $ \mathbf{y} = (y_1,\ldots,y_n)^\top$, $ \mathbf{X}=[\mathbf{1},\,\mathbf{x}]$ (columna de 1s y la de $x_i$),
$\boldsymbol\varepsilon=(\varepsilon_1,\ldots,\varepsilon_n)^\top$ con $\boldsymbol\varepsilon\sim \mathcal N(\mathbf{0},\sigma^2\mathbf{I})$.
El proyector sobre el espacio columna de $\mathbf{X}$ es $\mathbf{H}=\mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top$ y el **proyector al complemento** es $\mathbf{M}=\mathbf{I}-\mathbf{H}$.

Los residuos cumplen $\mathbf{e}=\mathbf{y}-\hat{\mathbf{y}}=\mathbf{M}\mathbf{y}=\mathbf{M}\boldsymbol\varepsilon$, luego

```{math}
:label: eq:ssres_quad
SS_{\text{Res}}
= \mathbf{e}^\top \mathbf{e}
= \boldsymbol\varepsilon^\top \mathbf{M}\,\boldsymbol\varepsilon,
```

que es una **forma cuadrática** en $\boldsymbol\varepsilon$ (simétrica porque $\mathbf{M}$ es simétrica).

Propiedades clave:
- $\mathbf{M}$ es **idempotente** y simétrica: $\mathbf{M}^2=\mathbf{M}$, $\mathbf{M}^\top=\mathbf{M}$.
- $\operatorname{rango}(\mathbf{M}) = n - \operatorname{rango}(\mathbf{X}) = n-2$ (en regresión simple hay dos parámetros: $\beta_0,\beta_1$).

Entonces, por el resultado clásico sobre formas cuadráticas de normales, si $\boldsymbol\varepsilon\sim \mathcal N(\mathbf{0},\sigma^2\mathbf{I})$ y
$\mathbf{A}$ es idempotente simétrica de rango $r$, se tiene
$$
\frac{\boldsymbol\varepsilon^\top \mathbf{A}\,\boldsymbol\varepsilon}{\sigma^2}\sim \chi^2_r.
$$
Aplicándolo con $\mathbf{A}=\mathbf{M}$ y $r=n-2$, se obtiene {eq}`eq:ssres_chi2`.

**¿Por qué se restan dos grados de libertad?**  
Porque el ajuste consume dos parámetros libres ($\beta_0$ y $\beta_1$), reduciendo la dimensión del subespacio de residuos de $n$ a $n-2$. Por eso el **ancho** del espacio donde viven los residuos (y, por ende, de $SS_{\text{Res}}$) es $n-2$.

_Referencia sugerida:_ [@DraperSmith1998].


