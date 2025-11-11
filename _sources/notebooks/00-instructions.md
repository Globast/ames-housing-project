---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---

# Capítulo 0 · Instrucciones de reproducción
## Instrucciones de reproducción y Demostraciones
> **Overview**: Este capitulo presenta cómo compilar el libro, la estructura de las carpetas y las dependencias para garantizar reproducibilidad. Incluye demostraciones teóricas y comandos mínimos para construir el proyecto. 



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
### Modelo y notación

El modelo lineal es:

```{math}
:label: eq:7.m1-modelo
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, 
\qquad 
\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_n)
```


donde:

- X: matriz de diseño n × k  
- β: vector de parámetros  
- ε: vector de errores  


### Estimador OLS y matriz hat

El estimador de mínimos cuadrados es:

```{math}
:label: eq:7.m2-beta
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}
```

Las predicciones:

```{math}
:label: eq:7.m3-yhat-H
\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{H}\mathbf{y},
\qquad
\mathbf{H} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top
```

Propiedades de H:
- Simétrica: H' = H  
- Idempotente: H² = H  
- Rango: rank(H) = k  


### Residuos y operador (I - H)

Los residuos:

```{math}
:label: eq:7.m4-residuos
\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I}-\mathbf{H})\mathbf{y}
```

Sustituyendo y = Xβ + ε y usando (I - H)X = 0:

```{math}
:label: eq:7.m5-e-from-eps
\mathbf{e} = (\mathbf{I}-\mathbf{H})\boldsymbol{\varepsilon}
```

Cada residuo es una combinación lineal de los errores originales.


### Suma de cuadrados residual

```{math}
:label: eq:7.m6-ssr-expand
\mathrm{SSR} 
= \mathbf{e}^\top \mathbf{e} 
= \big[(\mathbf{I}-\mathbf{H})\boldsymbol{\varepsilon}\big]^\top
   \big[(\mathbf{I}-\mathbf{H})\boldsymbol{\varepsilon}\big]
= \boldsymbol{\varepsilon}^\top (\mathbf{I}-\mathbf{H})^\top(\mathbf{I}-\mathbf{H}) \boldsymbol{\varepsilon}
```

Como (I - H) es simétrica e idempotente:

```{math}
:label: eq:7.m7-ssr-qf
\mathrm{SSR} = \boldsymbol{\varepsilon}^\top (\mathbf{I}-\mathbf{H}) \boldsymbol{\varepsilon}
```

El SSR es una forma cuadrática en los errores.


### Rango y grados de libertad

```{math}
:label: eq:7.m8-rank
\operatorname{rank}(\mathbf{I}-\mathbf{H}) = n - \operatorname{rank}(\mathbf{H}) = n-k
```

→ El espacio de los residuos tiene dimensión n - k.  
→ Solo n - k residuos son independientes (pues X'e = 0).


### Distribución Chi-cuadrado

Si ε ~ N(0, σ²I) y A es simétrica e idempotente de rango r:

```{math}
:label: eq:7.m9-chi2-theorem
\frac{1}{\sigma^2}\,\boldsymbol{\varepsilon}^\top \mathbf{A}\,\boldsymbol{\varepsilon} \sim \chi^2_r
```

Aplicando A = I - H:

```{math}
:label: eq:7.m10-chi2-apply
\frac{1}{\sigma^2}\,\boldsymbol{\varepsilon}^\top (\mathbf{I}-\mathbf{H})\,\boldsymbol{\varepsilon} \sim \chi^2_{n-k}
```

Por tanto:

```{math}
:label: eq:7.m11-ssr-chi2
\frac{\mathrm{SSR}}{\sigma^2} \sim \chi^2_{n-k}
```


### Por qué no basta con elevar los residuos al cuadrado


- Los errores εᵢ son normales independientes, por lo que (εᵢ / σ)² ~ χ²₁  
- Pero los residuos eᵢ son combinaciones lineales → no independientes  
- Además, Var(eᵢ) = σ²(1 - hᵢᵢ)  
- Por eso, solo la suma total e'e sigue χ², no cada residuo por separado.



### Intuición geométrica

- ε vive en un espacio de dimensión n  
- H proyecta sobre el subespacio de predicciones (dimensión k)  
- (I - H) proyecta sobre el espacio ortogonal de residuos (dimensión n - k)  
- La **longitud al cuadrado** de esa proyección, dividida por σ², sigue χ²ₙ₋ₖ


### Resultado final

```{math}
:label: eq:7.m12-resultado
\mathrm{SSR} = \boldsymbol{\varepsilon}^\top (\mathbf{I}-\mathbf{H}) \boldsymbol{\varepsilon},
\qquad
\frac{\mathrm{SSR}}{\sigma^2} \sim \chi^2_{n-k}
```

En regresión lineal simple (k = 2): n - 2 grados de libertad.

> **Key takeaways**
>- Hay instrucciones claras de build y control de versiones, con dataset esperado y seeds fijas.
>- Las demostraciones cubren de forma compacta OLS en álgebra lineal y propiedades del SSR.
>- El capítulo es la “bitácora” de reproducibilidad del libro.
>-La reproducibilidad se garantiza fijando semillas y documentando versiones de paquetes.
