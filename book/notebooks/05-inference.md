---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# Capítulo 5: Inferencia y grados de libertad
```{code-cell} ipython3
from pathlib import Path
DATA_PATH = Path("../data/AmesHousing_codificada.csv")  # relativo a book/notebooks/
assert DATA_PATH.is_file(), "No se encontró '../data/AmesHousing_codificada.csv'"
print("Usando CSV:", DATA_PATH.resolve())
```


---

## **1️⃣ Definición de grados de libertad**

En un modelo lineal con $n$ observaciones y $k$ variables explicativas:

$$
Df_{Model} = k, \quad Df_{Residuals} = n - k - 1
$$

- $Df_{Model}$: número de parámetros estimados (sin contar el intercepto).  
- $Df_{Residuals}$: grados de libertad asociados a los errores o residuos.

---

## **2️⃣ Errores estándar y valores p**

El **error estándar** de cada coeficiente $\hat{\beta}_j$ mide la precisión de su estimación y se calcula como:

$$
SE(\hat{\beta}_j) = \sqrt{\sigma^2 (X^\top X)^{-1}_{jj}}
$$

donde:

- $\sigma^2 = \frac{SSE}{n - k - 1}$ es la varianza residual estimada.  
- $(X^\top X)^{-1}_{jj}$ es el elemento $j$-ésimo de la diagonal de la matriz inversa.

El **estadístico t** se define como:

$$
t_j = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}
$$

y el **valor p** se obtiene comparando este estadístico con una distribución t de Student con $Df_{Residuals}$ grados de libertad.

---

## **3️⃣ Interpretación de la tabla de coeficientes**

Los resultados del modelo incluyen:

| Parámetro | Descripción |
|------------|--------------|
| **coef** | Estimación del parámetro $\hat{\beta}_j$ |
| **std err** | Error estándar del estimador |
| **t** | Estadístico de prueba |
| **P>|t|** | Valor p de la hipótesis $H_0: \beta_j = 0$ |
| **[0.025, 0.975]** | Intervalo de confianza al 95% |

---

## **4️⃣ Significancia y efecto práctico**

- Si $p < 0.05$: se **rechaza $H_0$** y la variable es **estadísticamente significativa**.  
- La **magnitud** del coeficiente indica su **efecto práctico**: cuánto cambia $y$ ante un cambio unitario en esa variable, manteniendo las demás constantes.

Por ejemplo:
- Un aumento de 1 punto en `OverallQual` puede incrementar el precio promedio en miles de dólares.
- `GrLivArea` tiene un efecto proporcional: cada metro adicional eleva el valor de la vivienda.

---

## **5️⃣ Resumen conceptual**

$$
\hat{\beta}_j \pm t_{\alpha/2, Df_{Residuals}} \times SE(\hat{\beta}_j)
$$

Este intervalo permite construir intervalos de confianza al 95% para cada parámetro, mostrando el rango plausible donde se encuentra el valor real de $\beta_j$.
```{code-cell} ipython3
import pandas as pd
import statsmodels.api as sm

# Usar el dataset limpio
df = pd.read_csv(DATA_PATH)
vars_modelo = ['Overall Qual','Gr Liv Area','Garage Cars','Garage Area','Total Bsmt SF','1st Flr SF','Year Built','Year Remod/Add','Full Bath','Garage Yr Blt','TotRms AbvGrd','Fireplaces','Mas Vnr Area','BsmtFin SF 1']

# Modelo OLS
X_sm = sm.add_constant(df[vars_modelo])
y = df["SalePrice"]
modelo = sm.OLS(y, X_sm).fit()

# Resumen de inferencia
tabla_inferencia = modelo.summary2().tables[1]
tabla_inferencia = tabla_inferencia.rename(columns={
    'Coef.': 'coef',
    'Std.Err.': 'std err',
    'P>|t|': 'p-value'
})

# Mostrar tabla
tabla_inferencia
```
