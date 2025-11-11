---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# Capítulo 5: Inferencia y grados de libertad
> **Overview**:
Se realiza inferencia: intervalos de confianza, pruebas de hipótesis sobre coeficientes y ajuste global del modelo. Se reportan supuestos y su impacto en la validez inferencial.

**Definir ruta de datos**
```{code-cell} ipython3
from pathlib import Path
import pandas as pd

# Definir ruta de datos relativa al capítulo (ejecutado desde book/notebooks/)
DATA_PATH = Path("../data/AmesHousing_sin_outliers.csv")
assert DATA_PATH.is_file(), f"No se encontró '{DATA_PATH}'"
print("Usando CSV:", DATA_PATH.resolve())

# Lectura canónica a reutilizar en el capítulo
df = pd.read_csv(DATA_PATH)
df.shape
```


## Grados de libertad

En regresión lineal, sea:

- $n$ = número de observaciones  
- $k$ = número de variables predictoras (sin contar el intercepto)

Se definen:

$$
\text{Df}_{\text{model}} = k
$$
**Ecuación 5.1.2.** Grados de libertad del modelo.

$$
\text{Df}_{\text{model}} = 9
$$

Representa la cantidad de información utilizada para estimar los $k$ coeficientes.

$$
\text{Df}_{\text{residual}} = n - k - 1
$$
**Ecuación 5.1.2.** Grados de libertad de los residuos.

$$
\text{Df}_{\text{residual}} = 2768 - 9 - 1
$$

$$
\text{Df}_{\text{residual}} = 2758
$$

Representa los grados de libertad restantes después de ajustar el modelo.

## Errores estándar

En un modelo de regresión lineal, los **errores estándar de los coeficientes** miden la incertidumbre asociada a cada estimador $\hat{\beta}_j$.  
En otras palabras, nos indican cuánto esperaríamos que varíen los coeficientes si repitiéramos el experimento con nuevas muestras del mismo tamaño. Coeficientes con errores estándar grandes son menos precisos y más sensibles al muestreo.

La suma de cuadrados de los residuos se define como:

$$
SS_{\text{Res}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Ecuación 5.2.1.** Suma de cuadrados de los residuos.

Esta cantidad representa la **variabilidad de `y` no explicada por el modelo**.

La varianza de los errores se estima dividiendo $SS_{\text{Res}}$ por los grados de libertad de los residuos ($\text{Df}_{\text{residual}}$):

$$
\hat{\sigma}^2 = \frac{SS_{\text{Res}}}{\text{Df}_{\text{residual}}}
$$

**Ecuación 5.2.2.** Estimación de la varianza de los errores.

La matriz de varianzas-covarianzas de los coeficientes se calcula como:

$$
\text{Var}(\hat{\beta}) = \hat{\sigma}^2 (X^\top X)^{-1}
$$

**Ecuación 5.2.3.** Matriz de varianzas-covarianzas de los coeficientes.

Finalmente, el **error estándar** de cada coeficiente $\hat{\beta}_j$ se obtiene tomando la raíz cuadrada de la diagonal correspondiente:

$$
SE(\hat{\beta}_j) = \sqrt{[\text{Var}(\hat{\beta})]_{jj}}
$$

**Ecuación 5.2.4.** Error estándar de los coeficientes.

```{code-cell} ipython3
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv(DATA_PATH)
vars_modelo = [
    'Overall Qual', 'Gr Liv Area', 'Garage Cars',
    'Total Bsmt SF', '1st Flr SF', 'Full Bath',
    'Year Built', 'Fireplaces', 'Lot Area'
]

X_sm = sm.add_constant(df[vars_modelo])
y = df["SalePrice_log"]
modelo = sm.OLS(y, X_sm).fit()

tabla_inferencia = modelo.summary2().tables[1]
tabla_inferencia = tabla_inferencia.rename(columns={
    'Coef.': 'coef',
    'Std.Err.': 'std err',
    'P>|t|': 'p-value'
})

mean_price = np.exp(df["SalePrice_log"]).mean()
tabla_inferencia["coef"] = tabla_inferencia["coef"] * mean_price

tabla_inferencia

n = df.shape[0]
print(n)
```

## Valores P

Consecuentemente, los **valores p** permiten evaluar la significancia estadística de cada coeficiente $\hat{\beta}_j$.  
En otras palabras, nos indican la probabilidad de obtener un coeficiente tan extremo como el observado si, en realidad, el coeficiente fuera cero (hipótesis nula $H_0: \beta_j = 0$).

Para calcular el valor p, primero se construye el **estadístico t** de cada coeficiente:

$$
t_j = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}
$$

**Ecuación 5.3.1.** Estadístico t de los coeficientes.

Bajo la hipótesis nula $H_0: \beta_j = 0$, este estadístico sigue una distribución t con $\text{Df}_{\text{residual}} = n - k - 1$ grados de libertad.

Luego, el valor p se obtiene como:

$$
p_j = 2 \cdot P(T > |t_j|)
$$

**Ecuación 5.3.2.** Valor p bilateral, donde $T \sim t_{\text{Df}_{\text{residual}}}$.

Un valor p pequeño (típicamente menor a 0.05) indica que hay evidencia suficiente para rechazar la hipótesis nula, es decir, que el coeficiente es significativamente distinto de cero, con lo cual se podría afirmar que existe una relación lineal.

Un valor p grande sugiere que no hay evidencia suficiente para afirmar que el coeficiente difiere de cero, sugiriendo así la inexistencia de una relación lineal entre la variable predictora y la de respuesta.

```{code-cell} ipython3
import pandas as pd
import numpy as np
import statsmodels.api as sm

pd.set_option('display.float_format', '{:.2f}'.format)
np.set_printoptions(suppress=True, precision=2)

df = pd.read_csv(DATA_PATH)
vars_modelo = [
    'Overall Qual', 'Gr Liv Area', 'Garage Cars',
    'Total Bsmt SF', '1st Flr SF', 'Full Bath',
    'Year Built', 'Fireplaces', 'Lot Area'
]

X_sm = sm.add_constant(df[vars_modelo])
y = df["SalePrice_log"]
modelo = sm.OLS(y, X_sm).fit()

tabla_inferencia = modelo.summary2().tables[1]
tabla_inferencia = tabla_inferencia.rename(columns={
    'Coef.': 'coef',
    'Std.Err.': 'std err',
    'P>|t|': 'p-value'
})

mean_price = np.exp(df["SalePrice_log"]).mean()
tabla_inferencia["coef"] = tabla_inferencia["coef"] * mean_price

tabla_inferencia
```

**Tabla 5.3.1.** Inferencia estadística del modelo 1.

Coeficientes con **errores estándar** muy pequeños, como los de `Overall Qual`, `Gr Liv Area` o `Year Built`, indican que estas estimaciones son bastante precisas. Por el contrario, un coeficiente con error estándar relativamente más grande, como `Full Bath` o `Garage Cars`, refleja mayor incertidumbre en la estimación de su efecto sobre el precio de la vivienda.  

**Valores p** menores que 0.05 se consideran significativos, indicando que los coeficientes son distintos de 0. En este modelo, la mayoría de las variables cumplen este criterio, de forma que contribuyen significativamente a explicar `SalePrice`. Por el contrario, `1st Flr SF` y `Full Bath` tienen valores p mayores a 0.05, sugiriendo que su efecto podría no ser relevante al controlar por los demás predictores.

> **Key takeaways**
>- Intervalos y p-valores dependen de supuestos de homocedasticidad y normalidad de errores.
>- Los efectos significativos son coherentes con el EDA, reforzando validez del modelo.
>- Se reconoce el riesgo de error tipo I por múltiples comparaciones.
>Los intervalos de confianza comunican precisión y deben acompañar toda estimación puntual.
>-La interpretación sustantiva exige distinguir significancia estadística de relevancia práctica.