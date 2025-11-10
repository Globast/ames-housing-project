---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# Capítulo 4: Formulación matricial del modelo OLS

## Overview
Se formula y ajusta un modelo OLS usando álgebra matricial. Se presentan las ecuaciones clave (normal equations) y se verifica la equivalencia con APIs de alto nivel.
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
## 4.1 Definición del modelo

El modelo de regresión lineal puede expresarse de forma matricial como:

$$
Y = X\beta + \varepsilon
$$

**Ecuación 4.1.1.** Regresión lineal.

La función de pérdida de mínimos cuadrados busca minimizar la suma de los errores al cuadrado:

$$
S(\beta) = (Y - X\beta)^\top (Y - X\beta)
$$

**Ecuación 4.1.2.** Función de pérdida mínimos cuadrados.

Donde los estimadores están dados por:

$$
\hat{\beta} = (X^\top X)^{-1}X^\top Y
$$

**Ecuación 4.1.3.** Estimadores de mínimos cuadrados.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv(DATA_PATH)

vars_modelo = [
    'Overall Qual', 'Gr Liv Area', 'Garage Cars',
    'Total Bsmt SF', '1st Flr SF', 'Full Bath',
    'Year Built', 'Fireplaces', 'Lot Area'
]

y_log = df["SalePrice_log"].values
X = df[vars_modelo].values
X = np.column_stack([np.ones(X.shape[0]), X])
nombres_vars = ["Intercept"] + vars_modelo

XtX_inv = np.linalg.inv(X.T @ X)
beta_log = XtX_inv @ (X.T @ y_log)

modelo_log = sm.OLS(y_log, sm.add_constant(df[vars_modelo])).fit()

mean_price = np.exp(df["SalePrice_log"]).mean()
beta_usd = beta_log * mean_price
ols_usd = modelo_log.params.values * mean_price

tabla_comparacion = pd.DataFrame({
    "Variable": nombres_vars,
    "Formulación matricial": np.round(beta_usd, 2),
    "Statsmodels": np.round(ols_usd, 2)
})

display(tabla_comparacion)
```

**Tabla 4.1.1.** Formulación matricial vs. Statsmodels.

Se construye un modelo de regresión lineal para predecir `SalePrice` a partir de nueve variables predictoras. 

Primero se calculan los coeficientes manualmente usando la formulación matricial de OLS, y luego se ajusta el mismo modelo con `statsmodels` para verificar los resultados. Se observan los mismos coeficientes.

Cada coeficiente indica cuánto se espera que cambie el `SalePrice` por un incremento de una unidad en la variable correspondiente, manteniendo constantes las demás variables. Por ejemplo, el coeficiente de `Overall Qual` es aproximadamente 17 686. Esto significa que, en promedio, por cada punto adicional en la calificación general de la casa, se espera que el precio de venta aumente unos 17 686 dólares, manteniendo constantes las otras variables del modelo.

## Takeaways
- La solución matricial \(\hat{\beta}=(X^TX)^{-1}X^Ty\) coincide con la obtenida vía librerías.
- El condicionamiento de \(X^TX\) anticipa problemas de varianza de coeficientes.
- Se sientan bases para contrastes de hipótesis.

