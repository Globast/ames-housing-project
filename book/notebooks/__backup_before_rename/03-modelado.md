# Capítulo 3 · Regresión lineal (OLS) e inferencia

> **Overview**: Ajustar OLS con transformaciones razonables, obtener ICs y pruebas, y evaluar significancia y magnitud de efectos.

**Lectura de datos detectada en el .ipynb:**
```{code-cell} python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

df = pd.read_csv(Path('data/ames_housing.csv'))
df['log_SalePrice'] = np.log(df['SalePrice'])

X = df[['Gr Liv Area','Overall Qual','Garage Cars','Total Bsmt SF','Year Built']].copy()
X = sm.add_constant(X)
y = df['log_SalePrice']

ols = sm.OLS(y, X).fit()
print(ols.summary())
```

**Ecuación del modelo** (estimada, ver {numref}`eq:ols`):
```{math}
:label: eq:ols
\log(SalePrice) = \beta_0 + \beta_1\,GrLivArea + \beta_2\,OverallQual + \cdots + \varepsilon
```

Interpreta los coeficientes clave y discute **significancia**, **IC al 95%** y **bondad de ajuste** ({numref}`tab:ols-sum`).

```{code-cell} python
# Tabla resumida de coeficientes con IC95%
summ = ols.summary2().tables[1]
summ
```

{takeaways}
