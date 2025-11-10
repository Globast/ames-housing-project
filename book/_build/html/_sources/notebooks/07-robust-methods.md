---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# Capítulo 5 · Métodos robustos

> **Overview**: Comparar OLS con estimadores robustos (Huber, RLM/Quantile) y evaluar estabilidad de coeficientes.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from pathlib import Path

df = pd.read_csv(Path('data/ames_housing.csv'))
df['log_SalePrice'] = np.log(df['SalePrice'])

X = df[['Gr Liv Area','Overall Qual','Garage Cars','Total Bsmt SF','Year Built']].copy()
X_sm = sm.add_constant(X)
y = df['log_SalePrice']

# OLS
ols = sm.OLS(y, X_sm).fit()

# Huber (sklearn)
hub = HuberRegressor().fit(X, y)

# Quantile (mediana)
qr = QuantileRegressor(quantile=0.5, alpha=0).fit(X, y)

# Comparación de coeficientes
import pandas as pd
coef = pd.DataFrame({
    'OLS': ols.params.reindex(['const'] + X.columns.tolist()),
    'Huber': pd.Series([hub.intercept_] + list(hub.coef_), index=['const'] + X.columns.tolist()),
    'Quantile(0.5)': pd.Series([np.nan] + list(qr.coef_), index=['const'] + X.columns.tolist())
})
coef
```

Discute diferencias y estabilidad. Analiza sensibilidad ante outliers retirando observaciones influyentes y comparando nuevamente.

{takeaways}
