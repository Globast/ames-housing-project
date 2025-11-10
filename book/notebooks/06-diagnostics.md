---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# Capítulo 4 · Diagnóstico del modelo

> **Overview**: Evaluar supuestos (linealidad, homocedasticidad, normalidad de residuos, influencia) y proponer correcciones.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path('data/ames_housing.csv'))
df['log_SalePrice'] = np.log(df['SalePrice'])
X = sm.add_constant(df[['Gr Liv Area','Overall Qual','Garage Cars','Total Bsmt SF','Year Built']])
y = df['log_SalePrice']
ols = sm.OLS(y, X).fit()

resid = ols.resid
fitted = ols.fittedvalues

# Residuos vs ajustados
plt.scatter(fitted, resid, alpha=0.5)
plt.axhline(0, ls='--')
plt.title('Residuos vs Ajustados')
plt.xlabel('Ajustados')
plt.ylabel('Residuos')
plt.show()

# QQ-plot
sm.qqplot(resid, line='45')
plt.title('QQ-plot de residuos')
plt.show()
```

Incluye referencias a figuras (p. ej., {numref}`fig:diag-resid`) y discute patrones, heterocedasticidad y outliers (DFBETAs, leverage, Cook).

{takeaways}
