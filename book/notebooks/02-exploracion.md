# Capítulo 2 · Exploración de datos (EDA)

> **Overview**: Explorar la distribución de `SalePrice` y relaciones con predictores clave; detectar valores atípicos y datos faltantes.

## Carga de datos
```{code-cell} python
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv(Path('data/ames_housing.csv'))
df.head()
```

## Limpieza mínima
```{code-cell} python
# Nos quedamos con columnas numéricas de interés para un primer vistazo
cols = ['SalePrice','Gr Liv Area','Overall Qual','Garage Cars','Garage Area','Total Bsmt SF', 'Year Built']
df_ = df[cols].copy()
df_.describe().T
```

## Figura 2.1 · Distribución de precios
```{code-cell} python
import matplotlib.pyplot as plt
ax = df_['SalePrice'].plot(kind='hist', bins=40)
ax.set_title('Histograma de SalePrice')
ax.set_xlabel('SalePrice')
plt.show()
```

Como se observa en {numref}`fig:eda-hist`, la distribución es asimétrica a la derecha, lo que motiva transformar `SalePrice`.

```{figure} ../data/placeholder.png
:name: fig:eda-hist
:align: center
:width: 600

Ejemplo de referencia cruzada a una figura numerada. Reemplaza por gráficos reales generados arriba (exportados con `plt.savefig(...)`).
```

## Tabla 2.1 · Correlaciones
```{code-cell} python
corrs = df_.corr(numeric_only=True).round(3).sort_values('SalePrice', ascending=False)
corrs
```

En la {numref}`tab:corr`, `Gr Liv Area` y `Overall Qual` muestran las correlaciones más altas con el precio.

```{table} Correlaciones con SalePrice
:name: tab:corr
:align: center

| Variable | Corr con SalePrice |
|---|---|
| Gr Liv Area | (completar) |
| Overall Qual | (completar) |
| Garage Cars | (completar) |
```

{takeaways}
