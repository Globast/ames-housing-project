---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# Capítulo 2: Descripción y limpieza del dataset

> **Overview**:

**Lectura de datos detectada en el .ipynb:** ../data/AmesHousing.csv

```{code-cell} ipython3
from pathlib import Path
DATA_PATH = Path("../data/ames_housing.csv")  # relativo a book/notebooks/
assert DATA_PATH.is_file(), "No se encontró '../data/ames_housing.csv'"
print("Usando CSV:", DATA_PATH.resolve())
```
## Carga del dataset

```{code-cell} ipython3
import pandas as pd
import numpy as np

data = pd.read_csv(DATA_PATH)
display(data.head())
```

**Tabla 2.1.1.** Conjunto de datos *Ames Housing*.

Esta tabla muestra las primeras observaciones del dataset original, permitiendo verificar la correcta carga de los datos y la estructura general de las variables.

```{code-cell} ipython3
fuente = "Ames Housing Dataset (De Cock, 2011) — Iowa State University"
tamano = data.shape[0]
n_variables = data.shape[1]
licencia = "Open Data, libre uso académico"

print(f"Fuente: {fuente}")
print(f"Tamaño: {tamano} registros")
print(f"Número de variables: {n_variables}")
print(f"Licencia: {licencia}")
```

**Tabla 2.1.2.** Metadatos *Ames Housing*.

```{code-cell} ipython3
faltantes = data.isna().mean() * 100
tipos = data.dtypes

tabla_faltantes = pd.DataFrame({
    "Tipo de variable": tipos,
    "% Faltantes": faltantes.round(2)
})

tabla_faltantes = (
    tabla_faltantes[tabla_faltantes["% Faltantes"] > 0]
    .sort_values(by="% Faltantes", ascending=False)
)

tabla_faltantes.head(20)
```

**Tabla 2.1.3.** Valores faltantes por variable.

El tratamiento de valores faltantes se realizó de forma diferenciada según el tipo de variable.  
Para las **variables numéricas**, se imputó la **mediana**, una medida robusta frente a valores extremos.  
En las **variables categóricas**, se reemplazaron los valores ausentes por la **moda**, preservando la categoría más frecuente.

La regla aplicada se resume en la siguiente expresión:

$$
x_{ij}^{*} =
\begin{cases}
\text{Mediana}(X_j) & \text{si } X_j \text{ es numérica}\\[4pt]
\text{Moda}(X_j) & \text{si } X_j \text{ es categórica}
\end{cases}
$$

```{code-cell} ipython3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data_limpia = data.copy()

num_cols = data_limpia.select_dtypes(include=np.number).columns
for col in num_cols:
    data_limpia[col].fillna(data_limpia[col].median(), inplace=True)

cat_cols = data_limpia.select_dtypes(exclude=np.number).columns
for col in cat_cols:
    data_limpia[col].fillna(data_limpia[col].mode()[0], inplace=True)
```

## 2.2 Tratamiento de outliers

Se aplicó el criterio de **tres desviaciones estándar (3Z)** para identificar observaciones atípicas en las variables numéricas de interés, incluyendo la variable objetivo `SalePrice` y 13 predictoras potenciales.

Bajo este método, un dato se considera *outlier* si su distancia a la media supera tres desviaciones estándar:

$$
|z_i| = \left| \frac{x_i - \bar{x}}{s} \right| > 3
$$

**Ecuación 2.2.1.** Outlier 3Z.

```{code-cell} ipython3
import pandas as pd
import numpy as np

vars_outliers = [
    "Overall Qual", "Gr Liv Area","Garage Area", "Garage Cars","Bedroom AbvGr",
    "Total Bsmt SF", "Year Remod/Add","TotRms AbvGrd","1st Flr SF", "Full Bath",
    "Year Built", "Fireplaces", "Lot Area","SalePrice"
]

data_sin_outliers = data_limpia.copy()

def detectar_outliers_3z(col):
    mean = col.mean()
    std = col.std()
    return (col - mean).abs() > 3*std

outliers_mask = np.zeros(len(data_sin_outliers), dtype=bool)
for var in vars_outliers:
    outliers_mask |= detectar_outliers_3z(data_sin_outliers[var])

antes = len(data_sin_outliers)
n_outliers = outliers_mask.sum()

data_sin_outliers = data_sin_outliers.loc[~outliers_mask].copy()
despues = len(data_sin_outliers)

print(f"Registros antes: {antes}")
print(f"Outliers detectados (3Z): {n_outliers}")
print(f"Registros después de eliminar outliers: {despues}")
```

**Tabla 2.2.1.** Resumen tratamiento de outliers.

En total, se detectaron **162 observaciones atípicas**, las cuales fueron eliminadas del conjunto de datos con el objetivo de reducir la influencia de valores extremos sobre el ajuste del modelo a aplicar.

Además, se aplicó una transformación logarítmica natural a la variable **SalePrice** con el fin de mejorar la relación lineal entre esta y las variables predictoras, reduciendo cualquier sesgo presente en la distribución original. 

$$
Y' = \log(1 + Y)
$$

**Ecuación 2.2.2.** Transformación logarítmica. <a id="eq-2-2-2"></a>

```{code-cell} ipython3
data_sin_outliers["SalePrice_log"] = np.log1p(data_sin_outliers["SalePrice"])
data_sin_outliers.to_csv("../data/AmesHousing_sin_outliers.csv", sep =",", index=False)
```

## 2.3 Codificación de variables categóricas

Las variables categóricas se transformaron en valores numéricos para facilitar su uso en modelos.  

Primero se codificaron las **variables ordinales** según su nivel, asignando valores enteros que respetan el orden lógico de las categorías (por ejemplo, *Po* < *Fa* < *TA* < *Gd* < *Ex*).  

Posteriormente, las **variables nominales** (sin orden inherente) se codificaron mediante variables *dummy*, creando una representación binaria para cada categoría (y una nueva columna).

```{code-cell} ipython3
data_ordinal = data_sin_outliers.copy()

map_calidad = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

map_poolqc = {"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}

map_bsmt_exposure = {"No": 1, "Mn": 2, "Av": 3, "Gd": 4}

map_functional = {
    "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
    "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8
}

cols_calidad = [
    "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond",
    "Heating QC", "Kitchen Qual", "Fireplace Qu",
    "Garage Qual", "Garage Cond"
]
for col in cols_calidad:
    data_ordinal[col] = data_ordinal[col].map(map_calidad)

data_ordinal["Pool QC"] = data_ordinal["Pool QC"].map(map_poolqc)
data_ordinal["Bsmt Exposure"] = data_ordinal["Bsmt Exposure"].map(map_bsmt_exposure)
data_ordinal["Functional"] = data_ordinal["Functional"].map(map_functional)

data_codificada = pd.get_dummies(data_ordinal, drop_first=True)

data_codificada.to_csv("../data/AmesHousing_codificada.csv", sep =",", index=False)

n_ordinales = len(cols_calidad) + 3  # las de cols_calidad + Pool QC, Bsmt Exposure, Functional

n_nominales = data_codificada.shape[1] - data_ordinal.shape[1]

tabla_transformaciones = pd.DataFrame({
    "Tipo de transformación": ["Ordinales recodificadas", "Nominales codificadas (dummies)"],
    "Cantidad de variables": [n_ordinales, n_nominales]
})

display(tabla_transformaciones)
```

**Tabla 2.3.1.** Resumen variables categóricas codificadas.

El resultado muestra que se transformaron **12 variables ordinales** mediante recodificación numérica y **139 variables nominales** a través de codificación *dummy*.  

Con estas transformaciones, el dataset quedó sin valores faltantes, sin outliers y con todas las variables en formato adecuado para su modelación.

```{code-cell} ipython3
resumen = pd.DataFrame({
    "Etapa": ["Original", "Después de limpieza"],
    "Registros": [data.shape[0], data_codificada.shape[0]],
    "Columnas": [data.shape[1], data_codificada.shape[1]],
    "Faltantes totales": [data.isna().sum().sum(), data_codificada.isna().sum().sum()]
})

resumen
```

**Key**
 Resumen genera limpieza de datos.

El conjunto original contenía 2930 registros y 82 variables, con un total de 15 749 valores faltantes.  
Tras el proceso de imputación, eliminación de outliers, transformación logarítmica y codificación de variables, el dataset final quedó compuesto por 2768 observaciones y 222 variables, sin valores ausentes.

