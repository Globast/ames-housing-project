---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---

# Capitulo 2

## Overview


```{code-cell} ipython3
from pathlib import Path
DATA_PATH = Path("../data/ames_housing.csv")  # relativo a book/notebooks/
assert DATA_PATH.is_file(), "No se encontró '../data/ames_housing.csv'"
print("Usando CSV:", DATA_PATH.resolve())
```
```{code-cell} ipython3
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import zscore

# ===============================
# 1. Cargar dataset y metadatos
# ===============================

data = pd.read_csv("ames_housing.csv", sep = ",")
# Fuente, tamaño, número de variables, licencia
fuente = "Ames Housing Dataset (De Cock, 2011) — Iowa State University"
tamano = data.shape[0]
n_variables = data.shape[1]
licencia = "Open Data, libre uso académico"

print(f"Fuente: {fuente}")
print(f"Tamaño: {tamano} registros")
print(f"Número de variables: {n_variables}")
print(f"Licencia: {licencia}")

# ===============================
# 2. Porcentaje de valores faltantes
# ===============================

faltantes = data.isna().mean().sort_values(ascending=False) * 100
tipos = data.dtypes
tabla_faltantes = pd.DataFrame({
    "Tipo de variable": tipos,
    "% Faltantes": faltantes.round(2)
})
tabla_faltantes.head(20)
```
```{code-cell} ipython3
# ===============================
# 3. Manejo de faltantes
# ===============================
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


data_limpia = data.copy()

# Numéricas: reemplazar por la mediana
num_cols = data_limpia.select_dtypes(include=np.number).columns
for col in num_cols:
    data_limpia[col].fillna(data_limpia[col].median(), inplace=True)

# Categóricas: reemplazar por la moda
cat_cols = data_limpia.select_dtypes(exclude=np.number).columns
for col in cat_cols:
    data_limpia[col].fillna(data_limpia[col].mode()[0], inplace=True)

print("Faltantes después del tratamiento:")
print(data_limpia.isna().sum().sum())
```
```{code-cell} ipython3
# ===============================
# 4. Manejo de outliers
# ===============================
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

# Seleccionar solo variables numéricas
X = data_limpia.select_dtypes(include=np.number).copy()

# Estandarizar las variables
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Calcular matriz de covarianza e inversa
mean_vec = X_scaled.mean().values
cov_mat = np.cov(X_scaled.values, rowvar=False)
inv_covmat = np.linalg.inv(cov_mat)

# Función para calcular distancia de Mahalanobis
def distancia_mahalanobis(x, mean_vec, inv_covmat):
    diff = x - mean_vec
    return np.sqrt(np.dot(np.dot(diff, inv_covmat), diff.T))

# Calcular distancia para cada observación
distancias = X_scaled.apply(lambda row: distancia_mahalanobis(row.values, mean_vec, inv_covmat), axis=1)

# Determinar umbral con distribución chi-cuadrado
k = X_scaled.shape[1]
alpha = 0.001  # Nivel de significancia (≈ 99.9%)
umbral = np.sqrt(chi2.ppf(1 - alpha, df=k))

# Identificar y eliminar outliers
outliers_mask = distancias > umbral
antes = len(data_limpia)
despues = antes - outliers_mask.sum()
data_sin_outliers = data_limpia.loc[~outliers_mask].copy()

print(f"Registros antes: {antes}")
print(f"Outliers detectados (Mahalanobis): {outliers_mask.sum()}")
print(f"Registros después de eliminar outliers: {despues}")
```
```{code-cell} ipython3
# ===============================
# 5. Transformación logarítmica
# ===============================
# Aplicamos log a la variable objetivo y otras muy sesgadas

data_sin_outliers["SalePrice_log"] = np.log1p(data_sin_outliers["SalePrice"])
data_sin_outliers["GrLivArea_log"] = np.log1p(data_sin_outliers["Gr Liv Area"])
```
```{code-cell} ipython3
# ===============================
# 6. Transformación de variables categóricas ordinales
# ===============================
# Creamos una copia para no alterar data_sin_outliers
data_ordinal = data_sin_outliers.copy()

# ---- Mapas de conversión ----

# Escala de calidad general (Po–Ex)
map_calidad = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

# Escala de piscina (Fa–Ex)
map_poolqc = {"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}

# Exposición del sótano (No–Gd)
map_bsmt_exposure = {"No": 1, "Mn": 2, "Av": 3, "Gd": 4}

# Funcionalidad (Sal–Typ)
map_functional = {
    "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
    "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8
}

# ---- Aplicar conversiones ----

# Variables ordinales Po–Ex
cols_calidad = [
    "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond",
    "Heating QC", "Kitchen Qual", "Fireplace Qu",
    "Garage Qual", "Garage Cond"
]
for col in cols_calidad:
    data_ordinal[col] = data_ordinal[col].map(map_calidad)

# Otras ordinales
data_ordinal["Pool QC"] = data_ordinal["Pool QC"].map(map_poolqc)
data_ordinal["Bsmt Exposure"] = data_ordinal["Bsmt Exposure"].map(map_bsmt_exposure)
data_ordinal["Functional"] = data_ordinal["Functional"].map(map_functional)

print("Variables ordinales transformadas a numéricas.")

# ===============================
# 7. Codificación de variables categóricas nominales
# ===============================
# Solo las verdaderamente nominales serán convertidas en dummies
data_codificada = pd.get_dummies(data_ordinal, drop_first=True)

print("Codificación de variables nominales completada.")
print(f"Forma final del dataset: {data_codificada.shape}")
```
```{code-cell} ipython3
# ===============================
# 8. Tabla comparativa antes vs después
# ===============================

resumen = pd.DataFrame({
    "Etapa": ["Original", "Después de limpieza"],
    "Registros": [data.shape[0], data_codificada.shape[0]],
    "Columnas": [data.shape[1], data_codificada.shape[1]],
    "Faltantes totales": [data.isna().sum().sum(), data_codificada.isna().sum().sum()]
})

resumen
```
```{code-cell} ipython3
data_sin_outliers.to_csv('ames_housing_clean.csv', sep =",", index = False)
data_codificada.to_csv('ames_housing_cod.csv', sep =",", index = False)
```
## Takeaways
