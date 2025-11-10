---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---

# Capitulo 2

**Overview.** Comparar un pipeline de preprocesamiento para housing que: (1) carga y diagnostica el CSV crudo, (2) **imputa faltantes** (mediana/moda), (3) **filtra outliers multivariados** vía distancia de **Mahalanobis** con umbral \(\chi^2\), (4) aplica **transformaciones logarítmicas** para estabilizar varianza, (5) **codifica** variables (ordinal + _one-hot_) y (6) **exporta** dos artefactos reproducibles para modelado:
- `../data/ames_housing_clean.csv` → datos limpios **sin outliers** y con columnas log.
- `../data/ames_housing_cod.csv` → versión **codificada** (ordinal + dummies) lista para modelos lineales.

> El capítulo se ejecuta desde `book/notebooks/`. Por eso las rutas apuntan a `../data/...`.

## Rutas y utilidades

```{code-cell} ipython3
from pathlib import Path

# Directorio de trabajo esperado: book/notebooks/
DATA_IN = Path("../data/ames_housing.csv")
DATA_OUT_CLEAN = Path("../data/ames_housing_clean.csv")
DATA_OUT_COD = Path("../data/ames_housing_cod.csv")

assert DATA_IN.is_file(), "No se encontró el archivo de entrada '../data/ames_housing.csv'. Verifica la ruta relativa desde book/notebooks/."

DATA_IN, DATA_OUT_CLEAN, DATA_OUT_COD
```

## Importaciones

```{code-cell} ipython3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# SciPy es útil para el umbral chi-cuadrado; incluimos alternativa por si no está disponible.
try:
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
```

## 1) Carga y diagnóstico rápido

```{code-cell} ipython3
df = pd.read_csv(DATA_IN)

n_rows, n_cols = df.shape
missing_pct = df.isna().mean().sort_values(ascending=False)

diagnostico = {
    "filas": n_rows,
    "columnas": n_cols,
    "columnas_con_faltantes": int((missing_pct > 0).sum()),
    "porcentaje_faltantes_max": float(missing_pct.max() if n_cols else 0)*100,
}

diagnostico, missing_pct.head(10)
```

**Descripción:** Leemos el CSV crudo, contamos filas y columnas y calculamos el porcentaje de `NaN` por variable.

## 2) Imputación de valores faltantes (mediana/moda)

```{code-cell} ipython3
df_imputed = df.copy()

num_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_imputed.select_dtypes(exclude=[np.number]).columns.tolist()

# Numéricas → mediana
for c in num_cols:
    if df_imputed[c].isna().any():
        df_imputed[c] = df_imputed[c].fillna(df_imputed[c].median())

# Categóricas → moda
for c in cat_cols:
    if df_imputed[c].isna().any():
        moda = df_imputed[c].mode(dropna=True)
        if len(moda) > 0:
            df_imputed[c] = df_imputed[c].fillna(moda.iloc[0])
        else:
            df_imputed[c] = df_imputed[c].fillna("Missing")

df_imputed.isna().sum().sum()
```

**Descripción:** Reemplazamos `NaN` numéricos por la **mediana** y categóricos por la **moda**. El resultado debe quedar sin faltantes.

## 3) Outliers multivariados con distancia de Mahalanobis

```{code-cell} ipython3
df_maha = df_imputed.copy()

# Usamos solo columnas numéricas para Mahalanobis
X = df_maha[num_cols].astype(float).to_numpy()
mu = X.mean(axis=0)
Xc = X - mu

# Covarianza y su inversa (pseudoinversa para mayor estabilidad numérica)
cov = np.cov(Xc, rowvar=False)
cov_inv = np.linalg.pinv(cov)

# Distancia de Mahalanobis al cuadrado: d^2 = x' S^{-1} x
d2 = np.einsum("ij,jk,ik->i", Xc, cov_inv, Xc)

# Umbral Chi^2 con p grados de libertad y alfa=0.999 (robusto). Si no hay SciPy, usamos cuantiles empíricos.
p = X.shape[1]
if SCIPY_AVAILABLE:
    thr = chi2.ppf(0.999, df=p)
else:
    thr = float(np.quantile(d2, 0.999))

mask_inliers = d2 <= thr
outliers_count = int((~mask_inliers).sum())

df_clean = df_maha.loc[mask_inliers].reset_index(drop=True)
outliers_count, df_clean.shape
```

**Descripción:** Calculamos \(d^2\) de Mahalanobis con pseudoinversa para evitar problemas de multicolinealidad. Eliminamos observaciones con \(d^2\) por encima del umbral \(\chi^2\) al 99.9%.

## 4) Transformaciones logarítmicas (estabilizar varianza)

```{code-cell} ipython3
df_log = df_clean.copy()

# Algunas bases Ames nombran 'GrLivArea' y otras 'Gr Liv Area'.
def pick_first_existing(cols, candidates):
    for name in candidates:
        if name in cols:
            return name
    return None

sale_col = pick_first_existing(df_log.columns, ["SalePrice", "Sale Price"])
grliv_col = pick_first_existing(df_log.columns, ["GrLivArea", "Gr Liv Area"])

# Aplicamos log1p si existen (evita log(0))
if sale_col is not None:
    df_log[sale_col + "_log"] = np.log1p(df_log[sale_col])

if grliv_col is not None:
    df_log[grliv_col + "_log"] = np.log1p(df_log[grliv_col])

sale_col, grliv_col, [c for c in df_log.columns if c.endswith("_log")]
```

**Descripción:** Creamos columnas `_log` para precios y área habitable si están presentes.

## 5) Codificación: ordinal + one-hot

```{code-cell} ipython3
df_enc = df_log.copy()

# Mapeos ordinales típicos en Ames; se aplican solo si la columna existe.
qual_map = {"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
exposure_map = {"No":1, "Mn":2, "Av":3, "Gd":4}
bsmtfin_map = {"Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5, "GLQ":6}
functional_map = {"Sal":1, "Sev":2, "Mod":3, "Min2":4, "Min1":5, "Typ":6}

ordinal_candidates = {
    "Exter Qual": qual_map, "Exter Cond": qual_map,
    "Bsmt Qual": qual_map, "Bsmt Cond": qual_map, "Bsmt Exposure": exposure_map,
    "Kitchen Qual": qual_map, "Fireplace Qu": qual_map,
    "Garage Qual": qual_map, "Garage Cond": qual_map,
    "Pool QC": qual_map, "Functional": functional_map,
    "BsmtFin Type 1": bsmtfin_map, "BsmtFin Type 2": bsmtfin_map,
    # variantes sin espacio:
    "ExterQual": qual_map, "ExterCond": qual_map,
    "BsmtQual": qual_map, "BsmtCond": qual_map, "BsmtExposure": exposure_map,
    "KitchenQual": qual_map, "FireplaceQu": qual_map,
    "GarageQual": qual_map, "GarageCond": qual_map,
    "PoolQC": qual_map, "Functional": functional_map,
    "BsmtFinType1": bsmtfin_map, "BsmtFinType2": bsmtfin_map,
}

for col, mapping in ordinal_candidates.items():
    if col in df_enc.columns:
        df_enc[col] = df_enc[col].map(mapping).astype("float")

# Identificamos categóricas después del mapeo (las que siguen siendo object o category)
cat_after = df_enc.select_dtypes(include=["object", "category"]).columns.tolist()

df_dum = pd.get_dummies(df_enc, columns=cat_after, drop_first=True)

df_log.shape, df_dum.shape
```

**Descripción:** Mapeamos ordinales (si existen) y luego aplicamos _one-hot_ al resto (con `drop_first=True` para evitar colinealidad perfecta).

## 6) Exportar artefactos

```{code-cell} ipython3
# Guardamos 'clean' (sin outliers + con columnas _log, antes de dummies)
df_log.to_csv(DATA_OUT_CLEAN, index=False)

# Guardamos 'cod' (lista para modelado)
df_dum.to_csv(DATA_OUT_COD, index=False)

DATA_OUT_CLEAN.exists(), DATA_OUT_COD.exists(), (df_log.shape, df_dum.shape)
```

**Descripción:** Escribimos `../data/ames_housing_clean.csv` y `../data/ames_housing_cod.csv`. Los tamaños deben coincidir con lo reportado arriba.

## Takeaways (para modelado lineal)

- **Imputación simple (mediana/moda)** reduce varianza de estimadores frente a _listwise deletion_ y mantiene el tamaño muestral.
- **Mahalanobis + umbral \(\chi^2\)** proporciona una regla **multivariada** (no univariante) para outliers; usar **pseudoinversa** mejora estabilidad con multicolinealidad.
- **Transformaciones logarítmicas** (p. ej., `SalePrice_log`) ayudan a aproximar **normalidad de residuos** y **homocedasticidad**, claves para OLS.
- **Codificación ordinal** preserva el **orden semántico** de calidades; _one-hot_ con `drop_first=True` evita **trampas de la variable ficticia**.
- Exportar **dos capas** (clean vs codificada) agiliza la comparación entre **OLS** y **métodos robustos** en capítulos posteriores.
