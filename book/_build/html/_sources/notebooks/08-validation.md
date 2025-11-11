---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# Capítulo 8 — Validación y selección de modelos

> **Overview**:
Este capítulo evalúa generalización con partición train/test y métricas (R², RMSE, MAE). Implementa Ridge y Lasso con escalamiento y búsqueda de 
𝛼; compara OLS vs. regularizados para estabilidad y sesgo-varianza. 


**Definir ruta de datos**

```{code-cell} ipython3
from pathlib import Path

# Rutas deterministas: este archivo se ejecuta desde su propia carpeta (p.ej. book/notebooks/)
DATA_PATH = Path("../data/AmesHousing_codificada.csv")  # relativo a book/notebooks/
assert DATA_PATH.is_file(), "No se encontró '../data/AmesHousing_codificada.csv'"
print("Usando CSV:", DATA_PATH.resolve())
```
## Validación train/test

Al evaluar un modelo de regresión, es fundamental medir qué tan bien predice datos que no ha visto durante su entrenamiento. Para ello, se suele dividir el conjunto de datos en dos partes: **entrenamiento (train)** y **prueba (test)**, donde el primero se utiliza para ajustar los parámetros del modelo y aprender patrones en los datos, y el de prueba se reserva para evaluar la capacidad de generalización del modelo, es decir, qué tan bien puede predecir datos nuevos.  

De esta forma es posible detectar problemas como el **sobreajuste** (overfitting), donde el modelo se ajusta demasiado a los datos de entrenamiento y falla al generalizar.

Para cuantificar este desempeño se utilizan métricas de error y de ajuste. Las más comunes son **R²**, **RMSE** y **MAE**.

El **coeficiente de determinación (R²)** mide la proporción de la varianza de la variable dependiente \(y\) que es explicada por el modelo:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$
**Ecuación 8.1.1.** Coeficiente de determinación.

Un R² cercano a 1 indica un buen ajuste, mientras que un valor cercano a 0 indica que el modelo no explica la variabilidad de los datos.

Por su parte, la **raíz del error cuadrático medio (RMSE)** se interpreta como el error promedio en las mismas unidades que la variable \(y\), con mayor sensibilidad a errores grandes:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$
**Ecuación 8.1.2.** Raíz del error cuadrático medio.

Esta métrica penaliza más los errores grandes.

El **error absoluto medio (MAE)** es el promedio de los errores absolutos, es decir, indica en promedio cuánto se desvía la predicción de los valores reales:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$
**Ecuación 8.1.3.** Error absoluto medio.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import statsmodels.api as sm

data_modelo_base = pd.read_csv(DATA_PATH)

data_modelo_base = data_modelo_base[['SalePrice_log', 'Overall Qual', 'Gr Liv Area',
                                     'Garage Cars', 'Total Bsmt SF', '1st Flr SF',
                                     'Full Bath', 'Year Built', 'Fireplaces', 'Lot Area']]

X = data_modelo_base[['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF',
                      '1st Flr SF', 'Full Bath', 'Year Built', 'Fireplaces', 'Lot Area']]
y = data_modelo_base[['SalePrice_log']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_const = sm.add_constant(X_train_scaled)
X_test_scaled_const = sm.add_constant(X_test_scaled)

def evaluar_modelo(modelo, X_train, y_train, X_test, y_test):
    modelo_fit = modelo(y_train, X_train).fit()
    
    y_pred_train = modelo_fit.predict(X_train)
    y_pred_test = modelo_fit.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    overfit = r2_train - r2_test

    resultados = pd.DataFrame({
        'Conjunto': ['Entrenamiento', 'Prueba'],
        'R2': [r2_train, r2_test],
        'RMSE': [rmse_train, rmse_test],
        'MAE': [mae_train, mae_test],
        'Overfitting_R2': [overfit, overfit]
    })
    
    return resultados, y_pred_train, y_pred_test

resultados_eval_ols, y_pred_train, y_pred_test = evaluar_modelo(
    sm.OLS, X_train_scaled_const, y_train, X_test_scaled_const, y_test
)

display(resultados_eval_ols)
```

**Tabla 8.1.1.** Validación train/test modelo 1.

Los resultados indican que, en el conjunto de entrenamiento, el modelo es capaz de explicar aproximadamente un 84.8% de la variabilidad total de los datos. En el conjunto de prueba, este valor aumenta ligeramente a 85.5%, lo que sugiere que el modelo tiene una buena capacidad de generalización y no está sobreajustado a los datos de entrenamiento.

En cuanto a los errores, tanto el RMSE como el MAE presentan valores muy similares entre entrenamiento y prueba. Esto indica que el modelo mantiene un buen nivel de precisión al predecir datos no vistos.

## Regresión Ridge

La **regresión Ridge** es una extensión de la regresión lineal que incorpora un término de regularización L2 para controlar el sobreajuste y estabilizar los coeficientes cuando existe **colinealidad** entre las variables predictoras.

La función de pérdida de Ridge modifica la función de mínimos cuadrados para incluir la penalización sobre los coeficientes:

$$
S_{\text{Ridge}}(\beta) = (Y - X \beta)^\top (Y - X \beta) + \alpha \, \beta^\top \beta
$$

**Ecuación 8.2.1.** Función de pérdida Ridge.

Donde $ \alpha \ge 0 $ es el hiperparámetro de regularización.

Al introducir esta penalización, se **reduce la varianza de los coeficientes**, evitando que tomen valores excesivamente grandes y mejorando la **capacidad de generalización** del modelo. Si $ \alpha = 0 $, esta solución se reduce a la regresión lineal ordinaria ([Ecuación 4.1.1](#ecuacion-411-regresion-lineal)).

```{code-cell} ipython3
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

data = pd.read_csv(DATA_PATH)

vars_modelo = [
    'Overall Qual', 'Gr Liv Area', 'Garage Cars',
    'Total Bsmt SF', '1st Flr SF', 'Full Bath',
    'Year Built', 'Fireplaces', 'Lot Area'
]

X = data[vars_modelo].values
y = data['SalePrice_log'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_int, X_val, y_train_int, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)
print(f"Tamaños -> Train: {X_train_int.shape[0]}, Validación: {X_val.shape[0]}, Test final: {X_test.shape[0]}")
```

**Tabla 8.2.1.** Tamaños validación Ridge.

El conjunto de entrenamiento cuenta con 1 549 observaciones y se utiliza para ajustar los parámetros del modelo y aprender los patrones de los datos; el conjunto de validación tiene 388 observaciones y permite ajustar hiperparámetros y evaluar el desempeño del modelo durante el entrenamiento sin utilizar los datos de prueba; el conjunto de prueba final posee 831 observaciones y se reserva exclusivamente para evaluar la capacidad de generalización del modelo, es decir, su desempeño en datos que no han sido vistos previamente, asegurando así un flujo de trabajo robusto y confiable.

```{code-cell} ipython3
alphas = np.logspace(-3, 3, 13)
resultados = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train_int, y_train_int)
  
    y_val_pred = ridge.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae_val = mean_absolute_error(y_val, y_val_pred)
    
    resultados.append((alpha, r2_val, rmse_val, mae_val))

resultados_df = pd.DataFrame(resultados, columns=["alpha", "R2_val", "RMSE_val", "MAE_val"])
display(resultados_df)
```

**Tabla 8.2.2.** Evaluación de valores de penalización Ridge.

Se observa que para los valores pequeños de $ \alpha\ $ probados (0.001 a 10), las métricas de validación son prácticamente constantes, con un **R²** cercano a 0.8857, un **RMSE** alrededor de 0.1239 y un **MAE** cerca de 0.0944. El mejor desempeño se alcanza con el alpha mínimo evaluado, **0.001**, lo que indica que el modelo obtiene un buen ajuste incluso con **regularización casi nula**, es decir, prácticamente sin penalización sobre los coeficientes. A medida que la penalización aumenta ($ \alpha \ge 100 $), el desempeño comienza a deteriorarse, reflejando que la regularización excesiva provoca que el modelo pierda capacidad para capturar la variabilidad de los datos.

```{code-cell} ipython3
best_row = resultados_df.loc[resultados_df["R2_val"].idxmax()]
best_alpha = best_row["alpha"]
ridge_final = Ridge(alpha=best_alpha, random_state=42)
ridge_final.fit(X_train_scaled, y_train)

y_pred_train = ridge_final.predict(X_train_scaled)
y_pred_test = ridge_final.predict(X_test_scaled)

resultados_eval = pd.DataFrame({
    "Conjunto": ["Entrenamiento", "Prueba"],
    "R2": [r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)],
    "RMSE": [
        np.sqrt(mean_squared_error(y_train, y_pred_train)),
        np.sqrt(mean_squared_error(y_test, y_pred_test))
    ],
    "MAE": [
        mean_absolute_error(y_train, y_pred_train),
        mean_absolute_error(y_test, y_pred_test)
    ]
})

overfitting_r2 = abs(resultados_eval.loc[0, "R2"] - resultados_eval.loc[1, "R2"])
resultados_eval["Overfitting_R2"] = [overfitting_r2, overfitting_r2]
resultados_eval_ridge = resultados_eval.copy()
display(resultados_eval_ridge)
```

**Tabla 8.2.3.** Resumen evaluación Ridge.

El modelo muestra un desempeño sólido tanto en el conjunto de entrenamiento como en el de prueba. En el conjunto de entrenamiento, el **R²** es 0.8485, con un **RMSE** de 0.1462 y un **MAE** de 0.1024, mientras que en el conjunto de prueba el **R²** aumenta ligeramente a 0.8554, con un **RMSE** de 0.1454 y un **MAE** de 0.1055. Además, la diferencia absoluta entre el R² de entrenamiento y prueba es muy baja (0.00699), lo que indica que el modelo generaliza de manera adecuada a datos no vistos.

## Regresión Lasso

```{code-cell} ipython3
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

data = pd.read_csv(DATA_PATH)

vars_modelo = [
    'Overall Qual', 'Gr Liv Area', 'Garage Cars',
    'Total Bsmt SF', '1st Flr SF', 'Full Bath',
    'Year Built', 'Fireplaces', 'Lot Area'
]

X = data[vars_modelo].values
y = data['SalePrice_log'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_int, X_val, y_train_int, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)
print(f"Tamaños -> Train interno: {X_train_int.shape[0]}, Validación: {X_val.shape[0]}, Test final: {X_test.shape[0]}")
```

**Tabla 8.3.1.** Tamaños validación Lasso.

```{code-cell} ipython3
alphas = np.logspace(-3, 0, 10)  # 10^-3 a 10^0
resultados = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train_int, y_train_int)
    
    y_val_pred = lasso.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae_val = mean_absolute_error(y_val, y_val_pred)
    
    resultados.append((alpha, r2_val, rmse_val, mae_val))

resultados_df = pd.DataFrame(resultados, columns=["alpha", "R2_val", "RMSE_val", "MAE_val"])
display(resultados_df)
```

**Tabla 8.3.2.** Evaluación de valores de penalización Lasso.

Al evaluar el modelo Lasso, se observa que para valores pequeños de $ \alpha\ $ (0.001 a 0.01), las métricas de validación son óptimas, con un **R²** cercano a 0.8857, un **RMSE** alrededor de 0.1239 y un **MAE** cerca de 0.0943, indicando un ajuste sólido a los datos de validación.  

A medida que alpha aumenta ($ \alpha \ge 0.02 $), el desempeño del modelo comienza a deteriorarse: R² disminuye, mientras que RMSE y MAE aumentan, de forma que Lasso alcanza su mejor desempeño prácticamente sin regularización, similar a lo observado con Ridge, y que valores grandes de alpha producen subajuste.

```{code-cell} ipython3

best_row = resultados_df.loc[resultados_df["R2_val"].idxmax()]
best_alpha = best_row["alpha"]

lasso_final = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
lasso_final.fit(X_train_scaled, y_train)

coeficientes = pd.DataFrame({
    "Variable": vars_modelo,
    "Coeficiente": lasso_final.coef_
})
coeficientes["Es_cero"] = coeficientes["Coeficiente"] == 0
display(coeficientes)
```

**Tabla 8.3.3.** Coeficientes Lasso.

Se muestra que la regresión Lasso ha reducido algunos coeficientes a cero, lo que refleja su capacidad de realizar selección automática de variables mediante la regularización L1. Específicamente, las variables `1st Flr SF` y `Full Bath` tienen coeficiente 0, lo que indica que el modelo las considera no relevantes para predecir `SalePrice_log` en presencia de las otras variables.

```{code-cell} ipython3

y_pred_train = lasso_final.predict(X_train_scaled)
y_pred_test = lasso_final.predict(X_test_scaled)

resultados_eval = pd.DataFrame({
    "Conjunto": ["Entrenamiento", "Prueba"],
    "R2": [r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)],
    "RMSE": [np.sqrt(mean_squared_error(y_train, y_pred_train)),
             np.sqrt(mean_squared_error(y_test, y_pred_test))],
    "MAE": [mean_absolute_error(y_train, y_pred_train),
            mean_absolute_error(y_test, y_pred_test)]
})

overfitting_r2 = abs(resultados_eval.loc[0, "R2"] - resultados_eval.loc[1, "R2"])
resultados_eval["Overfitting_R2"] = [overfitting_r2, overfitting_r2]

resultados_eval_lasso = resultados_eval.copy()
display(resultados_eval_lasso)
```

**Tabla 8.3.4.** Resumen evaluación Lasso.

En el **conjunto de entrenamiento**, el **R²** es 0.8479, con un **RMSE** de 0.1465 y un **MAE** de 0.1024. En el **conjunto de prueba**, el **R²** aumenta ligeramente a 0.8566, con un **RMSE** de 0.1448 y un **MAE** de 0.1053. La diferencia absoluta entre el R² de entrenamiento y prueba, es baja (0.0087), lo que indica que el modelo **no presenta sobreajuste significativo** y generaliza correctamente a datos no vistos.  

En conjunto, estas métricas confirman que el modelo Lasso logra un ajuste consistente y preciso, con la ventaja adicional de haber reducido algunos coeficientes a cero, mejorando la interpretabilidad del modelo.

## OLS vs. Ridge vs. Lasso

```{code-cell} ipython3
comparativa = pd.DataFrame({
    "Modelo": ["OLS", "Ridge", "Lasso"],
    "R2_Test": [
        resultados_eval_ols.loc[resultados_eval_ols["Conjunto"] == "Prueba", "R2"].values[0],
        resultados_eval_ridge.loc[resultados_eval_ridge["Conjunto"] == "Prueba", "R2"].values[0],
        resultados_eval_lasso.loc[resultados_eval_lasso["Conjunto"] == "Prueba", "R2"].values[0]
    ],
    "RMSE_Test": [
        resultados_eval_ols.loc[resultados_eval_ols["Conjunto"] == "Prueba", "RMSE"].values[0],
        resultados_eval_ridge.loc[resultados_eval_ridge["Conjunto"] == "Prueba", "RMSE"].values[0],
        resultados_eval_lasso.loc[resultados_eval_lasso["Conjunto"] == "Prueba", "RMSE"].values[0]
    ],
    "MAE_Test": [
        resultados_eval_ols.loc[resultados_eval_ols["Conjunto"] == "Prueba", "MAE"].values[0],
        resultados_eval_ridge.loc[resultados_eval_ridge["Conjunto"] == "Prueba", "MAE"].values[0],
        resultados_eval_lasso.loc[resultados_eval_lasso["Conjunto"] == "Prueba", "MAE"].values[0]
    ],
    "Overfitting_R2": [
        resultados_eval_ols["Overfitting_R2"].values[0],
        resultados_eval_ridge["Overfitting_R2"].values[0],
        resultados_eval_lasso["Overfitting_R2"].values[0]
    ]
})

display(
    comparativa.style
    .format({
        "R2_Test": "{:.4f}",
        "RMSE_Test": "{:.4f}",
        "MAE_Test": "{:.4f}",
        "Overfitting_R2": "{:.4f}"
    })
)
```

**Tabla 8.4.1.** Métricas de evaluación OLS vs. Ridge vs. Lasso.

Todos los modelos presentan valores de R² muy similares, alrededor de 0.855–0.857, indicando que los tres explican aproximadamente el 85–86% de la variabilidad de los datos de prueba. Los errores de predicción son también comparables entre los modelos, con RMSE ≈ 0.145 y MAE ≈ 0.105. 

En el caso de estudio Lasso logra el valor ligeramente más bajo en ambas métricas, sugiriendo un ajuste marginalmente más preciso. Además, simplifica el modelo al eliminar variables menos relevantes, mejorando la interpretabilidad sin sacrificar precisión.

> **Key takeaways**
>- Buen desempeño out-of-sample y sin sobreajuste marcado en el modelo base.
>- Regularización estabiliza coeficientes; exceso de 𝛼 degrada ajuste.
>- Seleccionar por validación (no solo por ajuste en entrenamiento).