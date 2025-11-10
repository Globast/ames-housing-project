---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# Capitulo 8. Validación y selección de modelo (Train/Test)

```{code-cell} ipython3
from pathlib import Path
DATA_PATH = Path("../data/AmesHousing_codificada.csv")  # relativo a book/notebooks/
assert DATA_PATH.is_file(), "No se encontró '../data/AmesHousing_codificada.csv'"
print("Usando CSV:", DATA_PATH.resolve())
```

En este paso se realiza:

1. División de datos en entrenamiento (70%) y prueba (30%)  
2. Estandarización de variables (para modelos regularizados)  
3. Adición de intercepto manual  
4. Evaluación mediante métricas: R², RMSE, MAE y overfitting
```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---Dividir datos ---

data_modelo_base = pd.read_csv(DATA_PATH)

data_modelo_base = data_modelo_base[['SalePrice','Overall Qual','Gr Liv Area','Garage Area','Total Bsmt SF','1st Flr SF','Year Built','Year Remod/Add','Full Bath','Fireplaces','Mas Vnr Area','BsmtFin SF 1']]
X = data_modelo_base[['Overall Qual','Gr Liv Area','Garage Area','Total Bsmt SF','1st Flr SF','Year Built','Year Remod/Add','Full Bath','Fireplaces','Mas Vnr Area','BsmtFin SF 1']]
y = data_modelo_base[['SalePrice']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---Estandarizar variables ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Añadir intercepto ---
X_train_scaled_const = sm.add_constant(X_train_scaled)
X_test_scaled_const = sm.add_constant(X_test_scaled)

# --- Función de evaluación ---
def evaluar_modelo(modelo, X_train, y_train, X_test, y_test):
    # Ajustar modelo
    modelo_fit = modelo(y_train, X_train).fit()
    
    # Predicciones
    y_pred_train = modelo_fit.predict(X_train)
    y_pred_test = modelo_fit.predict(X_test)
    
    # Métricas entrenamiento
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    
    # Métricas prueba
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # Overfitting: diferencia entre entrenamiento y prueba
    overfit = r2_train - r2_test
    
    # Devolver resultados
    resultados = pd.DataFrame({
        'R2': [r2_train, r2_test],
        'RMSE': [rmse_train, rmse_test],
        'MAE': [mae_train, mae_test],
        'Overfitting_R2': [overfit, overfit]
    }, index=['Train', 'Test'])
    
    return resultados, y_pred_train, y_pred_test

# ---Evaluar OLS ---
resultados_eval, y_pred_train, y_pred_test = evaluar_modelo(sm.OLS, X_train_scaled_const, y_train, X_test_scaled_const, y_test)
resultados_eval
```
