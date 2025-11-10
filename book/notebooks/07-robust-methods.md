---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# Capítulo 7: Remedios y métodos robustos


```{code-cell} ipython3
from pathlib import Path
DATA_PATH = Path("../data/AmesHousing_codificada.csv")  # relativo a book/notebooks/
assert DATA_PATH.is_file(), "No se encontró '../data/AmesHousing_codificada.csv'"
print("Usando CSV:", DATA_PATH.resolve())
```

---

## **Contexto**

Cuando el supuesto de **homocedasticidad** se viola (es decir, los residuos no tienen varianza constante), los estimadores OLS siguen siendo **insesgados**, pero sus **errores estándar y valores p dejan de ser válidos**.

Para corregirlo sin modificar los coeficientes, se usan las **correcciones de varianza robustas**, conocidas como **HC0, HC1, HC2 y HC3**.

---

## **Definición general**

La matriz de varianza-covarianza robusta se define como:

$$
\text{Var}(\hat{\beta}) = (X^\top X)^{-1} \left( X^\top \Omega X \right) (X^\top X)^{-1}
$$

donde \( \Omega \) es una matriz diagonal con los residuos al cuadrado corregidos según el tipo de estimador HC.

---

## **Tipos de corrección (HC0–HC3)**

| Método |  Descripción |
|:-------|:-------------|
| **HC0** | White (1980). Asume grandes muestras. |
| **HC1** | Corrige el sesgo pequeño de muestra. |
| **HC2** | Ajusta según la influencia de cada observación. |
| **HC3** | Más conservador; recomendado con outliers. |
## **Implementacion en Python**
```{code-cell} ipython3
import statsmodels.api as sm
import pandas as pd
# Ajustar modelo OLS normal
data_modelo_base = pd.read_csv(DATA_PATH)

data_modelo_base = data_modelo_base[['SalePrice','Overall Qual','Gr Liv Area','Garage Cars','Garage Area','Total Bsmt SF','1st Flr SF','Year Built','Year Remod/Add','Full Bath','Garage Yr Blt','TotRms AbvGrd','Fireplaces','Mas Vnr Area','BsmtFin SF 1']]
X = data_modelo_base[['Overall Qual','Gr Liv Area','Garage Cars','Garage Area','Year Built','Total Bsmt SF','Year Remod/Add','1st Flr SF','Full Bath','Garage Yr Blt','Fireplaces','TotRms AbvGrd']]
y = data_modelo_base[['SalePrice']]

X = sm.add_constant(X)

modelo_base = sm.OLS(y, X).fit()

# Aplicar correcciones HC0–HC3
resultados_HC0 = modelo_base.get_robustcov_results(cov_type='HC0')
resultados_HC1 = modelo_base.get_robustcov_results(cov_type='HC1')
resultados_HC2 = modelo_base.get_robustcov_results(cov_type='HC2')
resultados_HC3 = modelo_base.get_robustcov_results(cov_type='HC3')

# Mostrar resumen comparativo
print("=== HC0 ===")
print(resultados_HC0.summary())
print("\n=== HC1 ===")
print(resultados_HC1.summary())
print("\n=== HC2 ===")
print(resultados_HC2.summary())
print("\n=== HC3 ===")
print(resultados_HC3.summary())
```
# **Resultados e interpretación**

---

Los resultados del modelo OLS mostraron heterocedasticidad (p < 0.05 en las pruebas Breusch–Pagan y White).  
Para corregir los errores estándar, se aplicaron las **correcciones robustas HC0, HC1, HC2 y HC3**, que ajustan la matriz de varianza sin modificar los coeficientes.


## **Resultados principales**

- Los **coeficientes estimados (\(\hat{\beta}\)) son idénticos** en todos los métodos → OLS sigue siendo insesgado.  
- Cambian los **errores estándar**, los **t-values** y, en menor medida, los **p-valores**.  
- Las diferencias entre HC0 y HC3 son mínimas, lo cual indica que el modelo es **robusto** frente a la heterocedasticidad leve.

---

## **Interpretación de los resultados**

| Variable | Significativa (p < 0.05)? | Comentario |
|:----------|:--------------------------|:------------|
| **OverallQual** | ✅ Sí | Mayor calidad percibida → mayor precio. |
| **GrLivArea** | ✅ Sí | Cada m² adicional de área habitable incrementa el valor. |
| **GarageCars** | ⚠️ No | El número de autos del garaje no tiene efecto fuerte controlando por área. |
| **GarageArea** | ✅ Sí | Tamaño del garaje sí afecta el precio. |
| **YearBuilt** | ✅ Sí | Casas más nuevas valen más. |
| **TotalBsmtSF** | ✅ Sí | Mayor área de sótano aumenta el precio. |
| **YearRemod/Add** | ✅ Sí | Renovaciones recientes incrementan valor. |
| **1stFlrSF** | ✅ Sí | Contribución positiva pero pequeña. |
| **FullBath** | ✅ Sí (negativa) | Posible correlación con otras variables de tamaño. |
| **GarageYrBlt** | ❌ No | No tiene efecto significativo. |
| **Fireplaces** | ✅ Sí | Mayor número de chimeneas aumenta valor. |
| **TotRmsAbvGrd** | ❌ No | Altamente correlacionada con área habitable. |

---

## **Conclusión**

- El modelo **mantiene estabilidad estadística**: los signos y la significancia no cambian entre HC0 y HC3.  
- La **heterocedasticidad fue corregida** mediante errores estándar robustos.  
- Los resultados más confiables, en presencia de posibles outliers o alta varianza, son los obtenidos con **HC3**.  
- En adelante, para reportar los coeficientes y su significancia, debe usarse la versión **OLS (HC3)**.

---
# Modelos robustos RLM con Huber y Tukey

Aquí usamos `RLM` para ajustar modelos robustos que reducen el impacto de outliers:

- **Huber**: penaliza menos outliers moderados.  
- **Tukey**: limita fuertemente el efecto de outliers extremos.  

Se comparan los coeficientes y se examinan los pesos de cada observación.

---

- Residuos pequeños → se tratan como en OLS (cuadrático).  
- Residuos grandes → penalización menor, reduciendo su efecto sobre los coeficientes.

---

## Funciones de pérdida comunes en RLM

| Función | Comportamiento | Comentario |
|:--------|:--------------|:-----------|
| **HuberT** | Cuadrática para residuos pequeños, lineal para grandes | Protege contra outliers moderados manteniendo eficiencia |
| **TukeyBiweight** | Penalización progresiva hasta eliminar la influencia de residuos extremos | Muy robusto frente a outliers, pero menos eficiente si no hay outliers |

---

## Características

- Cada observación recibe un **peso** según su residuo: residuos grandes → peso pequeño.  
- Mantiene coeficientes estables ante **outliers extremos**.  
- Ideal para datasets con heterocedasticidad leve o outliers moderados/extremos.

---

## **Implementación en Python**
```{code-cell} ipython3
# Modelos RLM
rlm_huber = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
rlm_tukey = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight()).fit()

# Comparación de coeficientes
rlm_df = pd.DataFrame({
    'OLS': modelo_base.params,
    'RLM_Huber': rlm_huber.params,
    'RLM_Tukey': rlm_tukey.params
})

# Pesos de observaciones (para análisis de outliers)
weights_df = pd.DataFrame({
    'Huber_weights': rlm_huber.weights,
    'Tukey_weights': rlm_tukey.weights
})

rlm_df, weights_df.head()
```
# **Bootstrap de coeficientes OLS**
Se realiza remuestreo bootstrap (B=1000 réplicas) para estimar la distribución
de los coeficientes sin asumir normalidad, obteniendo errores estándar y 
intervalos de confianza percentiles.
```{code-cell} ipython3
from sklearn.utils import resample
B = 1000
coef_boot = np.zeros((B, X.shape[1]))

for i in range(B):
    X_resample, y_resample = resample(X, y)
    model_bs = sm.OLS(y_resample, X_resample).fit()
    coef_boot[i, :] = model_bs.params

# Estadísticas bootstrap
coef_mean = coef_boot.mean(axis=0)
coef_se = coef_boot.std(axis=0)
ic_lower = np.percentile(coef_boot, 2.5, axis=0)
ic_upper = np.percentile(coef_boot, 97.5, axis=0)

bootstrap_df = pd.DataFrame({
    'Coef_mean': coef_mean,
    'SE_bootstrap': coef_se,
    'IC_2.5%': ic_lower,
    'IC_97.5%': ic_upper
}, index=X.columns)

bootstrap_df.to_csv('boostrap_df.csv', sep=",", index = False)
bootstrap_df
```
# 3️⃣ Resultados principales: Bootstrap

- Los coeficientes promedio son casi idénticos a OLS.
- Los **errores estándar se estiman directamente de la distribución bootstrap**, capturando asimetrías y posibles desviaciones de normalidad.
- Intervalos percentiles (2.5%-97.5%) permiten **IC robustos sin supuestos de normalidad**.
# Tabla comparativa final

Se comparan los resultados de los tres métodos principales:

- **Coeficientes estimados** (\(\hat{\beta}\))  
- **Errores estándar**: OLS, HC3 y Bootstrap  
- **Amplitud de intervalos de confianza al 95%**

Esto permite identificar discrepancias significativas y evaluar la robustez del modelo.
```{code-cell} ipython3
# Ejemplo: coeficientes OLS y errores estándar
coef_ols = modelo_base.params
se_ols = modelo_base.bse

# Errores estándar robustos HC3
se_hc3 = resultados_HC3.bse

# Bootstrap: media y SE
coef_boot_mean = bootstrap_df['Coef_mean']
se_boot = bootstrap_df['SE_bootstrap']

# Intervalos de confianza OLS y HC3 (amplitud)
ic_ols = modelo_base.conf_int().iloc[:,1] - modelo_base.conf_int().iloc[:,0]

# Para HC3, conf_int() es ndarray, así que hacemos operación directa
ic_hc3 = resultados_HC3.conf_int()[:,1] - resultados_HC3.conf_int()[:,0]

# Bootstrap: IC width
ic_boot = bootstrap_df['IC_97.5%'] - bootstrap_df['IC_2.5%']

# Crear DataFrame comparativo
comparative_df = pd.DataFrame({
    'Coef_OLS': coef_ols,
    'SE_OLS': se_ols,
    'SE_HC3': se_hc3,
    'Coef_Bootstrap': coef_boot_mean,
    'SE_Bootstrap': se_boot,
    'IC_width_OLS': ic_ols,
    'IC_width_HC3': ic_hc3,
    'IC_width_Bootstrap': ic_boot
})

comparative_df
```
```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

# Calcular variaciones porcentuales respecto al modelo base (OLS)
var_SE_HC3 = 100 * (comparative_df['SE_HC3'] - comparative_df['SE_OLS']) / comparative_df['SE_OLS']
var_SE_Boot = 100 * (comparative_df['SE_Bootstrap'] - comparative_df['SE_OLS']) / comparative_df['SE_OLS']

var_IC_HC3 = 100 * (comparative_df['IC_width_HC3'] - comparative_df['IC_width_OLS']) / comparative_df['IC_width_OLS']
var_IC_Boot = 100 * (comparative_df['IC_width_Bootstrap'] - comparative_df['IC_width_OLS']) / comparative_df['IC_width_OLS']

# Variables
variables = comparative_df.index
x = np.arange(len(variables))
width = 0.3

# Función para agregar etiquetas sobre barras
def add_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # desplazamiento vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# 1️⃣ Gráfico: Variación porcentual de errores estándar
fig, ax = plt.subplots(figsize=(12,5))
bars1 = ax.bar(x - width/2, var_SE_HC3, width, label='HC3')
bars2 = ax.bar(x + width/2, var_SE_Boot, width, label='Bootstrap')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(variables, rotation=45)
ax.set_ylabel('Variación % vs OLS')
ax.set_title('Variación porcentual de errores estándar respecto a OLS')
ax.legend()
add_labels(ax, bars1)
add_labels(ax, bars2)
plt.tight_layout()
plt.show()

# 2️⃣ Gráfico: Variación porcentual de amplitud de IC
fig, ax = plt.subplots(figsize=(12,5))
bars1 = ax.bar(x - width/2, var_IC_HC3, width, label='HC3')
bars2 = ax.bar(x + width/2, var_IC_Boot, width, label='Bootstrap')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(variables, rotation=45)
ax.set_ylabel('Variación % vs OLS')
ax.set_title('Variación porcentual de amplitud de IC 95% respecto a OLS')
ax.legend()
add_labels(ax, bars1)
add_labels(ax, bars2)
plt.tight_layout()
plt.show()
```
## Interpretación

- **Coeficientes:** Todos los métodos muestran valores muy similares → OLS sigue siendo insesgado.  
- **Errores estándar:** HC3 y Bootstrap suelen ser más conservadores que OLS clásico.  
- **Intervalos de confianza:** Bootstrap e IC HC3 son más amplios, reflejando mayor incertidumbre frente a outliers o heterocedasticidad.  

💡 **Conclusión:**  
- Diferencias entre métodos son pequeñas → modelo robusto frente a heterocedasticidad leve y outliers moderados.  
- Para decisiones conservadoras (p-values o IC), se recomienda **HC3 o Bootstrap**.
