---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# Capítulo 7: Remedios y métodos robustos

## Overview
Se exploran estimadores robustos (Huber/RLM, Regresión Cuantílica) y procedimientos de validación. Incluye una sección de **Bootstrap** para evaluar la variabilidad de los estimadores y comparar OLS, errores estándar HC3 y percentiles bootstrap.


```{code-cell} ipython3
from pathlib import Path
import pandas as pd

DATA_PATH = Path("../data/AmesHousing_codificada.csv")
assert DATA_PATH.is_file(), f"No se encontró '{DATA_PATH}'"
print("Usando CSV base:", DATA_PATH.resolve())

df = pd.read_csv(DATA_PATH)
df.shape
```


### 7.1 Correcciones para heterocedasticidad

De acuerdo con el supuesto de homocedasticidad ([Ecuación 6.2.1](#ecuacion-621-varianza-errores)), la presencia de heterocedasticidad puede provocar que los errores estándar de los coeficientes estén subestimados, afectando los valores $t$ y las decisiones de significancia. Para corregir este problema, se utilizan los estimadores de varianza robusta **HC** (Heteroscedasticity-Consistent), que ajustan los errores estándar sin cambiar los coeficientes estimados. 

- **HC0**: Estimador original de White, básico y consistente frente a heterocedasticidad.  
- **HC1**: Ajusta HC0 por los grados de libertad, corrigiendo ligeras subestimaciones.  
- **HC2**: Considera los *leverage points* de cada observación, dando más peso a observaciones influyentes.  
- **HC3**: Aproximación tipo *jackknife*, más conservadora y recomendada en muestras pequeñas por ofrecer errores estándar más cautelosos.

**Tabla 7.1.1.** Errores estándar OLS vs. HC0-HC3.

Se observa que los errores estándar aumentan ligeramente cuando se aplican las correcciones HC, especialmente para variables como `Overall Qual`, `Garage Cars` y `Full Bath`. Esto indica que los estimadores OLS originales podrían subestimar la variabilidad de los coeficientes si existe heterocedasticidad.

Por ejemplo, el coeficiente de `Overall Qual` tiene un error estándar de 0.00333 bajo OLS clásico, que se incrementa a 0.00447 bajo HC3, la corrección más conservadora. De manera similar, `Garage Cars` pasa de 0.00530 a 0.00687.  

Las variables con cambios mínimos en los errores estándar (como `Gr Liv Area` o `Lot Area`) sugieren que su variabilidad está poco afectada por heterocedasticidad.

**Tabla 7.1.2.** Intervalos de confianza OLS vs. HC0-HC3.

Se puede observar que los intervalos de confianza se ensanchan ligeramente cuando se aplican las correcciones HC, reflejando un aumento en la incertidumbre de los coeficientes. Por ejemplo, el coeficiente de `Overall Qual` tiene un intervalo de confianza de [0.095, 0.108] bajo OLS, que se amplía a [0.0929, 0.1104] con HC3. Lo mismo ocurre con `Garage Cars` y `Full Bath`, indicando que las inferencias de estos coeficientes son sensibles a la heterocedasticidad.

En contraste, intervalos de confianza prácticamente inalterados, como los de `Gr Liv Area` o `Lot Area`, sugieren que la heterocedasticidad tiene poco efecto sobre la precisión de estos coeficientes.  

En general, aplicar correcciones HC permite obtener intervalos de confianza más robustos, proporcionando inferencias más conservadoras y fiables cuando se sospecha heterocedasticidad.

### 7.2 Modelos robustos con funciones Huber/Tukey

Los **modelos de regresión robusta** son una extensión de la regresión lineal ordinaria diseñada para reducir la influencia de outliers o valores atípicos en la estimación de los coeficientes. Mientras que la regresión OLS pondera todos los residuales por igual, los modelos robustos asignan **menor peso a los residuales grandes**, permitiendo obtener estimaciones más confiables y estables.

Existen diversas funciones de pérdida que determinan cómo se penalizan los residuales, donde cada una equilibra de distinta manera la eficiencia para valores centrales y la robustez frente a outliers extremos:

- **Huber / TukeyHuberT()**:

$$
\rho(r) = 
\begin{cases} 
\frac{1}{2} r^2 & \text{si } |r| \le \delta \\[2mm]
\delta \left(|r| - \frac{1}{2}\delta \right) & \text{si } |r| > \delta
\end{cases}
$$  
**Ecuación 7.2.1.** Función de pérdida Huber.

Protege contra outliers moderados manteniendo eficiencia para valores centrales.

- **Tukey Biweight**:

$$
\rho(r) = 
\begin{cases}
\frac{c^2}{6} \left[ 1 - \left(1 - \left(\frac{r}{c}\right)^2 \right)^3 \right] & \text{si } |r| \le c \\[1mm]
\frac{c^2}{6} & \text{si } |r| > c
\end{cases}
$$  
**Ecuación 7.2.2.** Función de pérdida Tukey.

Limita el impacto de observaciones lejanas sin eliminarlas completamente.

**Tabla 7.2.1.** Coeficientes OLS vs. RLM. *Valores en escala logarítmica.*

Se observa que el intercepto (`const`) aumenta al usar modelos robustos, lo que refleja que los outliers tienden a sesgar hacia abajo la estimación en OLS. Por su parte, variables como `Overall Qual` y `Fireplaces` muestran coeficientes algo menores en modelos robustos, indicando que su efecto estaba ligeramente sobreestimado por la presencia de outliers. 

Para la mayoría de las demás variables (`Gr Liv Area`, `Garage Cars`, `Total Bsmt SF`, `Year Built`), las diferencias son pequeñas, lo que sugiere que los outliers no tienen un impacto fuerte en estas estimaciones.

**Tabla 7.2.2.** Pesos outliers Huber vs. Tukey.

La mayoría de los pesos son cercanos a 1 en ambos modelos robustos, lo que indica que la mayoría de las observaciones se ajusta bien al modelo y tiene plena influencia en la estimación de los coeficientes. 

Sin embargo, algunas observaciones presentan pesos menores, como la última fila, con Huber = 0.69 y Tukey = 0.68, lo que refleja que su residuo es relativamente grande y su efecto en el ajuste se atenúa.

Además, es evidente que Tukey aplica un castigo más fuerte a residuales extremos.

### 7.3 Bootstrap

El **bootstrap** es un método de remuestreo que permite estimar la variabilidad de los coeficientes de un modelo sin asumir una distribución específica de los errores. Consiste en generar múltiples muestras con reemplazo a partir de los datos originales y recalcular los estimadores para cada réplica, obteniendo así una **distribución empírica** de los coeficientes, a partir de la cual se calculan el error estándar y los intervalos de confianza.

**Tabla 7.3.1.** Resumen Bootstrap.

Se observa que variables como `Overall Qual`, `Gr Liv Area`, `Garage Cars` y `Year Built` tienen coeficientes claramente distintos de cero, con intervalos de confianza estrechos y consistentes, lo que sugiere estimaciones robustas y estables. Por el contrario, `1st Flr SF` y `Full Bath` presentan intervalos que incluyen el cero, indicando que su efecto sobre la variable respuesta podría no ser significativo.

### 7.4 OLS vs. HC3 vs. Bootstrap

**Tabla 7.4.1.** Coeficientes OLS vs. Bootstrap. *Valores en escala logarítmica.*

Se observa que la estimación de los parámetros es muy estable frente al remuestreo, lo que sugiere que la muestra utilizada es suficientemente representativa y que los coeficientes no dependen excesivamente de observaciones individuales.

En particular, las variables como `Overall Qual`, `Gr Liv Area` y `Fireplaces` muestran coeficientes positivos consistentes en ambos métodos, confirmando su relación directa con el precio de la vivienda. Por su parte, `Full Bath` mantiene un coeficiente ligeramente negativo, indicando que, controlando por las demás variables, su efecto es mínimo.

**Tabla 7.4.2.** Errores estándar OLS vs. HC3 vs. Bootstrap.<a id="tabla-742-se-ols-hc3-boot"></a>

Los errores estándar de OLS son generalmente menores que los obtenidos mediante HC3 o bootstrap, lo que sugiere que este modelo inicial podría subestimar la incertidumbre cuando hay heterocedasticidad presente o dependiendo de la muestra seleccionada.  

El método HC3, diseñado para ser más conservador frente a heterocedasticidad y leverage points, produce errores estándar ligeramente mayores que OLS, especialmente para variables como `Overall Qual`, `Garage Cars` y `Full Bath`. El bootstrap refleja un patrón muy similar al de HC3, confirmando la estabilidad de las estimaciones.

Es notable que las variables con errores estándar relativamente bajos (`Gr Liv Area`, `Lot Area`) están estimadas con gran precisión, mientras que aquellas con errores más altos (`Full Bath`, `Garage Cars`) presentan mayor incertidumbre en la estimación de su efecto sobre el precio de la vivienda.

**Tabla 7.4.3.** Amplitud intervalos de confianza OLS vs. HC3 vs. Bootstrap.

Alineados con los errores estándar ([Tabla 7.4.2](#tabla-742-se-ols-hc3-boot)), los intervalos calculados con OLS son consistentemente más estrechos que los obtenidos con HC3 o bootstrap, llegando a la misma conclusión de que el OLS inicial es menos robusto.


## 7.4 OLS vs. HC3 vs. Bootstrap


```{code-cell} ipython3
# Ruta canónica para resultados de Bootstrap (guardado/lectura)
from pathlib import Path
BOOT_CSV_PATH = Path("../data/bootstrap_results.csv")
print("Archivo Bootstrap:", BOOT_CSV_PATH.resolve())
```

```{code-cell} ipython3
from pathlib import Path
import pandas as pd

# Definir ruta de datos relativa al capítulo (ejecutado desde book/notebooks/)
DATA_PATH = Path("../data/AmesHousing_codificada.csv")
assert DATA_PATH.is_file(), f"No se encontró '{DATA_PATH}'"
print("Usando CSV base:", DATA_PATH.resolve())

# Lectura canónica a reutilizar en el capítulo
df = pd.read_csv(DATA_PATH)
df.shape
```
```{code-cell} ipython3
import statsmodels.api as sm
import pandas as pd

pd.set_option('display.float_format', '{:.6f}'.format)

data_modelo_base = pd.read_csv(DATA_PATH)

data_modelo_base = data_modelo_base[['SalePrice_log', 'Overall Qual', 'Gr Liv Area', 
                                     'Garage Cars', 'Total Bsmt SF', '1st Flr SF', 
                                     'Full Bath', 'Year Built', 'Fireplaces', 'Lot Area']]
X = data_modelo_base[['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF',
                      '1st Flr SF', 'Full Bath', 'Year Built', 'Fireplaces', 'Lot Area']]
y = data_modelo_base[['SalePrice_log']]

X = sm.add_constant(X)

modelo_base = sm.OLS(y, X).fit()

resultados_HC0 = modelo_base.get_robustcov_results(cov_type='HC0')
resultados_HC1 = modelo_base.get_robustcov_results(cov_type='HC1')
resultados_HC2 = modelo_base.get_robustcov_results(cov_type='HC2')
resultados_HC3 = modelo_base.get_robustcov_results(cov_type='HC3')

def get_confint_df(result, var_names):
    ci = result.conf_int()
    if isinstance(ci, pd.DataFrame):
        ci = ci.loc[var_names]
    else:
        ci = pd.DataFrame(ci, columns=["lower", "upper"], index=var_names)
    return ci

variables = modelo_base.params.index

se_df = pd.DataFrame({
    "OLS": modelo_base.bse,
    "HC0": resultados_HC0.bse,
    "HC1": resultados_HC1.bse,
    "HC2": resultados_HC2.bse,
    "HC3": resultados_HC3.bse,
})
se_df.index.name = "Variable"

ic_ols = get_confint_df(modelo_base, variables)
ic_hc0 = get_confint_df(resultados_HC0, variables)
ic_hc1 = get_confint_df(resultados_HC1, variables)
ic_hc2 = get_confint_df(resultados_HC2, variables)
ic_hc3 = get_confint_df(resultados_HC3, variables)

ic_df = pd.DataFrame({
    "OLS_lower": ic_ols.iloc[:,0],
    "OLS_upper": ic_ols.iloc[:,1],
    "HC0_lower": ic_hc0.iloc[:,0],
    "HC0_upper": ic_hc0.iloc[:,1],
    "HC1_lower": ic_hc1.iloc[:,0],
    "HC1_upper": ic_hc1.iloc[:,1],
    "HC2_lower": ic_hc2.iloc[:,0],
    "HC2_upper": ic_hc2.iloc[:,1],
    "HC3_lower": ic_hc3.iloc[:,0],
    "HC3_upper": ic_hc3.iloc[:,1],
})
ic_df.index = variables
ic_df.index.name = "Variable"

display(se_df.style.format("{:.6f}"))
```

```{code-cell} ipython3
display(ic_df.style.format("{:.6f}"))
```

```{code-cell} ipython3
import statsmodels.api as sm
import pandas as pd

rlm_huber = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()       # Función de pérdida Huber
rlm_tukey = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight()).fit() # Función de pérdida Tukey Biweight

rlm_df = pd.DataFrame({
    'OLS': modelo_base.params,
    'RLM_Huber': rlm_huber.params,
    'RLM_Tukey': rlm_tukey.params
})

weights_df = pd.DataFrame({
    'Huber_weights': rlm_huber.weights,
    'Tukey_weights': rlm_tukey.weights
})

display(rlm_df)
```

```{code-cell} ipython3
display(weights_df)
```

```{code-cell} ipython3
from sklearn.utils import resample
import numpy as np
import pandas as pd

B = 1000
coef_boot = np.zeros((B, X.shape[1]))

for i in range(B):
    X_resample, y_resample = resample(X, y)
    model_bs = sm.OLS(y_resample, X_resample).fit()
    coef_boot[i, :] = model_bs.params

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

bootstrap_df.to_csv(BOOT_CSV_PATH, sep=",", index=False)

bootstrap_df
```

```{code-cell} ipython3
import pandas as pd

coef_ols = modelo_base.params         
se_ols = modelo_base.bse               

se_hc3 = resultados_HC3.bse            

coef_boot_mean = bootstrap_df['Coef_mean']  
se_boot = bootstrap_df['SE_bootstrap']  

ic_ols = modelo_base.conf_int().iloc[:,1] - modelo_base.conf_int().iloc[:,0]

ic_hc3 = resultados_HC3.conf_int()[:,1] - resultados_HC3.conf_int()[:,0]

ic_boot = bootstrap_df['IC_97.5%'] - bootstrap_df['IC_2.5%']

coef_df = comparative_df[['Coef_OLS', 'Coef_Bootstrap']]
se_df = comparative_df[['SE_OLS', 'SE_HC3', 'SE_Bootstrap']]
ic_df = comparative_df[['IC_width_OLS', 'IC_width_HC3', 'IC_width_Bootstrap']]
coef_df
```

```{code-cell} ipython3
se_df
```

```{code-cell} ipython3
ic_df
```

## Takeaways
- Los estimadores robustos reducen la sensibilidad a outliers e incumplimientos de supuestos.
- El bootstrap permite estimar la variabilidad sin suposiciones paramétricas fuertes.
- Comparar OLS, HC3 y Bootstrap prioriza inferencias estables cuando hay heterocedasticidad.