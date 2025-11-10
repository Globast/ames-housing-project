---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# Capítulo 7 — Remedios y métodos robustos

## Overview

En este capítulo se ajusta un modelo lineal OLS para **Ames Housing** y se comparan errores estándar y
ancho de intervalos de confianza bajo OLS, correcciones robustas a la heterocedasticidad (HC0–HC3) y **bootstrap**.
También se estiman modelos **RLM** (Huber y Tukey) para mitigar la influencia de outliers.
Presentamos resultados en tablas y los discutimos, incluyendo recomendaciones prácticas.



## Configuración de rutas

```{code-cell} ipython3
from pathlib import Path

# Rutas deterministas: el capítulo se ejecuta desde la carpeta donde está este .md (p.ej. book/notebooks/)
DATA_PATH = Path("../data/AmesHousing_codificada.csv")  # relativo a book/notebooks/
BOOTSTRAP_OUT = Path("../data/bootstrap_df.csv")        # salida persistente en book/data/

assert DATA_PATH.is_file(), "No se encontró '../data/AmesHousing_codificada.csv'"
print("Usando CSV:", DATA_PATH.resolve())
print("Archivo bootstrap se guardará en:", BOOTSTRAP_OUT.resolve())
```

## 7.1 Correcciones para heterocedasticidad

De acuerdo con el supuesto de homocedasticidad ([Ecuación 6.2.1](#ecuacion-621-varianza-errores)), la presencia de heterocedasticidad puede provocar que los errores estándar de los coeficientes estén subestimados, afectando los valores $t$ y las decisiones de significancia. Para corregir este problema, se utilizan los estimadores de varianza robusta **HC** (Heteroscedasticity-Consistent), que ajustan los errores estándar sin cambiar los coeficientes estimados. 

- **HC0**: Estimador original de White, básico y consistente frente a heterocedasticidad.  
- **HC1**: Ajusta HC0 por los grados de libertad, corrigiendo ligeras subestimaciones.  
- **HC2**: Considera los *leverage points* de cada observación, dando más peso a observaciones influyentes.  
- **HC3**: Aproximación tipo *jackknife*, más conservadora y recomendada en muestras pequeñas por ofrecer errores estándar más cautelosos.

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

**Tabla 7.1.1.** Errores estándar OLS vs. HC0-HC3.

Se observa que los errores estándar aumentan ligeramente cuando se aplican las correcciones HC, especialmente para variables como `Overall Qual`, `Garage Cars` y `Full Bath`. Esto indica que los estimadores OLS originales podrían subestimar la variabilidad de los coeficientes si existe heterocedasticidad.

Por ejemplo, el coeficiente de `Overall Qual` tiene un error estándar de 0.00333 bajo OLS clásico, que se incrementa a 0.00447 bajo HC3, la corrección más conservadora. De manera similar, `Garage Cars` pasa de 0.00530 a 0.00687.  

Las variables con cambios mínimos en los errores estándar (como `Gr Liv Area` o `Lot Area`) sugieren que su variabilidad está poco afectada por heterocedasticidad.

```{code-cell} ipython3
display(ic_df.style.format("{:.6f}"))
```

**Tabla 7.1.2.** Intervalos de confianza OLS vs. HC0-HC3.

Se puede observar que los intervalos de confianza se ensanchan ligeramente cuando se aplican las correcciones HC, reflejando un aumento en la incertidumbre de los coeficientes. Por ejemplo, el coeficiente de `Overall Qual` tiene un intervalo de confianza de [0.095, 0.108] bajo OLS, que se amplía a [0.0929, 0.1104] con HC3. Lo mismo ocurre con `Garage Cars` y `Full Bath`, indicando que las inferencias de estos coeficientes son sensibles a la heterocedasticidad.

En contraste, intervalos de confianza prácticamente inalterados, como los de `Gr Liv Area` o `Lot Area`, sugieren que la heterocedasticidad tiene poco efecto sobre la precisión de estos coeficientes.  

En general, aplicar correcciones HC permite obtener intervalos de confianza más robustos, proporcionando inferencias más conservadoras y fiables cuando se sospecha heterocedasticidad.

## 7.2 Modelos robustos con funciones Huber/Tukey

Los **modelos de regresión robusta** son una extensión de la regresión lineal ordinaria diseñada para reducir la influencia de outliers o valores atípicos en la estimación de los coeficientes. Mientras que la regresión OLS pondera todos los residuales por igual, los modelos robustos asignan **menor peso a los residuales grandes**, permitiendo obtener estimaciones más confiables y estables.

Existen diversas funciones de pérdida que determinan cómo se penalizan los residuales, donde cada una equilibra de distinta manera la eficiencia para valores centrales y la robustez frente a outliers extremos:

- **Huber / TukeyHuberT()**:

```{math}
:label: eq:7.2.1-huber
\rho(r) = 
\begin{cases} 
\frac{1}{2} r^2 & \text{si } |r| \le \delta \\[2mm]
\delta \left(|r| - \frac{1}{2}\delta \right) & \text{si } |r| > \delta
\end{cases}
```
  
_Ecuación_ {eq}`eq:7.2.1-huber`. Función de pérdida Huber.

Protege contra outliers moderados manteniendo eficiencia para valores centrales.

- **Tukey Biweight**:

```{math}
:label: eq:7.2.2-tukey
\rho(r) = 
\begin{cases}
\frac{c^2}{6} \left[ 1 - \left(1 - \left(\frac{r}{c}\right)^2 \right)^3 \right] & \text{si } |r| \le c \\[1mm]
\frac{c^2}{6} & \text{si } |r| > c
\end{cases}
```
  
_Ecuación_ {eq}`eq:7.2.2-tukey`. Función de pérdida Tukey.

Limita el impacto de observaciones lejanas sin eliminarlas completamente.

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

**Tabla 7.2.1.** Coeficientes OLS vs. RLM. *Valores en escala logarítmica.*

Se observa que el intercepto (`const`) aumenta al usar modelos robustos, lo que refleja que los outliers tienden a sesgar hacia abajo la estimación en OLS. Por su parte, variables como `Overall Qual` y `Fireplaces` muestran coeficientes algo menores en modelos robustos, indicando que su efecto estaba ligeramente sobreestimado por la presencia de outliers. 

Para la mayoría de las demás variables (`Gr Liv Area`, `Garage Cars`, `Total Bsmt SF`, `Year Built`), las diferencias son pequeñas, lo que sugiere que los outliers no tienen un impacto fuerte en estas estimaciones.

```{code-cell} ipython3
display(weights_df)
```

**Tabla 7.2.2.** Pesos outliers Huber vs. Tukey.

La mayoría de los pesos son cercanos a 1 en ambos modelos robustos, lo que indica que la mayoría de las observaciones se ajusta bien al modelo y tiene plena influencia en la estimación de los coeficientes. 

Sin embargo, algunas observaciones presentan pesos menores, como la última fila, con Huber = 0.69 y Tukey = 0.68, lo que refleja que su residuo es relativamente grande y su efecto en el ajuste se atenúa.

Además, es evidente que Tukey aplica un castigo más fuerte a residuales extremos.

## 7.3 Bootstrap

El **bootstrap** es un método de remuestreo que permite estimar la variabilidad de los coeficientes de un modelo sin asumir una distribución específica de los errores. Consiste en generar múltiples muestras con reemplazo a partir de los datos originales y recalcular los estimadores para cada réplica, obteniendo así una **distribución empírica** de los coeficientes, a partir de la cual se calculan el error estándar y los intervalos de confianza.

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

bootstrap_df.to_csv('bootstrap_df.csv', sep=",", index=False)

bootstrap_df
```


**Tabla 7.3.1 — Resumen de bootstrap de coeficientes.** Esta tabla resume los resultados producidos por el código anterior.
Discusión: compare magnitudes relativas. En presencia de heterocedasticidad,
los SE HC3 suelen ser mayores que OLS. Si los IC se ensanchan (ver Tabla 7.2), las
conclusiones sobre significancia pueden cambiar. Con bootstrap, valide la estabilidad
de los coeficientes frente a remuestreo. (Ver «Takeaways» al final.)

Se observa que variables como `Overall Qual`, `Gr Liv Area`, `Garage Cars` y `Year Built` tienen coeficientes claramente distintos de cero, con intervalos de confianza estrechos y consistentes, lo que sugiere estimaciones robustas y estables. Por el contrario, `1st Flr SF` y `Full Bath` presentan intervalos que incluyen el cero, indicando que su efecto sobre la variable respuesta podría no ser significativo.

## 7.4 OLS vs. HC3 vs. Bootstrap

```{code-cell} ipython3

import pandas as pd
# 1Extraer y alinear objetos base
coef_ols = modelo_base.params
se_ols = modelo_base.bse
se_hc3 = resultados_HC3.bse

# Asegurar que bootstrap_df esté indexado por el nombre del coeficiente (si aplica)
if isinstance(bootstrap_df, pd.DataFrame) and 'variable' in bootstrap_df.columns and bootstrap_df.index.name != 'variable':
    bootstrap_df = bootstrap_df.set_index('variable')

coef_boot_mean = bootstrap_df['Coef_mean'].reindex(coef_ols.index)
se_boot = bootstrap_df['SE_bootstrap'].reindex(coef_ols.index)

# 2 Anchuras de IC (97.5 - 2.5) para cada método
ic_ols_width = (modelo_base.conf_int().iloc[:, 1] - modelo_base.conf_int().iloc[:, 0]).reindex(coef_ols.index)

hc3_ci = resultados_HC3.conf_int()
# Normalizar a DataFrame con el mismo índice
if isinstance(hc3_ci, pd.DataFrame):
    hc3_ci_df = hc3_ci
else:
    hc3_ci_df = pd.DataFrame(hc3_ci, index=coef_ols.index, columns=[0, 1])

ic_hc3_width = (hc3_ci_df.iloc[:, 1] - hc3_ci_df.iloc[:, 0]).reindex(coef_ols.index)
ic_boot_width = (bootstrap_df['IC_97.5%'] - bootstrap_df['IC_2.5%']).reindex(coef_ols.index)

# 3 Tabla maestra comparativa
comparative_df = pd.DataFrame({
    'Coef_OLS': coef_ols,
    'Coef_Bootstrap': coef_boot_mean,
    'SE_OLS': se_ols,
    'SE_HC3': se_hc3,
    'SE_Bootstrap': se_boot,
    'IC_width_OLS': ic_ols_width,
    'IC_width_HC3': ic_hc3_width,
    'IC_width_Bootstrap': ic_boot_width
})

# 4 Subtablas
coef_df = comparative_df[['Coef_OLS', 'Coef_Bootstrap']]
se_df = comparative_df[['SE_OLS', 'SE_HC3', 'SE_Bootstrap']]
ic_df = comparative_df[['IC_width_OLS', 'IC_width_HC3', 'IC_width_Bootstrap']]

# Mostrar por conveniencia
coef_df
```


**Tabla 7.4.1— Coeficientes OLS vs. Bootstrap.** *Valores en escala logarítmica.* Esta tabla resume los resultados producidos por el código anterior.
Discusión: compare magnitudes relativas. En presencia de heterocedasticidad,
los SE HC3 suelen ser mayores que OLS. Si los IC se ensanchan (ver Tabla 7.2), las
conclusiones sobre significancia pueden cambiar. Con bootstrap, valide la estabilidad
de los coeficientes frente a remuestreo. (Ver «Takeaways» al final.)

Se observa que la estimación de los parámetros es muy estable frente al remuestreo, lo que sugiere que la muestra utilizada es suficientemente representativa y que los coeficientes no dependen excesivamente de observaciones individuales.

En particular, las variables como `Overall Qual`, `Gr Liv Area` y `Fireplaces` muestran coeficientes positivos consistentes en ambos métodos, confirmando su relación directa con el precio de la vivienda. Por su parte, `Full Bath` mantiene un coeficiente ligeramente negativo, indicando que, controlando por las demás variables, su efecto es mínimo.

```{code-cell} ipython3
se_df
```


**Tabla 7.4.2 — Errores estándar comparados (OLS vs. HC0–HC3 vs. Bootstrap).** Esta tabla resume los resultados producidos por el código anterior.
Discusión: compare magnitudes relativas. En presencia de heterocedasticidad,
los SE HC3 suelen ser mayores que OLS. Si los IC se ensanchan (ver Tabla 7.2), las
conclusiones sobre significancia pueden cambiar. Con bootstrap, valide la estabilidad
de los coeficientes frente a remuestreo. (Ver «Takeaways» al final.)


Los errores estándar de OLS son generalmente menores que los obtenidos mediante HC3 o bootstrap, lo que sugiere que este modelo inicial podría subestimar la incertidumbre cuando hay heterocedasticidad presente o dependiendo de la muestra seleccionada.  

El método HC3, diseñado para ser más conservador frente a heterocedasticidad y leverage points, produce errores estándar ligeramente mayores que OLS, especialmente para variables como `Overall Qual`, `Garage Cars` y `Full Bath`. El bootstrap refleja un patrón muy similar al de HC3, confirmando la estabilidad de las estimaciones.

Es notable que las variables con errores estándar relativamente bajos (`Gr Liv Area`, `Lot Area`) están estimadas con gran precisión, mientras que aquellas con errores más altos (`Full Bath`, `Garage Cars`) presentan mayor incertidumbre en la estimación de su efecto sobre el precio de la vivienda.

```{code-cell} ipython3
ic_df
```


**Tabla 7.4.3 — Ancho de intervalos de confianza por método.** Esta tabla resume los resultados producidos por el código anterior.
Discusión: compare magnitudes relativas. En presencia de heterocedasticidad,
los SE HC3 suelen ser mayores que OLS. Si los IC se ensanchan (ver Tabla 7.2), las
conclusiones sobre significancia pueden cambiar. Con bootstrap, valide la estabilidad
de los coeficientes frente a remuestreo. (Ver «Takeaways» al final.)

Alineados con los errores estándar ([Tabla 7.4.2](#tabla-742-se-ols-hc3-boot)), los intervalos calculados con OLS son consistentemente más estrechos que los obtenidos con HC3 o bootstrap, llegando a la misma conclusión de que el OLS inicial es menos robusto.

## Discusión y análisis

**Sobre heterocedasticidad.** Si los errores estándar HC3 (o HC2) superan de manera consistente a los de OLS,
esto sugiere varianza no constante. En tal caso, la inferencia debe basarse en versiones robustas (HC3 recomendado).
Además, el **ancho de los IC** (Tabla 7.2) es un buen indicador del impacto en la precisión de la estimación.

**Sobre RLM (Huber/Tukey).** Cuando existen outliers influyentes, RLM puede reducir su peso (ver Tabla 7.3 y, si corresponde,
los pesos por observación). Cambios sustanciales en coeficientes o en su significancia, frente a OLS, ameritan diagnosticar
casos influyentes y revisar la especificación.

**Sobre bootstrap.** El bootstrap (Tabla 7.4) brinda una validación empírica de la variabilidad de los parámetros.
Considere comparar los percentiles 2.5%/97.5% de bootstrap con los IC teóricos; discrepancias marcadas sugieren
sensibilidad a supuestos o a la muestra.

## Takeaways

1. **Inferencia robusta:** En presencia de heterocedasticidad, utilice **HC3** para SE e IC; valide conclusiones frente a OLS.
2. **Diagnóstico de outliers:** Si RLM re-pondera fuertemente algunos casos, investigue esas observaciones (posibles errores o segmentos distintos).
3. **Validación por remuestreo:** Use **bootstrap** para verificar estabilidad de coeficientes y anchos de IC.
4. **Rutas deterministas:** Los datos se leen desde `DATA_PATH` y los resultados de bootstrap se guardan en `BOOTSTRAP_OUT` (book/data/).