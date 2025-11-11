# Capítulo 1: Introduccióny pregunta de investigación 
---
## Contexto del estudio

El conjunto de datos **Ames Housing Dataset** es una base de datos ampliamente utilizada en análisis estadístico y ciencia de datos, que recopila información detallada sobre más de 2900 viviendas ubicadas en la ciudad de Ames, Iowa (EE. UU.). Fue desarrollado por Dean De Cock como una alternativa moderna y más completa al clásico *Boston Housing Dataset*, e incluye **82 variables** que describen características estructurales, de ubicación y de calidad de las propiedades, junto con su **precio de venta**.  

## Preguntas de investigación

1. ¿Qué implicaciones prácticas pueden derivarse de los resultados para la valoración inmobiliaria en Ames, Iowa?
2. ¿Hasta qué punto los métodos de regularización (Ridge y Lasso) mejoran la capacidad predictiva frente al modelo OLS?
3. ¿Cuál es el precio promedio esperado de una vivienda cuando se mantienen constantes las demás variables relevantes?
4. ¿Qué tan bien predice el modelo los precios?

## Objetivos del estudio

**Objetivo general**

Aplicar de manera integrada los conceptos de regresión lineal, inferencia estadística, diagnóstico de supuestos y métodos robustos sobre el conjunto de datos *Ames Housing Dataset*, con el fin de identificar los factores que más influyen en el precio de venta de las viviendas y evaluar la estabilidad y precisión de los modelos obtenidos.

**Objetivos específicos**

1. **Describir y preparar el conjunto de datos Ames Housing**, documentando su estructura, tipos de variables y el tratamiento de valores faltantes, outliers, transformaciones y codificaciones aplicadas.  

2. **Explorar las relaciones entre variables** mediante análisis descriptivo y visual (histogramas, boxplots, correlaciones y mapas de calor), identificando los predictores más relevantes para el precio de venta.  

3. **Formular y estimar el modelo base de regresión lineal múltiple (OLS)**, expresándolo en su forma matricial, verificando la invertibilidad de matrices y comparando los resultados con la implementación en `statsmodels`.  

4. **Realizar inferencia estadística sobre los coeficientes del modelo**, interpretando errores estándar, valores *t*, *p*-values e intervalos de confianza para determinar la significancia y magnitud de los efectos.  

5. **Evaluar los supuestos clásicos del modelo lineal** (linealidad, homocedasticidad, normalidad, independencia y multicolinealidad) mediante pruebas y gráficos diagnósticos, documentando los resultados en una tabla resumen.  

6. **Aplicar métodos robustos frente a violaciones de supuestos**, incluyendo:
   - Correcciones HC0–HC3 para heterocedasticidad.  
   - Modelos RLM con funciones Huber y Tukey.  
   - Estimación de errores e intervalos por *bootstrap*.  

7. **Comparar los resultados de OLS y métodos robustos**, analizando diferencias en coeficientes, errores estándar e intervalos de confianza.  

8. **Implementar y validar modelos de regularización Ridge y Lasso**, ajustando hiperparámetros por validación cruzada y comparando su desempeño con OLS mediante métricas (R², RMSE, MAE).  

9. **Evaluar la estabilidad y capacidad predictiva de los modelos** en conjuntos de entrenamiento, validación y prueba, identificando posibles casos de sobreajuste.  

10. **Interpretar los hallazgos en términos prácticos y aplicados**, traduciendo los resultados estadísticos a conclusiones útiles para la valoración inmobiliaria y proponiendo líneas de investigación futura.

> **Key takeaways**
>- Las preguntas de investigación delimitan qué medir, cómo evaluarlo y con qué métricas.
>- La elección de la transformación de la respuesta condiciona la interpretación de los coeficientes.

