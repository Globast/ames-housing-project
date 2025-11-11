# Capítulo 10: Conclusiones y trabajo futuro

## Respuesta a preguntas de investigación

1. **¿Qué implicaciones prácticas pueden derivarse de los resultados para la valoración inmobiliaria en Ames, Iowa?**

El modelo indica que el precio de las viviendas depende principalmente de la calidad general, el espacio habitable, el área del garaje y la antigüedad. Una mejora de un punto en la calidad general incrementa el valor en aproximadamente 9%–10%, mientras que cada 100 ft² adicionales en superficie habitable aumentan el precio entre $7500 y $8000. Cada espacio adicional en el garaje aporta cerca de $12000–$14000, y una diferencia de diez años en antigüedad se asocia con una reducción del 3%–5% en el valor. Estos resultados pueden guiar decisiones de tasación, remodelación y fijación de precios en el mercado inmobiliario de Ames.

2. **¿Hasta qué punto los métodos de regularización (Ridge y Lasso) mejoran la capacidad predictiva frente al modelo OLS?**

El modelo OLS presentó un ajuste con R² = 0.84 y evidencia de heterocedasticidad. Los métodos Ridge y Lasso alcanzaron un R² cercano a 0.85, reduciendo el error de predicción y la sensibilidad ante variables correlacionadas. Lasso, además, eliminó predictores de bajo aporte sin pérdida de ajuste, mejorando la estabilidad y la interpretabilidad del modelo frente al OLS.

3. **¿Cuál es el precio promedio esperado de una vivienda cuando se mantienen constantes las demás variables relevantes?**

El intercepto del modelo Lasso corresponde a un precio promedio estimado cercano a $180000, representando una vivienda típica en Ames con valores medios en sus características. A partir de esa base, la calidad, el tamaño habitable y el garaje son los principales factores que aumentan el valor, mientras que la antigüedad lo reduce de forma moderada.

4. **¿Qué tan bien predice el modelo los precios?**

El modelo Lasso explica aproximadamente el 85% de la variación en los precios (R² = 0.85) y presenta un error absoluto medio entre $17000 y $19000. Los residuos no muestran patrones sistemáticos, lo que sugiere un buen ajuste general y ausencia de sobreajuste.

## Conclusiones

El análisis evidenció que los **Modelos Lineales Clásicos (OLS)** son una herramienta sólida para comprender los determinantes del precio de la vivienda, pero su desempeño depende del cumplimiento de los supuestos estadísticos. Se detectaron **outliers** y **heterocedasticidad**, lo que afectó la validez de los errores estándar. Al aplicar **métodos robustos** (Huber y Tukey) y de **regularización** (Ridge y Lasso), se obtuvo un modelo más **estable, interpretativo y predictivo**.

En términos prácticos, la **calidad general** y la **superficie habitable** son los factores con mayor influencia positiva en el valor, seguidos por el **garaje** y el **sótano**. La **antigüedad** ejerce un efecto negativo moderado. El **tamaño del terreno** mostró un efecto marginal, lo que confirma que los compradores priorizan el **espacio interior** y la **calidad constructiva** sobre la extensión del lote.

Los resultados empíricos validan la **teoría económica inmobiliaria** y demuestran que **combinar el enfoque clásico con técnicas modernas de regularización** permite desarrollar modelos más **precisos, robustos y útiles** para la toma de decisiones en **tasación y predicción de precios**.

## Trabajo futuro

**Líneas futuras de investigación**

Para ampliar el alcance del estudio, se propone integrar **modelos de Gradient Boosting**, en particular **XGBoost**, como alternativa no lineal y de mayor potencia predictiva (véase: *An Optimal House Price Prediction Algorithm: XGBoost*, Sharma et al., 2024. https://www.mdpi.com/2813-2203/3/1/3).

**Ventajas principales**

- No requiere los supuestos de **linealidad** u **homocedasticidad**.  
- Captura **interacciones complejas** entre variables (por ejemplo, *calidad × tamaño*).  
- Es **robusto frente a outliers** y escalas distintas de variables.  
- Mejora la **capacidad predictiva**: estudios muestran reducciones del **20%–40% en RMSE** frente a OLS.  
- Permite interpretar la **importancia de las variables** mediante **SHAP values**, que cuantifican la contribución individual de cada predictor.

**Resultados exploratorios**

**XGBoost** alcanzó un **R² ≈ 0.91**, superando a **OLS** y **RLM**, y ajustándose mejor a los extremos del rango de precios.  
Su capacidad para manejar **no linealidades** y **efectos combinados** lo convierte en una herramienta prometedora para **sistemas de valoración automatizada**.

**Líneas futuras de trabajo**

- Implementar **validación cruzada k-fold** para estimar el error de generalización.  
- Comparar el rendimiento de **XGBoost**, **Random Forests** y **Redes Neuronales**.  
- Desarrollar un **dashboard interactivo** (por ejemplo, en *Streamlit* o *Power BI*) para simular precios según atributos.  
- Explorar **enfoques Bayesianos y causales**, que aporten interpretabilidad y estimación de incertidumbre.

En conjunto, la integración de **métodos clásicos y modernos** fortalece tanto la **interpretabilidad** como la **efectividad empírica** del análisis de precios de vivienda en Ames, estableciendo una base sólida para **futuras aplicaciones en predicción y análisis inmobiliario**.



