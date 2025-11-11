# Capítulo 9: Interpretación aplicada

El modelo Lasso obtuvo un **R² ≈ 0.85**, lo que indica que explica cerca del **85% de la variación en el precio de venta** de las viviendas.

| Variable | Interpretación práctica |
|-----------|-------------------------|
| **Calidad general** | Presenta el coeficiente más alto en el modelo, lo que confirma que es el principal determinante del valor de la vivienda. Un incremento de un punto en la calificación de calidad se asocia con un aumento promedio del **9–10% en el precio**. El intervalo de confianza muestra una relación consistentemente positiva, lo que refuerza su peso estructural en la explicación del valor. |
| **Espacio habitable** | El coeficiente positivo y estadísticamente sólido indica que el tamaño interior tiene un efecto directo y significativo sobre el precio. Un incremento de **100 ft²** en el área habitable promedio eleva el valor estimado entre **USD7,500 y USD8,000**. Los intervalos estrechos sugieren alta precisión en la estimación. |
| **Área del garaje (capacidad del garaje)** | El efecto estimado muestra que cada espacio adicional en el garaje se asocia con un incremento de **USD12,000–USD14,000** en el precio. La magnitud del coeficiente refleja su relevancia práctica y coincide con el comportamiento observado en los datos, donde las viviendas con mayor capacidad de estacionamiento alcanzan precios más altos. |
| **Tamaño del sótano** | La relación estimada es positiva y de menor magnitud que la del área habitable. Por cada **100 ft² adicionales**, el precio aumenta entre **USD2,000 y USD2,500**, lo que indica que el espacio subterráneo agrega valor, aunque en menor proporción que las áreas principales. |
| **Área del primer piso** | Esta variable fue eliminada por el modelo Lasso, lo que indica que su efecto está capturado por el espacio habitable total. En términos prácticos, el impacto del primer piso no es independiente, sino que se integra en la medición global del tamaño de la vivienda. |
| **Número de baños completos** | También fue excluida por el modelo. Esto sugiere que, una vez considerados la calidad y el tamaño de la vivienda, la cantidad de baños no aporta información adicional relevante para explicar el precio. En otras palabras, su influencia está mediada por otros atributos estructurales. |
| **Año de la construcción** | Muestra un coeficiente positivo, lo que indica que las viviendas más recientes tienden a tener precios superiores. Una diferencia de diez años en la fecha de construcción se traduce en un aumento estimado del **3–5% en el valor**. El intervalo de confianza respalda una relación positiva consistente. |
| **Chimeneas** | El coeficiente es positivo pero de pequeña magnitud: una chimenea adicional se asocia con un incremento aproximado del **2–3% en el precio**. El intervalo indica una variabilidad moderada, pero la dirección del efecto se mantiene estable. |
| **Tamaño del terreno** | Muestra un efecto positivo marginal y con un intervalo más amplio, lo que sugiere una relación débil con el precio una vez controlados la calidad y el tamaño habitable. El valor del lote importa menos que el espacio interior en la percepción de valor de las viviendas. |

**Tabla 9.1.1.** Interpretación práctica de variables.

En conjunto, el modelo identifica a la **calidad general**, el **espacio habitable**, la **capacidad del garaje** y el **año de la construcción** como los factores más determinantes del precio. Por su parte, las variables excluidas (área del primer piso y número de baños) muestran que su aporte al modelo es redundante frente a las características más relevantes.

Ahora bien, los resultados obtenidos **describen asociaciones, no relaciones de causa y efecto**. Por ejemplo, que una vivienda más grande tenga un precio mayor no significa que aumentar el tamaño garantice automáticamente ese incremento: puede deberse a que las viviendas más amplias suelen ubicarse en zonas con mejores servicios o calidades constructivas superiores.

Además, existen **factores omitidos** que pueden influir en el precio y que el modelo no incorpora directamente, entre ellos los **factores de mercado:** año de venta, demanda local o condiciones económicas.

## Takeaways
- Se documentan resultados adicionales con el mismo estándar reproducible.
- Las conclusiones complementan la interpretación de capítulos anteriores.

---
 
