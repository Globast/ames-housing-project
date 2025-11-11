# Capítulo 10: Conclusiones y trabajo futuro
Este capítulo cierra el recorrido conectando lo que preguntamos al inicio con lo que efectivamente aprendimos. El objetivo es entregar un mapa claro: qué respuestas respaldan los datos, con qué grado de confianza, qué no sabemos aún y cómo avanzar. Mantendremos el tono didáctico, privilegiando ejemplos, tablas-resumen y “reglas de bolsillo”.

## 10.1. ¿Qué preguntamos y qué respondimos? 

A continuación, una síntesis que liga cada pregunta inicial con la respuesta operativa y la evidencia que la sustenta en el proyecto Ames Housing.

| Pregunta inicial | Respuesta breve (operativa) | Evidencia empírica | Alcance e interpretación |
|---|---|---|---|
| ¿Cuánto “pesa” el área habitable en el precio? | Regla de bolsillo: **+100 ft² ⇒ ≈ +\$7.5k** en el precio promedio. | Coeficiente OLS positivo y estable para `GrLivArea`; consistencia bajo especificaciones con controles y errores estándar robustos. | Asociación condicionada; el retorno marginal puede atenuarse en tamaños extremos y variar por calidad/vecindario. |
| ¿La calidad general es tan influyente como parece? | Sí, cambios de 1 punto en `OverallQual` muestran incrementos notables en precio. | Coeficientes grandes y estadísticamente significativos; efectos estables con controles. | Señal fuerte, pero puede capturar componentes de acabados y estado no totalmente observados. |
| ¿Garaje y sótano añaden valor de forma sistemática? | En promedio, sí; su efecto es positivo, aunque menor que área/calidad. | Coeficientes positivos en OLS; robustos a HC3; sensibles a la distribución. | Efecto depende del segmento; utilidad marginal decreciente plausible. |
| ¿El modelo predice razonablemente bien? | Sí, con buen ajuste en rango medio; peor en colas. | Métricas fuera de muestra aceptables; diagnósticos muestran heterocedasticidad manejable. | No extrapolar a propiedades extremadamente atípicas; evaluar intervalos de predicción. |
.

## Integración teórica + empírica 

La teoría sugiere que el precio capitaliza atributos estructurales (tamaño, calidad) y contextuales (ubicación). Los datos del proyecto confirman el papel central de área y calidad, y además señalan variaciones por entorno y por no linealidad. La tabla siguiente resume el “puente” entre conceptos y resultados.

| Concepto teórico | Evidencia empírica observada | Implicación práctica |
|---|---|---|
| Productividad del espacio (más metros, más utilidad) | `GrLivArea` con coeficiente positivo y estable; mejores métricas al permitir curvaturas en tamaños altos. | El metro adicional “rinde”, pero no siempre al mismo ritmo; considerar tramos. |
| Calidad como multiplicador del valor | `OverallQual` con efecto grande y robusto; interacciones con área mejoran ajuste. | Un pie cuadrado en alta calidad vale más que en baja calidad. |
| Ubicación como capitalización de amenidades | Inclusión de dummies/controles de vecindario cambia magnitudes de coeficientes. | Parte del “efecto tamaño” era vecindario; imprescindible controlar entorno. |
| Imperfecciones de mercado y heterocedasticidad | Residuos con patrón; HC3 y bootstrap estabilizan inferencia. | Reportar siempre bandas de incertidumbre robustas. |

## Calidad de la evidencia 

La fuerza de los hallazgos depende de qué tan bien se cumplan los supuestos. Esta lectura rápida ayuda a comunicar honestamente el grado de certeza.

| Aspecto | Estado en el proyecto | Riesgo si se viola | Qué hicimos / Qué falta |
|---|---|---|---|
| Linealidad global | Aceptable en rango medio; señales de curvatura en colas | Sesgo en coeficientes marginales | Añadimos términos/curvaturas; sugerimos GAM/ splines (§10.6) |
| Exogeneidad (sin omitidas relevantes) | Parcial, mejora con controles de vecindario | Sesgo por variables omitidas | Más controles, interacciones; avanzar a diseños causales (§10.6) |
| Homocedasticidad | No se cumple estrictamente | Inferencia optimista | Usamos HC3/ bootstrap; proponer modelos con varianza flexible |
| Influencia de outliers | Presente en segmentos extremos | Estimaciones inestables | Diagnósticos y robustez; revisar estimadores resistentes |

##  “Reglas de bolsillo” con incertidumbre (para decidir mejor)

Las reglas ayudan a negociar, pero deben ir con su banda de confianza. La siguiente tabla ilustra el formato de reporte sugerido (valores de ejemplo para ilustrar la forma).

| Regla | Estimado central | IC 95% (método robusto) | Comentario pedagógico |
|---|---|---|---|
| +100 ft² en área habitable | +\$7,500 | [+\$3,600, +\$11,400] | Es una **asociación condicional**; puede variar por calidad/vecindario. |
| +1 punto en calidad general | +\$x | [+\$x₁, +\$x₂] | A veces captura mejoras de acabados; prudencia al inferir causalidad. |
| +1 plaza de garaje | +\$y | [+\$y₁, +\$y₂] | Efecto medio menor que área/calidad; depende del segmento. |

> Sugerencia de formato: si el modelo está en log-precio, convierta a **porcentajes** para una comunicación más natural entre zonas con distintos niveles de precios.

## Lo que **no** podemos concluir (todavía)

No podemos afirmar que aumentar el área **causa** el aumento exacto reportado en todos los contextos. Parte del efecto puede provenir de localización, estado interno o preferencias del comprador no observadas. Tampoco aseguramos que el retorno de 100 ft² sea constante para viviendas extremadamente pequeñas o de lujo; en esos bordes, la pendiente cambia y conviene modelar tramos o usar enfoques no lineales.

## Trabajo futuro: cómo dar el siguiente salto

Para pasar de buenas asociaciones a conclusiones más sólidas y decisiones más finas, proponemos un plan en cuatro líneas complementarias.

### Inferencia Bayesiana 

La perspectiva bayesiana permite integrar información previa (por ejemplo, estimaciones históricas del mercado) y producir intervalos creíbles directamente sobre cantidades de interés, como el efecto de +100 ft² en distintos vecindarios. Además, los modelos **jerárquicos** bayesianos permiten “compartir fuerza” entre barrios: vecindarios con pocos datos se benefician de la información de vecinos similares (shrinkage), estabilizando estimaciones.

| Pieza | ¿Qué aporta? | Resultado esperado |
|---|---|---|
| Priors informativos/weakly-informative | Evitan sobreajustes, estabilizan coeficientes | Bandas más realistas en muestras pequeñas |
| Modelos jerárquicos por vecindario | Efectos específicos con pooling parcial | Mapas de retornos por zona con menor varianza |
| Predicción posterior (PPC) | Diagnósticos en el espacio de los datos | Validaciones más intuitivas para usuarios |

### Causalidad aplicada

Para evaluar efectos causales se requieren supuestos y diseños explícitos. Tres rutas prácticas:

1. **DAGs + control dirigido**: explicitar el grafo causal, identificar conjuntos de ajuste mínimos y evitar sobrecontrol.  
2. **Diseños cuasi-experimentales**:  
   - **RDD** cuando hay umbrales de política o clasificación (por ejemplo, rangos impositivos).  
   - **DiD** si existen shocks locales/temporales observables.  
   - **Variables instrumentales** cuando hay una fuente válida de variación exógena en área o calidad.  
3. **Emparejamiento / peso por propensión**: construir grupos comparables para evaluar diferencias promedio con menor sesgo.

Cada enfoque exige **pruebas de validez** (balance, placebo, sensibilidad). La decisión no es “mecánica”: se elige en función de la pregunta y de la estructura del mercado.

### No linealidad e interpretabilidad (GAM/ML con lupa)

La evidencia sugiere curvaturas en colas y posibles interacciones. Proponemos:

- **GAM** (Modelos Aditivos Generalizados) para capturar formas suaves por variable con interpretabilidad.  
- **Bosques/Boosting** para mejorar predicción en colas, siempre acompañados de herramientas explicativas como **efectos acumulados** (PDP), **ICE** y **SHAP** con verificación de estabilidad.  
- **Intervalos conformales** para predicciones con cobertura controlada, útiles en tasación operativa.

###  Efectos espaciales y temporalidad

Incorporar **dependencia espacial** (SAR/SEM) y medidas de accesibilidad real (tiempos a amenidades) puede redefinir el peso del vecindario. Si hay series en el tiempo, usar ventanas móviles o modelos con efectos fijos temporales ayuda a separar tendencias del “ruido” coyuntural.

## Checklist de replicación y transferencia

Para que un tercero replique y utilice el trabajo sin fricciones, recomendamos este flujo operativo:

1. Documentar rutas de datos limpias y versiones de librerías empleadas.  
2. Proveer un cuaderno end-to-end con: carga de datos, especificaciones comparadas, diagnósticos, y generación de tablas y figuras.  
3. Incluir un anexo de **sensibilidad a omitidas** y otro de **reglas de bolsillo** por tramos de tamaño y por vecindario.  
4. Exportar artefactos de servicio: diccionario de variables, plantillas de reporte e intervalos de predicción listos para consumo.

## Resumen ejecutivo

El proyecto confirma, con consistencia, que área habitable y calidad general son pilares del precio, y que su efecto se matiza por vecindario y por no linealidad en los extremos. Las reglas de bolsillo —como “+100 ft² ≈ +\$7.5k”— son útiles para decidir, siempre acompañadas por bandas de incertidumbre y la advertencia de que describen asociaciones, no causalidad asegurada. El camino natural para fortalecer conclusiones pasa por adoptar inferencia bayesiana para una incertidumbre mejor calibrada, por incorporar diseños causales cuando la pregunta lo requiera y por modelar curvaturas e interacciones con herramientas interpretables. Así, pasamos de una buena brújula a un mapa más preciso para valoración, negociación y política pública.

