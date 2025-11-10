# Capítulo 1: Introduccióny pregunta de investigación 
---
> **Overview**:
Este capítulo presenta el dataset **Ames Housing**, su contexto y relevancia para modelado e inferencia.
Planteamos preguntas de investigación sobre la contribución de variables estructurales/vecinales al precio
y sobre la estabilidad de coeficientes bajo métodos robustos.
Definimos objetivos, alcance y criterios de éxito con énfasis en reproducibilidad y validez inferencial.
---
## Contexto del dataset y su relevancia estadística
Usaremos el **Ames Housing** (versión “AmesHousing.csv”), un registro inmobiliario del condado de Story, Iowa (EE. UU.).
Describe atributos estructurales, de sitio y de vecindario; la respuesta es SalePrice (USD).
`SalePrice` suele presentar asimetría positiva, por lo que `log(SalePrice)` mejora supuestos de MCO.
Existen faltantes altos en variables “estructurales” (p. ej., `Alley`, `Pool QC`), y correlaciones esperadas con `Overall Qual`, `Gr Liv Area`, `Garage Area/Cars`, `Total Bsmt SF`, `Year Built`.

## Preguntas de investigación
- ¿Qué variables estructurales y de vecindario explican en mayor medida `SalePrice`/`log(SalePrice)` al controlar por calidad y antigüedad?  
- ¿Qué tan estables son los coeficientes bajo métodos robustos (MCO vs. Huber/Tukey) frente a outliers y heterocedasticidad?  
- ¿Hay no linealidades relevantes o interacciones (`Overall Qual × Neighborhood`) que mejoren la explicación sin perder interpretabilidad?

## Objetivo general
Desarrollar un marco reproducible para explicar y modelar el precio de vivienda en Ames, comparando enfoques lineales (clásicos y robustos) y no lineales, priorizando validez inferencial y estabilidad.

## Objetivos específicos
1) Preparación: auditoría de faltantes, imputación, codificación y estandarización; VIF.  
2) Lineal: MCO sobre `log(SalePrice)` + diagnósticos (BP, QQ, Cook) y contraste con estimadores robustos.  
3) Alternativos (opcional): GAM/boosting para no linealidades; importancia de variables.  
4) Validación: k-fold; métricas RMSE/MAE/R² out-of-sample; sensibilidad; reporte reproducible.

## Criterios de éxito
**Reproducibilidad** (semillas/dependencias/CI), validez inferencial (heterocedasticidad controlada, VIF moderado, sin influencia extrema),
desempeño (p. ej., RMSE ≤ ~0,5·IQR, MAE ≤ ~0,35·IQR; R² ≥ 0,75) y estabilidad (signo consistente y magnitudes similares entre MCO y robusto).

## Alcances y supuestos
Respuesta primaria: `log(SalePrice)`; tratamiento de NA “estructurales” (categoría “None” o exclusión razonada); posibles splines/transformaciones en extensiones.

---

> **Key takeaways**
- El Ames Housing es ideal para contrastar enfoques explicativos vs. predictivos en un escenario realista.
- `log(SalePrice)` ayuda a mitigar asimetría y heterocedasticidad en modelos lineales.
- Calidad global y tamaño habitable concentran gran parte de la señal explicativa del precio.
- Los estimadores robustos permiten evaluar la estabilidad de conclusiones frente a outliers.
- Éxito = reproducibilidad, validez inferencial y buen rendimiento fuera de muestra.