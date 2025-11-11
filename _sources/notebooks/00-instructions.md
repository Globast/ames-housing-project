---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---

# Capítulo 0 · Instrucciones de reproducción
## Instrucciones de reproducción y Demostraciones
> **Overview**: Este capitulo presenta instrucciones de reproducción. Se muestran las demostraciones solicitadas.


## Cómo compilar el libro:

```bash
jupyter-book build .
```

**Dependencias**: ver `requirements.txt` en la raíz de `book/`.

**Dataset**: se espera en `data/ames_housing.csv`.
- Descarga manual desde Kaggle o usa la API:
  ```bash
  kaggle datasets download -d prevek18/ames-housing-dataset -p data/ --unzip
  mv data/AmesHousing.csv data/ames_housing.csv
  ```

**Control de versiones** 
```
book/
├── data/
├── notebooks/
├── _build/
└── _config.yml
```

**Semillas reproducibles**: todos los experimentos fijan `random_state`.
**Mapa del libro**: ver la barra lateral y `_toc.yml`.



## Demostraciones solicitadas

## Modelo y notación

El modelo lineal es:

y = Xβ + ε,  ε ~ N(0, σ²Iₙ)

donde:

- X: matriz de diseño n × k  
- β: vector de parámetros  
- ε: vector de errores  


## Estimador OLS y matriz hat

El estimador de mínimos cuadrados es:

β̂ = (X'X)⁻¹ X'y

Las predicciones:

ŷ = Xβ̂ = H y,  H = X(X'X)⁻¹X'

Propiedades de H:
- Simétrica: H' = H  
- Idempotente: H² = H  
- Rango: rank(H) = k  


## Residuos y operador (I - H)

Los residuos:

e = y - ŷ = (I - H)y

Sustituyendo y = Xβ + ε y usando (I - H)X = 0:

e = (I - H)ε

Cada residuo es una combinación lineal de los errores originales.


## Suma de cuadrados residual

SSR = e'e = [(I-H)ε]'[(I-H)ε] = ε'(I-H)'(I-H)ε

Como (I - H) es simétrica e idempotente:

SSR = ε'(I - H)ε

El SSR es una forma cuadrática en los errores.


## Rango y grados de libertad

rank(I - H) = n - rank(H) = n - k

→ El espacio de los residuos tiene dimensión n - k.  
→ Solo n - k residuos son independientes (pues X'e = 0).


## Distribución Chi-cuadrado

Si ε ~ N(0, σ²I) y A es simétrica e idempotente de rango r:

(1/σ²) ε'Aε ~ χ²ᵣ

Aplicando A = I - H:

(1/σ²) ε'(I - H)ε ~ χ²ₙ₋ₖ

Por tanto:

SSR / σ² ~ χ²ₙ₋ₖ


## Por qué no basta con elevar los residuos al cuadrado

- Los errores εᵢ son normales independientes, por lo que (εᵢ / σ)² ~ χ²₁  
- Pero los residuos eᵢ son combinaciones lineales → no independientes  
- Además, Var(eᵢ) = σ²(1 - hᵢᵢ)  
- Por eso, solo la suma total e'e sigue χ², no cada residuo por separado.


## Intuición geométrica

- ε vive en un espacio de dimensión n  
- H proyecta sobre el subespacio de predicciones (dimensión k)  
- (I - H) proyecta sobre el espacio ortogonal de residuos (dimensión n - k)  
- La longitud al cuadrado de esa proyección, dividida por σ², sigue χ²ₙ₋ₖ


## Resultado final

SSR = ε'(I - H)ε  
SSR / σ² ~ χ²ₙ₋ₖ  

En regresión lineal simple (k = 2): n - 2 grados de libertad.
