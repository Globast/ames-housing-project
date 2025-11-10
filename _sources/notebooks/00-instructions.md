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
### Definición del modelo

En regresión lineal, el modelo puede expresarse de forma matricial como:

$$
Y = X\beta + \varepsilon
$$

donde

$$
Y =
\begin{bmatrix}
y_1 \\ y_2 \\ \vdots \\ y_n
\end{bmatrix},\quad
X =
\begin{bmatrix}
1 & x_1 \\
1 & x_2 \\
\vdots & \vdots \\
1 & x_n
\end{bmatrix},\quad
\beta =
\begin{bmatrix}
\beta_0 \\ \beta_1
\end{bmatrix},\quad
\varepsilon =
\begin{bmatrix}
\varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n
\end{bmatrix}.
$$

Al realizar la multiplicación $X\beta + \varepsilon$, se obtiene:

$$
\begin{bmatrix}
1 & x_1 \\
1 & x_2 \\
\vdots & \vdots \\
1 & x_n
\end{bmatrix}
\begin{bmatrix}
\beta_0 \\ \beta_1
\end{bmatrix}
+
\begin{bmatrix}
\varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n
\end{bmatrix}
=
\begin{bmatrix}
\beta_0 + \beta_1 x_1 + \varepsilon_1 \\
\beta_0 + \beta_1 x_2 + \varepsilon_2 \\
\vdots \\
\beta_0 + \beta_1 x_n + \varepsilon_n
\end{bmatrix}.
$$

Esto muestra que la expresión matricial $Y = X\beta + \varepsilon$ representa exactamente el modelo clásico:

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
$$

## Función de pérdida

La función de pérdida de mínimos cuadrados busca minimizar la suma de los errores al cuadrado:

$$
S(\beta) = (Y - X\beta)^\top (Y - X\beta)
$$

Al desarrollar el producto escalar, se obtiene:

$$
S(\beta) = Y^\top Y \;-\; 2\,\beta^\top X^\top Y \;+\; \beta^\top X^\top X\,\beta
$$

El término $\,\beta^\top X^\top Y\,$ es un escalar porque resulta del producto de un vector fila $Y^\top$ de dimensión $1\times n$, una matriz $X$ de $n\times p$, y un vector columna $\beta$ de $p\times 1$, dando un resultado $1\times 1$.

## Estimador de mínimos cuadrados

Para hallar los valores de $\beta$ que minimizan $S(\beta)$, se deriva con respecto a $\beta$:

$$
\frac{\partial S(\beta)}{\partial \beta}
=
-2X^\top Y
+
\frac{\partial}{\partial \beta}\big(\beta^\top X^\top X\,\beta\big).
$$

Usando $\dfrac{\partial}{\partial \beta}(\beta^\top A\beta)=(A+A^\top)\beta$ y que $X^\top X$ es simétrica:

$$
\frac{\partial S(\beta)}{\partial \beta} = -2X^\top Y + 2X^\top X\,\beta
$$

Igualando a cero para minimizar:

$$
-2X^\top Y + 2X^\top X\,\beta = 0
\quad\Rightarrow\quad
X^\top X\,\beta = X^\top Y
$$

Despejando:

$$
\hat{\beta} = (X^\top X)^{-1}X^\top Y
$$

## Desarrollo de los estimadores $\hat B_0$ y $\hat B_1$

En la regresión simple, la matriz $X$ tiene dos columnas: una de unos (intercepto) y otra con los valores $x_i$. Entonces:

$$
X =
\begin{bmatrix}
1 & x_1 \\
1 & x_2 \\
\vdots & \vdots \\
1 & x_n
\end{bmatrix},
\qquad
X^\top X =
\begin{bmatrix}
n & \sum x_i \\
\sum x_i & \sum x_i^2
\end{bmatrix}.
$$

De este modo:

$$
(X^\top X)^{-1} =
\frac{1}{\,n\sum x_i^2 - (\sum x_i)^2\,}
\begin{bmatrix}
\sum x_i^2 & -\sum x_i \\
-\sum x_i & n
\end{bmatrix},
\qquad
X^\top Y =
\begin{bmatrix}
\sum y_i \\ \sum x_i y_i
\end{bmatrix}.
$$

Multiplicando,

$$
\hat{\beta} =
\frac{1}{\,n\sum x_i^2 - (\sum x_i)^2\,}
\begin{bmatrix}
\sum x_i^2 \sum y_i - (\sum x_i)(\sum x_i y_i) \\
n\sum x_i y_i - (\sum x_i)(\sum y_i)
\end{bmatrix}.
$$

Estas expresiones son equivalentes a las fórmulas clásicas:

$$
\hat{\beta}_1 =
\frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2},
\qquad
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}
$$

## Invertibilidad de $X^\top X$

Para que la solución $\hat{\beta} = (X^\top X)^{-1}X^\top Y$ exista, $X^\top X$ debe ser invertible. En regresión simple:

$$
X^\top X =
\begin{bmatrix}
n & \sum x_i \\
\sum x_i & \sum x_i^2
\end{bmatrix},
\qquad
\det(X^\top X) = n\sum x_i^2 - (\sum x_i)^2 = n\sum (x_i - \bar{x})^2.
$$

El determinante de $X^\top X$ depende directamente de la variación de los $x_i$.  
Si todos los valores $x_i$ fueran iguales, $\sum (x_i - \bar{x})^2 = 0$ y el determinante también, impidiendo la inversión. Por tanto, si

$$
\det(X^\top X) > 0 \;\Leftrightarrow\; \sum (x_i - \bar{x})^2 > 0,
$$

la matriz es **invertible** y la solución $\hat{\beta}$ está bien definida. En regresión múltiple, esto se traduce en la *ausencia de multicolinealidad perfecta* entre las variables explicativas.


## Key takeaways
