# Capítulo 9: Interpretación aplicada

## El mapa general

Partimos de un modelo lineal en el que el precio de venta depende de varias características de la casa. En términos simples, el modelo estima cuánto cambia el precio promedio cuando modificamos una característica y mantenemos constantes las demás. Ese “cuánto” es el coeficiente, y es la pieza que vamos a transformar en un lenguaje cotidiano. Cuando digamos “manteniendo lo demás constante”, piensa en comparar dos viviendas iguales en todo, salvo en una dimensión específica, como el área habitable.

##  De coeficientes a mensajes que cualquiera entiende

Supongamos que el modelo nos indica que, por cada pie cuadrado adicional de área habitable, el precio promedio sube aproximadamente 75 dólares. Traducido a una unidad más natural, podríamos decir lo siguiente: un incremento de 100 ft² en área habitable se asocia con un aumento de 7.500 dólares en el precio. Esta es la regla de bolsillo que ayuda a negociar y a valorar: si dos casas son idénticas y una tiene 100 ft² más de área habitable, el modelo sugiere que la diferencia de precio esperada ronda esos 7.500 dólares.

Ahora bien, un número sin su incertidumbre es como una brújula sin escala. Todo coeficiente viene acompañado de un rango plausible, que llamamos intervalo de confianza. Si el margen de error nos lleva a un intervalo de 36 a 114 dólares por ft², entonces al hablar de 100 ft² adicionales diríamos que el aumento esperado va entre 3.600 y 11.400 dólares. De esta forma, no solo comunicamos un valor central, sino también la franja de resultados verosímiles según los datos y el modelo.

Cuando el modelo trabaja con el logaritmo del precio, la lectura cambia de “dólares” a “porcentajes”. En ese caso, un mismo aumento de 100 ft² implicaría un cambio porcentual aproximado en el precio. Esta forma de reporte es especialmente útil cuando se comparan mercados con niveles de precio muy distintos, porque el lenguaje porcentual es automáticamente escalable y facilita la comparación entre zonas o periodos.

También es importante hablar de escalas. No todas las variables están en unidades que el público percibe de manera inmediata. Estandarizar, por ejemplo, permite comparar qué variable “mueve” más el precio cuando subimos cada una en una desviación estándar. Pero para la comunicación cotidiana suele ser más natural regresar a unidades concretas, como 100 ft², un punto adicional en la calidad general o una plaza adicional de garaje. La clave es elegir unidades que el público pueda visualizar.

##  Lo que el modelo dice… y lo que no dice

Aquí conviene hacer una pausa pedagógica. La frase “100 ft² adicionales aumentan el precio promedio en 7.500 dólares” describe una asociación bajo el lente del modelo, no necesariamente una relación causal pura. Ese matiz no es un tecnicismo; es la diferencia entre usar el modelo como brújula y usarlo como sentencia. ¿Por qué no podemos prometer causalidad? Porque existe la posibilidad de que otras piezas, no medidas o no perfectamente medidas, estén empujando parte de esa relación.

Piensa en el vecindario. Si las casas más grandes tienden a ubicarse en zonas con mejores colegios o parques, una porción del “efecto” que atribuimos al tamaño podría ser en realidad el efecto de la localización. Lo mismo pasa con el estado de la vivienda: propietarios que invierten en acabados o eficiencia energética quizá también tienen más área, de modo que el coeficiente del tamaño absorbería un premio que corresponde a la calidad. Incluso los errores de medición —áreas mal registradas, calidades subjetivas— pueden distorsionar la lectura. Y, por si fuera poco, el retorno de añadir metros puede no ser lineal: en segmentos de lujo, 100 ft² extra no valen lo mismo que en segmentos medios, y el modelo lineal, si no se ajusta, simplifica una realidad que puede ser curvilínea.

Por eso, cuando comunicamos resultados, debemos ser explícitos: estamos describiendo asociaciones bajo un conjunto de supuestos. Si deseamos inferir causalidad, necesitamos diseños adicionales, como variables de control más ricas, efectos fijos de vecindario, términos no lineales, interacciones o, idealmente, estrategias cuasi experimentales.

## Variables omitidas que pueden cambiar la película

La localización fina es la protagonista silenciosa en casi todos los mercados inmobiliarios. No basta con una ciudad o un distrito; a veces son las diferencias de microbarrio las que inclinan la balanza. El estado interno de la vivienda también es crucial: reformas recientes en cocina y baños, mantenimiento, aislamiento térmico y luminosidad natural capturan atributos de confort que el tamaño por sí solo no abarca. Las condiciones del mercado —tasas de interés, estacionalidad, shocks locales— introducen variaciones temporales que, si no se controlan, pueden inflar o atenuar coeficientes. Y no olvidemos las posibles interacciones: el área puede “rendir” más cuando la calidad es alta, de modo que 100 ft² adicionales en una casa de calidad superior no equivalen al mismo aumento en otra de calidad básica. Estas capas invisibles explican por qué, al incluir dummies de vecindario o términos de interacción, muchas veces los coeficientes cambian de tamaño e incluso de signo.

## Cómo robustecer la interpretación sin perder sencillez

Una forma de ganar confianza es comparar especificaciones: empezar con un OLS base, añadir controles de vecindario, permitir curvaturas con polinomios o splines y permitir que variables clave interactúen. Cuando hay dudas sobre la variabilidad del error, conviene reportar errores estándar robustos como HC3. El bootstrap de coeficientes ofrece bandas de incertidumbre más realistas cuando nos apartamos un poco de los supuestos ideales. Además, las gráficas de efectos parciales son una herramienta didáctica poderosa: muestran cómo cambia el precio esperado cuando movemos una variable y mantenemos el resto en valores representativos, y permiten ver si la pendiente se mantiene o se aplana en distintos rangos. Finalmente, un análisis de sensibilidad a omitidas ayuda a cuantificar cuánta “fuerza” debería tener lo no observado para derribar la interpretación actual; es una manera transparente de reconocer lo que sabemos y lo que no.

## Recomendaciones prácticas para utilizar el modelo en el mundo real

Si estás comprando o vendiendo, piensa en el coeficiente como una regla de bolsillo con bandas. La cifra central de 7.500 dólares por 100 ft² te da una primera referencia para discutir, pero el intervalo de confianza te recuerda que hay variación plausible entre 3.600 y 11.400 dólares. No extrapoles alegremente a casos muy extremos, porque el retorno marginal del tamaño puede cambiar en viviendas demasiado pequeñas o demasiado grandes. Y recuerda que el valor de un pie cuadrado no vive en el vacío: el barrio, la calidad de los acabados y la distribución interna pueden multiplicar o reducir su impacto.

Si valoras propiedades de manera profesional, considera segmentar los efectos por tramos de área y reportar escenarios con distintos percentiles del mercado. Incorporar efectos fijos de vecindario y explorar interacciones entre área y calidad suele estabilizar la lectura y evitar confusiones. Comunicar siempre el intervalo de confianza y, cuando sea pertinente, el efecto en términos porcentuales si el modelo está en log-precio, ayuda a que distintos públicos comparen manzanas con manzanas.

Si miras esto desde la política pública, la historia sugiere que la plusvalía no es solo una cuestión de metros, sino de entorno. Intervenciones en infraestructura barrial, acceso a transporte, espacios verdes y calidad ambiental pueden generar retornos sociales considerables, a veces mayores que expandir metros sin mejorar el contexto. El modelo, bien leído, te indica dónde el metro adicional “rinde” más porque el entorno lo potencia.

## Señales de alerta que conviene revisar

Cuando al añadir controles de localización o calidad el coeficiente del área cambia de manera notable, estás observando el rastro del sesgo por variables omitidas. Si los residuos muestran patrones claros en función del tamaño, probablemente necesitas flexibilidad no lineal. Si unas pocas observaciones muy grandes dominan la estimación, conviene evaluar la robustez a valores influyentes. Y, sobre todo, recuerda que un R² alto no convierte al modelo en una máquina de causalidad; la prueba real está en la estabilidad de los efectos y en su coherencia al cruzar especificaciones y muestras.

## Cierre en un minuto

La frase “un incremento de 100 ft² se asocia con 7.500 dólares más en el precio” es una traducción práctica del coeficiente del área habitable. Tiene poder comunicativo porque da una magnitud operativa, pero debe ir acompañada de su intervalo de confianza y del recordatorio de que describe asociación, no causalidad garantizada. Las diferencias de barrio, las calidades internas, las no linealidades y las interacciones pueden modificar ese número, así que la buena práctica es contrastar modelos, reportar incertidumbre y presentar escenarios. Usado con criterio, el modelo ofrece una brújula confiable para decidir mejor, negociar con información y planear con perspectiva.
