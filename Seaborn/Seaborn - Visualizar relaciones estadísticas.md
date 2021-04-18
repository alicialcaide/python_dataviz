```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
```


```python
titanic = sb.load_dataset("titanic")
titanic.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



<b>Diagramas de dispersión categóricos</b>

Relación entre la tarifa que pagó cada viajero respecto a la clase en la que viajaba. Además, se hace distinción de si eran hombres o mujeres.

Algunos de los parámetros que podemos usar para completar el plot serían: 
* hue: para crear una leyenda.
* jitter: <b>True</b> para controlar la magnitud de la fluctuación o <b>False</b> para desactivarlo.
* kind="swarm" : ajusta los puntos a lo largo del eje categórico mediante un algoritmo que evita que se superpongan. Puede proporcionar una mejor representación de la distribución de observaciones, aunque solo funciona bien para conjuntos de datos relativamente pequeños.
* El orden de las etiquetas se puede cambiar usando order= ["etiqueta 1", "etiqueta 2"]



```python
sb.catplot(x="fare", y="class", jitter=True, hue="sex", data=titanic)
```




    <seaborn.axisgrid.FacetGrid at 0x1a7eadd6d00>




![png](output_3_1.png)


<b>Distribuciones de observaciones dentro de categorías</b>

A medida que aumenta el tamaño del conjunto de datos, los diagramas de dispersión categóricos se vuelven limitados en la información que pueden proporcionar sobre la distribución de valores dentro de cada categoría. Cuando esto sucede, existen varios enfoques para resumir la información de distribución de manera que faciliten las comparaciones entre los niveles de categoría.

<b>Boxplots</b>

Este tipo de gráfico muestra los valores de los tres cuartiles de la distribución junto con los valores extremos.
* Al añadir kind="box" podemos visulizar el boxplot. 
* Cuando añadimos una leyenda con hue="sex", el cuadro para cada nivel de la variable semántica se mueve a lo largo del eje categórico para que no se superpongan. Este comportamiento se llama "dodging" (esquivar) y está activado de forma predeterminada porque se supone que la variable semántica está anidada dentro de la variable categórica principal. Si ese no es el caso, se puede desactivar la evasión con dodge=False.

<code>boxplot()</code>


```python
#Usando catplot + kind = "box"
sb.catplot(x="class", y="age", kind="box", hue="sex", data=titanic)

sb.catplot(x="class", y="age", kind="box", hue="sex", dodge=False, data=titanic)
```




    <seaborn.axisgrid.FacetGrid at 0x1a7ebfe3fd0>




![png](output_5_1.png)



![png](output_5_2.png)


<b>Boxenplot</b>

Dibuja una gráfica que es similar a una gráfica de caja pero optimizada para mostrar más información sobre la forma de la distribución. Es más adecuado para conjuntos de datos más grandes.

<code>boxenplot()</code>


```python
#Usando catplot + kind = "boxen"
sb.catplot(x="class", y="age", kind="boxen", data=titanic)
```




    <seaborn.axisgrid.FacetGrid at 0x1a7ed7e8c70>




![png](output_7_1.png)



```python
#Usando boxenplot()
sb.boxenplot(x="class", y="age", data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a7ed71a220>




![png](output_8_1.png)


<b>Violenplot</b>

Combina un diagrama de caja con el procedimiento de estimación de la densidad del kernel. Este enfoque utiliza la estimación de la densidad del kernel para proporcionar una descripción más completa de la distribución de valores. Además, los valores de cuartiles y bigotes del diagrama de caja se muestran dentro del violín. La desventaja es que, debido a que la trama de violín usa un KDE, hay algunos otros parámetros que pueden necesitar ajustes, lo que agrega cierta complejidad en relación con la gráfica de caja sencilla.

Usando split = True también es posible "dividir" los violines cuando el parámetro de hue tiene solo dos niveles, lo que puede permitir un uso más eficiente del espacio. 

Finalmente, hay varias opciones para la trama que se dibuja en el interior de los violines, incluidas las formas de mostrar cada observación individual en lugar de los valores de resumen de la gráfica de caja. Para esto usamos inner="stick".

<code>violenplot()</code>


```python
sb.catplot(x="age", y="class", kind="violin", hue="sex", split=True, inner="stick", data=titanic)
```




    <seaborn.axisgrid.FacetGrid at 0x1a7eda79af0>




![png](output_10_1.png)


<b>Estimación estadística dentro de categorías</b>

Para otras aplicaciones, en lugar de mostrar la distribución dentro de cada categoría, es posible que desee mostrar una estimación de la tendencia central de los valores. Seaborn tiene dos formas principales de mostrar esta información. Es importante destacar que la API básica para estas funciones es idéntica a la de las mencionadas anteriormente.

<b> Bar plots </b>

En seaborn, la función <code>barplot ()</code> opera en un conjunto de datos completo y aplica una función para obtener la estimación (tomando la media por defecto). Cuando hay varias observaciones en cada categoría, también usa bootstrapping para calcular un intervalo de confianza alrededor de la estimación, que se traza usando barras de error:


```python
sb.catplot(x="class", y="age", hue="sex", kind="bar", data=titanic)
```




    <seaborn.axisgrid.FacetGrid at 0x1a7ef341a30>




![png](output_12_1.png)


Un caso especial para el diagrama de barras es cuando desea mostrar el número de observaciones en cada categoría en lugar de calcular una estadística para una segunda variable. Esto es similar a un histograma sobre una variable categórica, en lugar de cuantitativa. En seaborn, es fácil hacerlo con la función <code>countplot ()</code>:


```python
#Usando catplot + kind = count
sb.catplot(x="who", kind="count", data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a7ef7170d0>




![png](output_14_1.png)



```python
#Usado directamente countplot()
sb.countplot(x="who", data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a7ef7a60d0>




![png](output_15_1.png)


Tanto <code>barplot ()</code> como <code>countplot ()</code> se pueden invocar con todas las opciones discutidas anteriormente, junto con otras que se muestran en la documentación detallada de cada función:


```python
sb.catplot(y="who", hue="class", kind="count", data=titanic)
```




    <seaborn.axisgrid.FacetGrid at 0x1a7f08c3700>




![png](output_17_1.png)


<b>Point plots</b>

La función <code>pointplot ()</code> ofrece un estilo alternativo para visualizar la misma información. Esta función también codifica el valor de la estimación con la altura en el otro eje, pero en lugar de mostrar una barra completa, traza la estimación puntual y el intervalo de confianza. Además, <code>pointplot ()</code> conecta puntos de la misma categoría de tono. Esto hace que sea fácil ver cómo la relación principal está cambiando en función de la semántica de tono.


```python
#Usando catplot + kind = point
sb.catplot(x="sex", y="survived", kind="point", hue="class", data=titanic)
```




    <seaborn.axisgrid.FacetGrid at 0x1a7ed9119a0>




![png](output_19_1.png)



```python
#Usando pointplot()
sb.pointplot(x="sex", y="survived", hue="class", data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a7ec181e20>




![png](output_20_1.png)


Si bien las funciones categóricas carecen de la semántica de estilo de las funciones relacionales, aún puede ser una buena idea variar el marcador y / o el estilo de línea junto con el tono para hacer que las figuras sean lo más accesibles y se reproduzcan bien en blanco y negro.


```python
#Usando catplot + kind = point
sb.catplot(x="class", y="survived", hue="sex", markers=["^", "o"], linestyles=["-", "--"], kind="point", data=titanic)
```




    <seaborn.axisgrid.FacetGrid at 0x1a7f0fadd90>




![png](output_22_1.png)


<b>Plotting “wide-form” data</b>

Si bien se prefiere el uso de datos de "formato largo" o "ordenados", estas funciones también se pueden aplicar a datos de "formato amplio" en una variedad de formatos, incluidos Pandas DataFrames o matrices numpy bidimensionales. Estos objetos deben pasarse directamente al parámetro de datos:


```python
sb.catplot(data=titanic, orient="h", kind="box")
```




    <seaborn.axisgrid.FacetGrid at 0x1a7f20bba90>




![png](output_24_1.png)


Además, las funciones a nivel de ejes aceptan vectores de Pandas u objetos numpy en lugar de variables en un <code>DataFrame</code>:


```python
sb.violinplot(x=titanic.sex, y=titanic.fare)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a7f2fa9790>




![png](output_26_1.png)


Es posible modificar el tamaño y la forma de los gráficos configurándolo con los comandos de matplotlib:


```python
f, ax = plt.subplots(figsize=(10, 5))
sb.countplot(y="class", data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a7f40ef490>




![png](output_28_1.png)


<b>Showing multiple relationships with facets</b>

<code>catplot ()</code> se basa en FacetGrid, lo que significa que es fácil agregar variables de facetado para visualizar relaciones de dimensiones superiores:


```python
sb.catplot(x="class", y="fare", hue="who", col="sex", aspect=.7, jitter=True, data=titanic)
```




    <seaborn.axisgrid.FacetGrid at 0x1a7f44f30d0>




![png](output_30_1.png)

