# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

titanic = sb.load_dataset("titanic")
titanic.head(5)

## Diagramas de dispersión categóricos

sb.catplot(x="fare", y="class", jitter=True, hue="sex", data=titanic)

## Distribuciones de observaciones dentro de categorías

# Usando catplot + kind = "box"
sb.catplot(x="class", y="age", kind="box", hue="sex", data=titanic)

sb.catplot(x="class", y="age", kind="box", hue="sex", dodge=False, data=titanic)

## Boxenplot

# Usando catplot + kind = "boxen"
sb.catplot(x="class", y="age", kind="boxen", data=titanic)

# Usando boxenplot()
sb.boxenplot(x="class", y="age", data=titanic)

## Violenplot

sb.catplot(x="age", y="class", kind="violin", hue="sex", split=True, inner="stick", data=titanic)

## Estimación estadística dentro de categorías

sb.catplot(x="class", y="age", hue="sex", kind="bar", data=titanic)

# Usando catplot + kind = count
sb.catplot(x="who", kind="count", data=titanic)

# Usado directamente countplot()
sb.countplot(x="who", data=titanic)

sb.catplot(y="who", hue="class", kind="count", data=titanic)

# Usando catplot + kind = point
sb.catplot(x="sex", y="survived", kind="point", hue="class", data=titanic)

# Usando pointplot()
sb.pointplot(x="sex", y="survived", hue="class", data=titanic)

# Usando catplot + kind = point
sb.catplot(x="class", y="survived", hue="sex", markers=["^", "o"], linestyles=["-", "--"], kind="point", data=titanic)

# Plotting “wide-form” data
sb.catplot(data=titanic, orient="h", kind="box")

sb.violinplot(x=titanic.sex, y=titanic.fare)

f, ax = plt.subplots(figsize=(10, 5))
sb.countplot(y="class", data=titanic)

# Mostrar múltiples relaciones

sb.catplot(x="class", y="fare", hue="who", col="sex", aspect=.7, jitter=True, data=titanic)
