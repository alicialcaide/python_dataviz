# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 22:00:10 2021

@author: Ali
"""

## Visualizar relaciones entre datos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#Cargamos el conjunto de datos
planets = sb.load_dataset("planets")
planets.head(5)

#Relacionar variables con diagramas de dispersión

sb.relplot(x="distance", y="mass", hue="method", data=planets)

sb.relplot(x="distance", y="mass", hue="year", style="number", data=planets)

sb.relplot(x="distance", y="mass", hue="year", palette="ch:r=-.5,l=.75", data=planets)

sb.relplot(x="distance", y="mass", hue="year", size="year", sizes=(15,150), data=planets)

#Enfatizando la continuidad con diagramas de líneas

a= sb.relplot(x="distance", y="mass", kind="line", data=planets)
a.fig.autofmt_xdate()

a= sb.relplot(x="distance", y="mass", kind="line", sort=False, data=planets)
a.fig.autofmt_xdate()

