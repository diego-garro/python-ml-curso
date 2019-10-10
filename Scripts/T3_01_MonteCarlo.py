#-*- coding: utf-8 -*-

# - Simulación de Monte Carlo para aproximar al numero Pi
# - Generamos dos numeros aleatorios x e y entre 0 y 1
# - Calculamos x * x + y * y
#       * Si el valor es inferior a 1 -> estamos dentro del círculo
#       * Si el valor es superior a 1 -> estamos fuera del círculo
# - Calculamos el numerp total de veces que caen dentro del círculo y
# lo dividimos entre el numero total de intentos para obtener una
# aproximación de la probabilidad de caer dentro del círculo.
# - Usamos esa probabilidad para aproximar el valor de Pi.
# - Repetimos el experimento un numero suficiente de veces para obtener
# diferentes aproximaciones de Pi (por ejemplo 100 veces).
# - Calculamos el promedio de los experimentos anteriores para obtener
# un valor final de Pi.

import numpy as np
import matplotlib.pyplot as plt

def pi_montecarlo(n, n_exp):
    pi_avg = 0
    pi_value_list = []
    for i in range(n_exp):
        value = 0
        x = np.random.uniform(0,1,n).tolist()
        y = np.random.uniform(0,1,n).tolist()
        for j in range(n):
            z = np.sqrt(x[j]**2 + y[j]**2)
            if z <= 1:
                value += 1
        float_value = float(value)
        pi_value = float_value * 4 / n
        pi_value_list.append(pi_value)
        pi_avg += pi_value

    pi = pi_avg / n_exp
    fig = plt.plot(pi_value_list)
    return(pi, fig)

pi, fig = pi_montecarlo(10000, 300)
print('El valor de Pi calculado es: {}'.format(pi))
plt.show()
