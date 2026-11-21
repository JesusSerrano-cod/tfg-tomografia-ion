# Simulación de la tomografía cuántica de un qubit

Código de las simulaciones del Trabajo Fin de Grado *Tomografía cuántica con un ion atrapado* (Doble Grado en Física y Matemáticas, Universidad de Granada, 2026), de Jesús Serrano Rodríguez.

## Qué hace

El script `simulacion_tomografia.py` simula la tomografía de estado de un qubit en el caso ideal: preparación, rotaciones y detección perfectas, de modo que la única fuente de error es el ruido de proyección debido al número finito de repeticiones.

Para cada uno de los seis estados cardinales y para distintos valores del número de repeticiones por eje $N$, el script:

1. Simula los conteos de las medidas en las bases $x$, $y$ y $z$ con una distribución binomial.
2. Reconstruye el estado con el **estimador lineal**, $\hat r_i = 2n_{i,+}/N - 1$.
3. Reconstruye el estado con el **estimador de máxima verosimilitud (EMV)**: si la reconstrucción lineal es un estado físico, se toma como EMV; si no, se maximiza la verosimilitud con la parametrización de Cholesky de la matriz densidad y el algoritmo L-BFGS-B, partiendo del estado completamente mezcla.
4. Repite el proceso $M$ veces y calcula el error medio en el vector de Bloch de ambos estimadores, la infidelidad media del EMV y la fracción de reconstrucciones lineales no físicas.

## Requisitos

- Python 3.9 o superior
- numpy
- scipy
- matplotlib
- qiskit

Se pueden instalar con:

```
pip install numpy scipy matplotlib qiskit
```

## Uso

```
python simulacion_tomografia.py
```

El script muestra por pantalla una tabla con los resultados para cada $N$ y guarda la figura `fig_simulacion.pdf` con el error frente a $N$ y la fracción de reconstrucciones no físicas.

Los parámetros de la simulación se modifican al final del script:

- `lista_N`: valores del número de repeticiones por eje.
- `M`: número de repeticiones del experimento de tomografía completo para cada estado y cada $N$ (por defecto, 400).

Con los valores por defecto la ejecución tarda unos minutos. La semilla del generador de números aleatorios es fija, de modo que los resultados son reproducibles.

## Licencia

MIT
