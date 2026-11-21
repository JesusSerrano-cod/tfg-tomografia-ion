import numpy as np  #Para hacer cálculos
from scipy.optimize import minimize     # Para implementar la MLE
import matplotlib.pyplot as plt         # Para dibujar gráficas
from qiskit.quantum_info import DensityMatrix, Operator, state_fidelity       # Para construir una matriz densidad

# =============================================================================
# SIMULACIÓN DE LA TOMOGRAFÍA DE UN QUBIT EN EL CASO IDEAL
# Preparación, rotaciones y detección perfectas: la única fuente de error es el
# ruido de proyección (número finito N de repeticiones por eje).
# Para cada uno de los 6 estados cardinales y cada N se repite M veces:
#   1) se simulan los conteos de cada eje con una distribución binomial
#   2) se reconstruye el estado con el método lineal
#   3) se reconstruye el estado con máxima verosimilitud (MLE)
# =============================================================================

rng = np.random.default_rng(2026)   # Semilla fija para que los resultados sean reproducibles

# Definimos las Pauli como objetos Operator
I = Operator.from_label('I')
X = Operator.from_label('X')
Y = Operator.from_label('Y')
Z = Operator.from_label('Z')

# Creamos los estados usando etiquetas estándar
# '0' y '1' -> Base Z
# '+' y '-' -> Base X
# 'r' y 'l' -> Base Y (right/left, que son +i y -i)
# Son a la vez los 6 estados que se preparan y los proyectores de la medida

proyectores = {
    'PZ+': DensityMatrix.from_label('0'),
    'PZ-': DensityMatrix.from_label('1'),
    'PX+': DensityMatrix.from_label('+'),
    'PX-': DensityMatrix.from_label('-'),
    'PY+': DensityMatrix.from_label('r'),
    'PY-': DensityMatrix.from_label('l')
}

# Convertimos los valores del diccionario a una lista fija (mismo orden que los conteos)
lista_proyectores = list(proyectores.values())


# ----------------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------------

def vector_bloch(rho):
    """Vector de Bloch r_i = tr(rho sigma_i) de una matriz densidad (array 2x2)."""
    return np.real(np.array([np.trace(rho @ X.data),
                             np.trace(rho @ Y.data),
                             np.trace(rho @ Z.data)]))


def simular_conteos(estado, Ncuentas):
    """Simula los conteos de un experimento ideal.
    En cada eje el número de resultados '+' sigue una binomial de parámetros
    N y p = tr(rho P_+). Devuelve los conteos en el mismo orden que la lista
    de proyectores: [z+, z-, x+, x-, y+, y-]."""
    conteos = []
    for clave in ['PZ+', 'PX+', 'PY+']:
        p_mas = np.real(np.trace(estado.data @ proyectores[clave].data))
        p_mas = min(max(p_mas, 0.0), 1.0)          # Evita errores de redondeo
        n_mas = rng.binomial(Ncuentas, p_mas)
        conteos += [n_mas, Ncuentas - n_mas]
    return np.array(conteos, dtype=float)


def reconstruccion_lineal(conteos, Ncuentas):
    """Método lineal: r_i = 2 n_{i,+}/N - 1 y rho = (I + r.sigma)/2."""
    valor_esperado_z = 2*conteos[0]/Ncuentas - 1
    valor_esperado_x = 2*conteos[2]/Ncuentas - 1
    valor_esperado_y = 2*conteos[4]/Ncuentas - 1
    densidadlin = 0.5 * (I + valor_esperado_x * X + valor_esperado_y*Y + valor_esperado_z*Z)
    return densidadlin.data


def es_fisica(rho):
    """Comprueba que rho es semidefinida positiva (la traza ya vale 1 por construcción)."""
    valores_propios = np.linalg.eigvalsh(rho)
    return valores_propios.min() >= -1e-12


def t_to_rho(t):
    """
    Convierte 4 parámetros reales en una matriz densidad física.
    t[0], t[1] -> diagonales (reales)
    t[2], t[3] -> parte real e imaginaria del elemento fuera de la diagonal
    """
    # Construimos T
    T = np.array([
        [t[0], 0],
        [complex(t[2], t[3]), t[1]]
    ])

    # Calculamos T_daga @ T
    T_dag = T.conj().T
    rho_no_norm = T_dag @ T

    # Normalizamos para que la traza sea 1
    rhomles = rho_no_norm / np.trace(rho_no_norm)
    return rhomles


def log_likelihood_negativa(t_params, datos, proyectores):
    # 1. Generamos rho física a partir de los parámetros de Cholesky
    rho = t_to_rho(t_params)

    log_L = 0
    for i, p in enumerate(proyectores):
        # 2. Probabilidad teórica según nuestro rho actual
        p_teorica = np.real(np.trace(rho @ p.data))

        # 3. Evitamos el log(0) sumando un número ínfimo
        p_teorica = max(p_teorica, 1e-15)

        # 4. Sumamos: conteos * log(probabilidad)
        log_L += datos[i] * np.log(p_teorica)

    return -log_L # Retornamos el negativo para minimizar


def optimizar_cholesky(conteos):
    """Minimiza la log-verosimilitud negativa con L-BFGS-B partiendo de rho = I/2."""
    t_init = np.array([1.0, 1.0, 0.0, 0.0])
    cotas = [(None, None), (None, None), (None, None), (None, None)]   
    res = minimize(
        log_likelihood_negativa,
        t_init,
        args=(conteos, lista_proyectores),
        method='L-BFGS-B',
        bounds=cotas,
        options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 500}
    )
    return t_to_rho(res.x)


def reconstruccion_mle(conteos, Ncuentas):
    """Estimador de máxima verosimilitud.
    1) Si la reconstrucción lineal es un estado físico, coincide con el EMV
       (proposición del capítulo 3) y no hace falta optimizar.
    2) Si no, se maximiza la verosimilitud con la parametrización de Cholesky."""
    rho_lin = reconstruccion_lineal(conteos, Ncuentas)
    if es_fisica(rho_lin):
        return rho_lin
    return optimizar_cholesky(conteos)


# ----------------------------------------------------------------------------
# Simulación
# ----------------------------------------------------------------------------

def simular(estado, Ncuentas, M):
    """Repite M veces el experimento para un estado y un N dados.
    Devuelve el error medio de cada estimador, la fidelidad media del EMV y
    la fracción de reconstrucciones lineales no físicas."""
    r_verdadero = vector_bloch(estado.data)
    err_lin, err_mle, fidelidad_mle, no_fisicas = [], [], [], 0

    for _ in range(M):
        conteos = simular_conteos(estado, Ncuentas)

        rho_lin = reconstruccion_lineal(conteos, Ncuentas)
        rho_mle = reconstruccion_mle(conteos, Ncuentas)

        if not es_fisica(rho_lin):
            no_fisicas += 1

        # Error euclídeo entre vectores de Bloch (= 2 x distancia traza)
        err_lin.append(np.linalg.norm(vector_bloch(rho_lin) - r_verdadero))
        err_mle.append(np.linalg.norm(vector_bloch(rho_mle) - r_verdadero))

        # Fidelidad (convención de Qiskit, con el cuadrado). Solo para el EMV,
        # que siempre es un estado físico
        fidelidad_mle.append(state_fidelity(estado, DensityMatrix(rho_mle)))

    return np.mean(err_lin), np.mean(err_mle), np.mean(fidelidad_mle), no_fisicas / M


if __name__ == '__main__':
    lista_N = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]   # Repeticiones por eje
    M = 400                                                    # Repeticiones del experimento

    resultados = {}
    for Ncuentas in lista_N:
        # Los 6 estados son equivalentes por simetría: se promedian los resultados
        por_estado = np.array([simular(estado, Ncuentas, M) for estado in proyectores.values()])
        resultados[Ncuentas] = por_estado.mean(axis=0)
        print(f"N = {Ncuentas:5d}   error lineal = {resultados[Ncuentas][0]:.4f}   "
              f"error EMV = {resultados[Ncuentas][1]:.4f}   F (EMV) = {resultados[Ncuentas][2]:.2e}   "
              f"lineal no física = {100*resultados[Ncuentas][3]:.1f} %")

    # ------------------------------------------------------------------------
    # Figura: (a) error frente a N, (b) fracción de reconstrucciones no físicas
    # ------------------------------------------------------------------------
    N_arr = np.array(lista_N)
    err_lin = np.array([resultados[n][0] for n in lista_N])
    err_mle = np.array([resultados[n][1] for n in lista_N])
    no_fis = np.array([resultados[n][3] for n in lista_N])

    AZUL, NARANJA, GRIS = '#2a78d6', '#eb6834', '#52514e'
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9})

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axs[0]
    ax.loglog(N_arr, err_lin, 'o-', color=AZUL, lw=1.2, ms=3.5, label='lineal')
    ax.loglog(N_arr, err_mle, 's-', color=NARANJA, lw=1.2, ms=3.5, label='EMV')
    ax.set_xlabel('Repeticiones por eje, $N$')
    ax.set_ylabel(r'Error $\|\hat{\mathbf{r}}-\mathbf{r}\|$')
    ax.legend(frameon=False, fontsize=8)
    ax.text(0, 1.04, '(a)', transform=ax.transAxes)

    ax = axs[1]
    ax.semilogx(N_arr, no_fis, 'o-', color=AZUL, lw=1.2, ms=3.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Repeticiones por eje, $N$')
    ax.set_ylabel('Fracción no física (lineal)')
    ax.text(0, 1.04, '(b)', transform=ax.transAxes)

    for a in axs:
        a.spines[['top', 'right']].set_visible(False)
        a.grid(alpha=.15, lw=.5)
    fig.tight_layout()
    fig.savefig('fig_simulacion.pdf')
    plt.show()
