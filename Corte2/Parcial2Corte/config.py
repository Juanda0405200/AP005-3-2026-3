"""
Módulo de configuración y datos.
Responsabilidad: concentrar los parámetros de entrenamiento y los datos del
problema XOR. Para cambiar el comportamiento del entrenamiento basta con
editar este archivo; la lógica de los demás módulos no se toca.
"""

# --- Parámetros de entrenamiento (modificables) ---
TASA_APRENDIZAJE = 0.5      # Qué tan grandes son los ajustes de los pesos
MAX_EPOCAS = 10000          # Número máximo de épocas de entrenamiento
MSE_OBJETIVO = 0.0005       # Si el MSE baja de este valor, se detiene el entrenamiento
INTERVALO_REPORTE = 1000    # Cada cuántas épocas se imprime el MSE

# --- Control de reinicios ---
# Con pesos aleatorios la red a veces queda atascada en un mínimo local.
MAX_INTENTOS = 10           # Máximo de reinicios con pesos nuevos
MSE_ACEPTABLE = 0.01        # MSE máximo para considerar que la red aprendió

# --- Datos de entrenamiento: tabla de verdad del XOR ---
# Cada elemento es (entradas, salida_esperada)
DATOS_XOR = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]


# --- Entrada de parámetros por el usuario ---
def _pedir_numero(mensaje, valor_por_defecto, tipo, minimo):
    """Pide un número por teclado. Enter conserva el valor por defecto.
    Si el dato no es válido, vuelve a preguntar."""
    while True:
        texto = input(f"{mensaje} [Enter = {valor_por_defecto}]: ").strip()
        if texto == "":
            return valor_por_defecto
        try:
            valor = tipo(texto)
        except ValueError:
            print("  Valor inválido, intenta de nuevo.")
            continue
        if valor <= minimo:
            print(f"  El valor debe ser mayor que {minimo}.")
            continue
        return valor


def pedir_parametros():
    """Pregunta al usuario si quiere cambiar los parámetros de entrenamiento.
    Devuelve (tasa_aprendizaje, max_epocas)."""
    respuesta = input("¿Deseas modificar los parámetros de entrenamiento? (s/n) [n]: ").strip().lower()
    if respuesta != "s":
        return TASA_APRENDIZAJE, MAX_EPOCAS
    tasa = _pedir_numero("Tasa de aprendizaje", TASA_APRENDIZAJE, float, 0)
    epocas = _pedir_numero("Número máximo de épocas", MAX_EPOCAS, int, 0)
    return tasa, epocas
