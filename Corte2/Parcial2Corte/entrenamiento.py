"""
Módulo de entrenamiento.
Responsabilidad: ejecutar el ciclo de aprendizaje (épocas), calcular el error
cuadrático medio (MSE) y decidir cuándo detenerse. No conoce los detalles
internos de las neuronas: solo usa la interfaz de la red.
"""
import reporte


def calcular_mse(red, datos):
    """Error cuadrático medio de la red sobre todos los datos."""
    suma_errores = 0.0
    for entradas, esperado in datos:
        salida = red.propagar_adelante(entradas)
        suma_errores += (salida - esperado) ** 2
    return suma_errores / len(datos)


def entrenar(red, datos, tasa_aprendizaje, max_epocas, mse_objetivo, intervalo_reporte):
    """
    Entrena la red. En cada época recorre todos los datos aplicando:
    propagación hacia adelante -> retropropagación -> actualización de pesos.
    Devuelve (mse_final, epocas_realizadas).
    """
    mse = calcular_mse(red, datos)
    reporte.mostrar_mse(0, mse)

    for epoca in range(1, max_epocas + 1):
        for entradas, esperado in datos:
            red.propagar_adelante(entradas)
            red.retropropagar(esperado)
            red.actualizar_pesos(tasa_aprendizaje)

        mse = calcular_mse(red, datos)
        if epoca % intervalo_reporte == 0:
            reporte.mostrar_mse(epoca, mse)

        if mse < mse_objetivo:       # Criterio de parada anticipada
            reporte.mostrar_mse(epoca, mse)
            return mse, epoca

    return mse, max_epocas
