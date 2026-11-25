"""
Módulo de reporte.
Responsabilidad: toda la salida por pantalla (pesos iniciales y finales, MSE
durante el entrenamiento y resultados finales del XOR). Así el resto de los
módulos se dedican solo a calcular.
"""


def mostrar_parametros(titulo, red):
    """Imprime los pesos y sesgos de cada neurona."""
    print(f"\n{titulo}")
    for nombre, pesos, sesgo in red.obtener_parametros():
        pesos_redondeados = [round(w, 4) for w in pesos]
        print(f"  {nombre}: pesos = {pesos_redondeados}, sesgo = {sesgo:.4f}")


def mostrar_mse(epoca, mse):
    """Imprime el error cuadrático medio de una época."""
    print(f"  Época {epoca:6d} | MSE = {mse:.6f}")


def mostrar_resultados(red, datos, mse_final):
    """Imprime la salida de la red para las cuatro combinaciones del XOR."""
    print("\nResultados finales del XOR:")
    print("  x1 x2 | esperado | salida de la red | redondeada")
    for entradas, esperado in datos:
        salida = red.propagar_adelante(entradas)
        print(f"  {entradas[0]}  {entradas[1]}  |    {esperado}     |     {salida:.4f}       |     {round(salida)}")
    print(f"\nError cuadrático medio final: {mse_final:.6f}")
