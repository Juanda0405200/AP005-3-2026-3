"""
Programa principal: red neuronal 2-2-1 que aprende la función XOR.
Solo coordina los demás módulos; la lógica vive en cada uno de ellos.
"""
import config
import reporte
from red_neuronal import RedNeuronal
from entrenamiento import entrenar


def main():
    # El usuario puede cambiar la tasa de aprendizaje y las épocas máximas
    tasa_aprendizaje, max_epocas = config.pedir_parametros()

    for intento in range(1, config.MAX_INTENTOS + 1):
        print(f"=== Intento {intento} ===")
        red = RedNeuronal()                              # pesos aleatorios
        reporte.mostrar_parametros("Pesos y sesgos INICIALES:", red)

        print("\nEntrenando...")
        mse, epocas = entrenar(red, config.DATOS_XOR, tasa_aprendizaje,
                               max_epocas, config.MSE_OBJETIVO,
                               config.INTERVALO_REPORTE)

        if mse < config.MSE_ACEPTABLE:                   # la red aprendió
            break
        print("  La red quedó en un mínimo local; se reinicia con otros pesos.\n")

    print(f"\nEntrenamiento terminado en {epocas} épocas.")
    reporte.mostrar_parametros("Pesos y sesgos FINALES:", red)
    reporte.mostrar_resultados(red, config.DATOS_XOR, mse)


if __name__ == "__main__":
    main()
