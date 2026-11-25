"""
Módulo de la red neuronal.
Responsabilidad: definir la ARQUITECTURA 2-2-1 (dos neuronas ocultas y una de
salida) y coordinar la propagación hacia adelante y la retropropagación del
error entre sus neuronas.
"""
from neurona import Neurona


class RedNeuronal:
    def __init__(self):
        # Capa oculta: 2 neuronas, cada una recibe las 2 entradas (x1, x2)
        self.capa_oculta = [Neurona(2), Neurona(2)]
        # Capa de salida: 1 neurona, recibe las salidas de las 2 ocultas
        self.neurona_salida = Neurona(2)

    def propagar_adelante(self, entradas):
        """Calcula la salida de la red para una entrada dada."""
        salidas_ocultas = [n.calcular_salida(entradas) for n in self.capa_oculta]
        return self.neurona_salida.calcular_salida(salidas_ocultas)

    def retropropagar(self, salida_esperada):
        """
        Propaga el error hacia atrás: primero la neurona de salida y luego las
        ocultas, que reciben el error ponderado por el peso que las conecta
        con la salida. Solo calcula los deltas; no modifica pesos todavía.
        """
        error = self.neurona_salida.salida - salida_esperada
        self.neurona_salida.calcular_delta(error)
        for i, neurona in enumerate(self.capa_oculta):
            error_oculta = self.neurona_salida.delta * self.neurona_salida.pesos[i]
            neurona.calcular_delta(error_oculta)

    def actualizar_pesos(self, tasa_aprendizaje):
        """Aplica el ajuste de pesos y sesgos en todas las neuronas."""
        for neurona in self.capa_oculta:
            neurona.ajustar_parametros(tasa_aprendizaje)
        self.neurona_salida.ajustar_parametros(tasa_aprendizaje)

    def obtener_parametros(self):
        """Devuelve [(nombre, pesos, sesgo)] de las 3 neuronas, para mostrarlos."""
        neuronas = [("Neurona 1 (oculta)", self.capa_oculta[0]),
                    ("Neurona 2 (oculta)", self.capa_oculta[1]),
                    ("Neurona 3 (salida)", self.neurona_salida)]
        return [(nombre, n.pesos, n.sesgo) for nombre, n in neuronas]
