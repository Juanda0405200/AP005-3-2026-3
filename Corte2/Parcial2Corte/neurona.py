"""
Módulo de la neurona artificial.
Responsabilidad: representar UNA neurona individual (pesos, sesgo y función
de activación) y saber calcular su salida y ajustar sus propios parámetros.
No sabe nada de la red ni del XOR.
"""
import math
import random


class Neurona:
    def __init__(self, n_entradas):
        # Inicialización aleatoria de pesos y sesgo en el rango [-1, 1]
        self.pesos = [random.uniform(-1, 1) for _ in range(n_entradas)]
        self.sesgo = random.uniform(-1, 1)
        # Valores guardados del último cálculo (se necesitan para el aprendizaje)
        self.entradas = []
        self.salida = 0.0
        self.delta = 0.0

    @staticmethod
    def sigmoide(x):
        """Función de activación: comprime cualquier valor al rango (0, 1)."""
        return 1 / (1 + math.exp(-x))

    def calcular_salida(self, entradas):
        """Propagación hacia adelante de esta neurona: sigmoide(suma ponderada + sesgo)."""
        self.entradas = entradas
        suma = sum(w * x for w, x in zip(self.pesos, entradas)) + self.sesgo
        self.salida = self.sigmoide(suma)
        return self.salida

    def calcular_delta(self, error):
        """
        Calcula el 'delta' (responsabilidad de esta neurona en el error).
        error = diferencia que llega a la neurona; la derivada de la sigmoide
        es salida * (1 - salida).
        """
        self.delta = error * self.salida * (1 - self.salida)

    def ajustar_parametros(self, tasa_aprendizaje):
        """Descenso del gradiente: corrige pesos y sesgo usando el delta."""
        for i in range(len(self.pesos)):
            self.pesos[i] -= tasa_aprendizaje * self.delta * self.entradas[i]
        self.sesgo -= tasa_aprendizaje * self.delta
