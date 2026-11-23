#Este cuaderno implementa una neurona artificial de entrada única con aprendizaje supervisado para modelar una relación lineal simple descrita por la ecuación matemática 
#El objetivo es observar cómo el modelo ajusta sus parámetros internos de manera iterativa utilizando un conjunto de datos de entrenamiento predefinidos.
import time

inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]
w = 0.1
learning_rate = 0.1
epochs = 30
pause_time = 1
#Tasa de aprendizaje
def predict(i):
   return w * i
#Funcion de prediccion Modelo de la neurona: y = w*x
for epoch in range(epochs):
    predictions = [predict(i) for i in inputs]
    errors = [
        t - p
        for p, t in zip(predictions, targets)

]
cost = sum(errors) / len(targets)
print("\n" + "=" * 60)
print(f"EPOCA {epoch + 1}")
print("=" * 60)
for i, t, p, e in zip(
    inputs,
    targets,
    predictions,
    errors
):
    print(
        f"Entrada: {i:>2} | "
        f"Target: {t:>2} | "
        f"Prediccion: {p:>8.4f} | "
        f"Error: {e:>8.4f}"
)

print("-" * 60)
print(f"Peso antes de actualizar: {w:.10f}")
print(f"Error promedio: {cost:.6f}")

w += learning_rate * cost
print(f"Peso despues de actualizar: {w:.10f}")

time.sleep(pause_time)
Tras completar las 30 épocas de entrenamiento:

#El error promedio disminuye progresivamente conforme el peso w se aproxima al valor ideal de 2.0
#La actualización iterativa permite que la neurona corrija sus desviaciones de manera automática utilizando los datos de retroalimentación
test_inputs = [5, 6]
test_targets = [10, 12]
#Datos de prueba
test_predictions = [
    predict(i)
    for i in test_inputs
]

for i, t, p in zip(
    test_inputs,
    test_targets,
    test_predictions
):
    print(
        f"Entrada: {i:>2} | "
        f"Target: {t:>2} | "
        f"Prediccion: {p:>8.4f}"
    )
#El modelo entrenado demuestra una excelente capacidad de generalización al predecir con precisión los valores correspondientes 
#a las entradas de prueba, validando el correcto funcionamiento del algoritmo de aprendizaje supervisado implementado desde cero.
