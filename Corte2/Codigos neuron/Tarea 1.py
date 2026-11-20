# Lista de entradas (x) para entrenar el modelo.
inputs = [1, 2, 3, 4]

# Lista de valores deseados u objetivos (y) correspondientes a cada entrada (el doble de cada entrada).
targets = [2, 4, 6, 8]

# Peso inicial de la red (parámetro ejecutor). Comienza en 0.1 y el modelo aprenderá a llevarlo a 2.0.
w = 0.1

# Tasa de aprendizaje: determina qué tan grandes son los pasos que da el modelo para ajustar el peso 'w'.
learning_rate = 0.1

# Función de predicción del modelo lineal: realiza la operación y = w * x (sin sesgo/bias).
def predict(i):
  return w*i

# Bloque de entrenamiento de la red
# Se ejecuta un bucle de 30 iteraciones (épocas) para ajustar progresivamente el peso 'w'.
for _ in range(30):
  # Genera una lista de predicciones multiplicando cada valor de 'inputs' por el peso 'w' actual.
  pred = [predict(i) for i in inputs]
  
  # Calcula el error de cada muestra restando la predicción individual (p) al valor real esperado (t).
  errors = [t - p for p, t in zip(pred, targets)]
  
  # Promedia los errores de todas las muestras para obtener la métrica de costo general de esta época.
  cost = sum(errors)/len(targets)
  
  # Imprime la lista de objetivos esperados.
  print(f"Targets: ", targets)
  
  # Imprime las predicciones realizadas en la iteración actual.
  print(f"Predictions: ", pred)
  
  # Imprime los errores calculados para cada una de las entradas.
  print(f"Errors: ", errors)
  
  # Muestra el estado del peso actual con 10 decimales y el costo con 6 decimales.
  print(f"Weight: {w: .10f}, Cost: {cost:.6f}")
  
  # Actualiza el peso 'w' sumándole el error promedio escalado por la tasa de aprendizaje.
  w += learning_rate*cost

# Bloque de prueba para verificar cómo se comporta la red con datos no vistos en el entrenamiento.
# Entradas de prueba (x).
test_inputs = [5, 6]

# Valores esperados para las pruebas (y = 2x).
test_targets = [10, 12]

# Genera las predicciones para las entradas de prueba usando el peso 'w' ya entrenado.
pred = [predict(i) for i in test_inputs]

# Muestra los resultados finales comparando entrada, objetivo real y predicción (con 4 decimales).
for i, t, p in zip(test_inputs, test_targets, pred):
  print(f"input:{i}, target:{t}, pred:{p:.4f}")
