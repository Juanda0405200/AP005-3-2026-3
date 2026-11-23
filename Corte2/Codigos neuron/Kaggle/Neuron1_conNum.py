import time
import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([1, 2, 3, 4], dtype=float)
targets = np.array([2, 4, 6, 8], dtype=float)

w = 0.1
learning_rate = 0.1
epochs = 30
pause_time = 0.2

weight_history = []
error_history = []
#Predicción vectorizada: Multiplica el peso por el arreglo completo de entradas de forma simultánea. 
#Cálculo de Errores: Obtiene la diferencia entre los objetivos y las predicciones, calculando el error
#promedio con funciones agregadas de NumPy. Actualización de Pesos: Modifica el parámetro aplicando la 
#regla de aprendizaje. Almacenamiento: Guarda los valores del peso y del error en listas independientes 
#para su posterior análisis gráfico.
for epoch in range(epochs):
    predictions = w * inputs
    errors = targets - predictions
    error_mean = np.mean(errors)
    
    print("\n" + "=" * 60)
    print(f"EPOCA {epoch + 1}")
    print("=" * 60)
    
    for i, t, p, e in zip(inputs, targets, predictions, errors):
        print(
            f"Entrada: {i:>4.1f} | "
            f"Target: {t:>5.1f} | "
            f"Prediccion: {p:>8.4f} | "
            f"Error: {e:>8.4f}"
        )
    
    print(f"Peso antes de actualizar: {w:.6f}")
    print(f"Error promedio: {error_mean:.6f}")
    
    w += learning_rate * error_mean
    
    print(f"Peso despues de actualizar: {w:.6f}")
    
    weight_history.append(w)
    error_history.append(error_mean)
    
    time.sleep(pause_time)
  #Se evalúa la capacidad predictiva del modelo optimizado utilizando datos de prueba independientes que no participaron en el entrenamiento.
  test_inputs = np.array([5, 6], dtype=float)
test_targets = np.array([10, 12], dtype=float)
test_predictions = w * test_inputs

print("\n" + "=" * 60)
print("PRUEBA FINAL")
print("=" * 60)

for i, t, p in zip(test_inputs, test_targets, test_predictions):
    print(
        f"Entrada: {i:>4.1f} | "
        f"Target: {t:>5.1f} | "
        f"Prediccion: {p:>8.4f}"
    )
#Se generan gráficos de línea para ilustrar de forma visual la convergencia del algoritmo
plt.figure()
plt.plot(range(1, epochs + 1), weight_history, marker="o")
plt.xlabel("Epoca")
plt.ylabel("Peso w")
plt.title("Evolucion del peso")
plt.grid()
plt.show()

plt.figure()
plt.plot(range(1, epochs + 1), error_history, marker="o")
plt.xlabel("Epoca")
plt.ylabel("Error promedio")
plt.title("Evolucion del error promedio")
plt.grid()
plt.show()
#El uso de NumPy incrementa la eficiencia de las operaciones numéricas y simplifica la sintaxis del código, 
#mientras que Matplotlib aporta una herramienta analítica indispensable para comprobar visualmente la correcta 
#convergencia del modelo hacia el valor óptimo.
