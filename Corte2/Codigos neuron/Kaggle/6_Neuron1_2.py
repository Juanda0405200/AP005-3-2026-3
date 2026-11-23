#Este cuaderno consolida la arquitectura de una neurona artificial completa mediante regresión lineal, incorporando el sesgo (bias)** junto con optimizaciones avanzadas:
import time
import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([1, 2, 3, 4], dtype=float)
targets = np.array([4, 6, 8, 10], dtype=float)

w = 0.1
b = 0.3

epochs = 300
learning_rate = 0.1
pause_time = 0.1

weight_history = []
bias_history = []
cost_history = []
#En cada época, el algoritmo ejecuta etapas analíticas
for epoch in range(epochs):
    predictions = w * inputs + b
    errors = (predictions - targets)**2
    cost = np.mean(errors)
    
    errors_d = 2 * (predictions - targets)
    weight_d = errors_d * inputs
    bias_d = errors_d
    
    gradient_w = np.mean(weight_d)
    gradient_b = np.mean(bias_d)
    
    print("\n" + "=" * 70)
    print(f"EPOCA {epoch + 1}")
    print("=" * 70)
    
    for i, t, p, e, wd, bd in zip(inputs, targets, predictions, errors, weight_d, bias_d):
        print(
            f"Entrada: {i:>4.1f} | "
            f"Target: {t:>5.1f} | "
            f"Pred: {p:>8.4f} | "
            f"Error^2: {e:>8.4f} | "
            f"dE/dw: {wd:>9.4f} | "
            f"dE/db: {bd:>9.4f}"
        )
    
    print(f"Costo promedio: {cost:.6f}")
    print(f"Gradiente dJ/dw: {gradient_w:.6f}")
    print(f"Gradiente dJ/db: {gradient_b:.6f}")
    print(f"Peso antes: {w:.6f}")
    print(f"Bias antes: {b:.6f}")
    
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b
    
    print(f"Peso despues: {w:.6f}")
    print(f"Bias despues: {b:.6f}")
    
    weight_history.append(w)
    bias_history.append(b)
    cost_history.append(cost)
    
    time.sleep(pause_time)
  #Se evalúa la precisión del modelo optimizado utilizando entradas de prueba independientes que no participaron en el entrenamiento.
  test_inputs = np.array([5, 6], dtype=float)
test_predictions = w * test_inputs + b

print("\n" + "=" * 70)
print("PRUEBA FINAL")
print("=" * 70)

for i, p in zip(test_inputs, test_predictions):
    print(
        f"Entrada: {i:>4.1f} | "
        f"Prediccion: {p:>8.4f}"
    )
#Se generan tres representaciones gráficas independientes para un análisis exhaustivo
plt.figure()
plt.plot(range(1, epochs + 1), cost_history)
plt.xlabel("Epoca")
plt.ylabel("Costo J")
plt.title("Evolucion del costo")
plt.grid()
plt.show()

plt.figure()
plt.plot(range(1, epochs + 1), weight_history, label="Peso w")
plt.plot(range(1, epochs + 1), bias_history, label="Sesgo b")
plt.xlabel("Epoca")
plt.ylabel("Valor")
plt.title("Evolucion de los parametros")
plt.legend()
plt.grid()
plt.show()

x_plot = np.linspace(0, 7, 100)
y_plot = w * x_plot + b

plt.figure()
plt.scatter(inputs, targets, label="Datos de entrenamiento")
plt.plot(x_plot, y_plot, label="Modelo aprendido")
plt.xlabel("Entrada")
plt.ylabel("Salida")
plt.title("Datos y modelo aprendido")
plt.legend()
plt.grid()
plt.show()
#La implementación integral con NumPy y Matplotlib permite modelar relaciones lineales con 
#sesgo de manera altamente eficiente y documentar visualmente el comportamiento de convergencia, 
#cumpliendo con los estándares de un reporte técnico formal.
