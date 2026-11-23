import time
import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([1, 2, 3, 4], dtype=float)
targets = np.array([2, 4, 6, 8], dtype=float)

w = 0.1
epochs = 10
learning_rate = 0.1
pause_time = 0.2

weight_history = []
cost_history = []
#En cada época de entrenamiento, el algoritmo ejecuta de forma vectorizada
for epoch in range(epochs):
    predictions = w * inputs
    errors = (predictions - targets)**2
    cost = np.mean(errors)
    
    errors_d = 2 * (predictions - targets)
    weight_d = errors_d * inputs
    gradient = np.mean(weight_d)
    
    print("\n" + "=" * 65)
    print(f"EPOCA {epoch + 1}")
    print("=" * 65)
    
    for i, t, p, e, ed, wd in zip(inputs, targets, predictions, errors, errors_d, weight_d):
        print(
            f"Entrada: {i:>4.1f} | "
            f"Target: {t:>5.1f} | "
            f"Pred: {p:>8.4f} | "
            f"Error^2: {e:>8.4f} | "
            f"dE/dy: {ed:>8.4f} | "
            f"dE/dw: {wd:>8.4f}"
        )
    
    print(f"Costo promedio: {cost:.6f}")
    print(f"Gradiente dJ/dw: {gradient:.6f}")
    print(f"Peso antes: {w:.6f}")
    
    w -= learning_rate * gradient
    
    print(f"Peso despues: {w:.6f}")
    
    weight_history.append(w)
    cost_history.append(cost)
    
    time.sleep(pause_time)
#Se evalúa el rendimiento de la neurona utilizando un conjunto de datos de prueba independientes para verificar su capacidad predictiva.
test_inputs = np.array([5, 6], dtype=float)
test_targets = np.array([10, 12], dtype=float)
test_predictions = w * test_inputs

print("\n" + "=" * 65)
print("PRUEBA FINAL")
print("=" * 65)

for i, t, p in zip(test_inputs, test_targets, test_predictions):
    print(
        f"Entrada: {i:>4.1f} | "
        f"Target: {t:>5.1f} | "
        f"Prediccion: {p:>8.4f}"
    )
#Evolución del costo: Muestra cómo el costo J disminuye exponencialmente hacia cero conforme avanza la optimización.
#Evolución del peso: Ilustra la trayectoria de convergencia del parámetro 
#hacia su valor ideal de 2.0.
plt.figure()
plt.plot(range(1, epochs + 1), cost_history, marker="o")
plt.xlabel("Epoca")
plt.ylabel("Costo J")
plt.title("Evolucion del costo")
plt.grid()
plt.show()

plt.figure()
plt.plot(range(1, epochs + 1), weight_history, marker="o")
plt.xlabel("Epoca")
plt.ylabel("Peso w")
plt.title("Evolucion del peso")
plt.grid()
plt.show()
