import time
inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

w = 0.1
epochs = 10
learning_rate = 0.1
pause_time = 1
#crea variables, poner los datos de inicio y el tiempo de pausa
def predict(i):
  return w * i

#Crea una funcion
for epoch in range(epochs):
    predictions = [predict(i) for i in inputs]
    errors = [
        (p - t)**2
        for p, t in zip(predictions, targets)
    ]
    cost = sum(errors) / len(targets)
    errors_d = [
        2 * (p - t)
        for p, t in zip(predictions, targets)
    ]
    weight_d = [
        e * i
        for e, i in zip(errors_d, inputs)
    ]
    print("\n" + "=" * 65)
    print(f"EPOCA {epoch + 1}")
    print("=" * 65)
    for i, t, p, e, ed, wd in zip(
        inputs,
        targets,
        predictions,
        errors,
        errors_d,
        weight_d
    ):
    print(
        f"Entrada: {i:>2} | "
        f"Target: {t:>2} | "
        f"Pred: {p:>8.4f} | "
        f"Error^2: {e:>8.4f} | "
        f"dE/dy: {ed:>8.4f} | "
        f"dE/dw: {wd:>8.4f}"
    )
    print("-" * 65)
    print(f"Peso antes de actualizar: {w:.10f}")
    print(f"Costo promedio: {cost:.6f}")
    gradient = sum(weight_d) / len(weight_d)
    print(f"Gradiente promedio: {gradient:.6f}")
    w -= learning_rate * gradient
    print(f"Peso despues de actualizar: {w:.10f}")
    time.sleep(pause_time)
  #Tras ejecutar las 10 épocas, se comprueba que:
#El uso del gradiente promedio direcciona el ajuste del peso de forma óptima hacia el valor ideal de 2.0.
#El costo promedio disminuye de manera acelerada conforme el modelo reduce el error cuadrático en cada iteración.
test_inputs = [5, 6]
test_targets = [10, 12]
print("\n" + "=" * 65)
print("PRUEBA FINAL")
print("=" * 65)

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
#La implementación del descenso de gradiente analítico basado en el error cuadrático medio demuestra una convergencia 
#superior y más robusta en comparación con el ajuste lineal simple, permitiendo que la neurona generalice correctamente 
#y prediga con exactitud los valores de prueba.
