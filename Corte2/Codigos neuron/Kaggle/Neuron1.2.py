import time

inputs = [1, 2, 3, 4]
targets = [4, 6, 8, 10]

w = 0.1
b = 0.3
epochs = 300
learning_rate = 0.1
pause_time = 1

def predict(i):
    return w * i + b
#se definen algunas variables y valores iniciales para el programa
for epoch in range(epochs):
    predictions = [
        predict(i)
        for i in inputs
    ]
    
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
    
    bias_d = [
        e
        for e in errors_d
    ]
    
    print("\n" + "=" * 70)
    print(f"EPOCA {epoch + 1}")
    print("=" * 70)
    
    for i, t, p, e, wd, bd in zip(inputs, targets, predictions, errors, weight_d, bias_d):
        print(
            f"Entrada: {i:>2} | "
            f"Target: {t:>2} | "
            f"Pred: {p:>8.4f} | "
            f"Error^2: {e:>8.4f} | "
            f"dE/dw: {wd:>9.4f} | "
            f"dE/db: {bd:>9.4f}"
        )
    
    gradient_w = sum(weight_d) / len(weight_d)
    gradient_b = sum(bias_d) / len(bias_d)
    
    print("-" * 70)
    print(f"Peso antes de actualizar: {w:.10f}")
    print(f"Bias antes de actualizar: {b:.10f}")
    print(f"Costo promedio: {cost:.6f}")
    print(f"Gradiente de w: {gradient_w:.6f}")
    print(f"Gradiente de b: {gradient_b:.6f}")
    
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b
    
    print(f"Peso despues de actualizar: {w:.10f}")
    print(f"Bias despues de actualizar: {b:.10f}")
    
    time.sleep(pause_time)
#Tras completar las 300 épocas: Se observa cómo tanto el peso w como el sesgo b convergen de manera estable 
#hacia sus valores ideales. El costo promedio se reduce drásticamente, comprobando que la neurona logró modelar
#correctamente la función de desplazamiento lineal.
test_inputs = [5, 6]

print("\n" + "=" * 70)
print("PRUEBA FINAL")
print("=" * 70)

test_predictions = [
    predict(i)
    for i in test_inputs
]

for i, p in zip(test_inputs, test_predictions):
    print(
        f"Entrada: {i:>2} | "
        f"Prediccion: {p:>8.4f}"
    )
#La incorporación del sesgo demuestra ser indispensable en arquitecturas neuronales cuando los datos no 
#cruzan el origen, permitiendo que el algoritmo ajuste con exactitud la ordenada al origen y mantenga 
#una buena capacidad de generalización en las pruebas finales.
