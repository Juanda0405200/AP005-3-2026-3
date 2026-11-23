from machine import Pin
import time

SW_X1 = Pin(18, Pin.IN, Pin.PULL_UP)
SW_X2 = Pin(19, Pin.IN, Pin.PULL_UP)
LED_X1 = Pin(25, Pin.OUT)
LED_X2 = Pin(26, Pin.OUT)
LED_Y = Pin(27, Pin.OUT)

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
targets = [0, 0, 0, 1]

w1, w2, b = 0, 0, 0
learning_rate = 1
max_epochs = 20
training_finished = False

def activation(z):
    return 1 if z >= 0 else 0

def predict(x1, x2):
    z = w1 * x1 + w2 * x2 + b
    return activation(z)

def show_visual_state(x1, x2, y):
    LED_X1.value(x1)
    LED_X2.value(x2)
    LED_Y.value(y)

def train_perceptron():
    global w1, w2, b, training_finished
    
    try:
        csv_file = open('datos_perceptron.csv', 'w')
        csv_file.write("Epoca,Errores,Z_00,Z_01,Z_10,Z_11\n")
    except Exception as e:
        print("Error al crear el archivo:", e)
        return

    for epoch in range(1, max_epochs + 1):
        errors_this_epoch = 0
        z_values = [0, 0, 0, 0]
        print(f"\n========== EPOCA {epoch} ==========")
        
        for i in range(4):
            x1, x2 = inputs[i]
            t = targets[i]
            
            z_values[i] = w1 * x1 + w2 * x2 + b
            y = predict(x1, x2)
            e = t - y
            
            show_visual_state(x1, x2, y)
            
            if e != 0:
                errors_this_epoch += 1
                w1 += learning_rate * e * x1
                w2 += learning_rate * e * x2
                b += learning_rate * e
                
            time.sleep(0.5)
            
        print(f"Errores durante la epoca: {errors_this_epoch}")
        
        csv_file.write(f"{epoch},{errors_this_epoch},{z_values[0]},{z_values[1]},{z_values[2]},{z_values[3]}\n")
        
        if errors_this_epoch == 0:
            print("APRENDIZAJE COMPLETADO")
            csv_file.close() 
            training_finished = True
            return
            
    print("Limite de epocas alcanzado")
    csv_file.close()
    training_finished = True

def manual_mode():
    x1 = SW_X1.value()
    x2 = SW_X2.value()
    y = predict(x1, x2)
    show_visual_state(x1, x2, y)
    time.sleep(0.1)

print("PERCEPTRON AND: ENTRENANDO...")
time.sleep(1)
train_perceptron()
print("DESPUES DEL ENTRENAMIENTO: MODO MANUAL")

while True:
    if training_finished:
        manual_mode()
