#include <Arduino.h>

// Pines de entrada y visualizacion
const int SW_X1 = 18;
const int SW_X2 = 19;
const int LED_X1 = 25;
const int LED_X2 = 26;
const int LED_Y = 27;

// Tabla de entrenamiento or
const int inputs[4][2] = {
{0, 0}, {0, 1}, {1, 0}, {1, 1}
};
const int targets[4] = {0, 0, 0, 1};

 // Parametros iniciales
int w1 = 0;
int w2 = 0;
int b = 0;
const int learning_rate = 1;
const int max_epochs = 20;

bool trainingFinished = false;
int lastX1 = -1, lastX2 = -1, lastY = -1;

// Escalon: z >= 0 produce la clase 1
int activation(int z) {
 return (z >= 0) ? 1 : 0;
}

int predict(int x1, int x2) {
 int z = w1*x1 + w2*x2 + b;
 return activation(z);
}

void showVisualState(int x1, int x2, int y) {
 digitalWrite(LED_X1, x1);
 digitalWrite(LED_X2, x2);
 digitalWrite(LED_Y, y);
}

void blinkOutputLed(int times, int delayMs) {
 for (int i = 0; i < times; i++) {
 digitalWrite(LED_Y, HIGH);
 delay(delayMs);
 digitalWrite(LED_Y, LOW);
 delay(delayMs);
 }
}

// Evalua los cuatro ejemplos sin actualizar pesos
int evaluate(bool showLeds, int waitMs) {
 int errors = 0;
 Serial.println("x1 x2 | Target | Prediccion | Error");
 
  for (int i = 0; i < 4; i++) {
   int x1 = inputs[i][0];
   int x2 = inputs[i][1];
   int t = targets[i];
   int y = predict(x1, x2);
   int e = t - y;

   if (e != 0) errors++;
   Serial.printf(" %d %d | %d | %d | %2d\n",
   x1, x2, t, y, e);

   if (showLeds) {
     showVisualState(x1, x2, y);
     delay(waitMs);
    }
  }
  Serial.printf("Errores totales: %d de 4\n", errors);
  return errors;
}

void trainPerceptron() {
 for (int epoch = 1; epoch <= max_epochs; epoch++) {
   int errorsThisEpoch = 0;
   Serial.println();
   Serial.printf("========== EPOCA %d ==========\n", epoch);

   for (int i = 0; i < 4; i++) {
     int x1 = inputs[i][0];
     int x2 = inputs[i][1];
     int t = targets[i];
     int y = predict(x1, x2);
     int e = t - y;

     showVisualState(x1, x2, y);
     Serial.printf("Patron %d -> x=( %d, %d), t= %d, y= %d, e= %d\n",
     i + 1, x1, x2, t, y, e);

     if (e != 0) {
       errorsThisEpoch++;
       Serial.println("Hay error: actualizar parametros");
       blinkOutputLed(2, 120);
       w1 += learning_rate * e * x1;
       w2 += learning_rate * e * x2;
       b += learning_rate * e;
      }

      Serial.printf("w1= %d, w2= %d, b= %d\n", w1, w2, b);
      delay(1400); // Visualizacion de cada patron
    }

   Serial.printf("Errores durante la epoca: %d\n", errorsThisEpoch);
   int remainingErrors = evaluate(true, 1000);

   if (remainingErrors == 0) {
     Serial.println("APRENDIZAJE COMPLETADO");
     Serial.printf("w1= %d, w2= %d, b= %d\n", w1, w2, b);
     trainingFinished = true;
     return;
    }
    delay(2400); // Pausa entre epocas
  }

 Serial.println("Limite de epocas alcanzado");
 trainingFinished = true;
}
void manualMode() {
  // Con pull-up: abierto = 1, cerrado = 0
 int x1 = digitalRead(SW_X1);
 int x2 = digitalRead(SW_X2);
 int y = predict(x1, x2);

 showVisualState(x1, x2, y);
 if (x1 != lastX1 || x2 != lastX2 || y != lastY) {
   Serial.printf("MODO MANUAL: x1= %d, x2= %d, y= %d\n", x1, x2, y);
   lastX1 = x1;
   lastX2 = x2;
   lastY = y;
  }
  delay(100);
}

void setup() {
 Serial.begin(115200);
 delay(1000);
 pinMode(SW_X1, INPUT);
 pinMode(SW_X2, INPUT);
 pinMode(LED_X1, OUTPUT);
 pinMode(LED_X2, OUTPUT);
 pinMode(LED_Y, OUTPUT);
 showVisualState(0, 0, 0);

 Serial.println("PERCEPTRON or: ANTES DEL ENTRENAMIENTO");
 evaluate(true, 1000);
 delay(2000);
 trainPerceptron();
 Serial.println("DESPUES DEL ENTRENAMIENTO: MODO MANUAL");
}

void loop() {
 if (trainingFinished) manualMode();
}

