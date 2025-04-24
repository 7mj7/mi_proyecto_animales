# metricas7_valdatagen.py

import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuración ---
ruta_validacion = os.path.join('..', 'datos', 'validacion')
model_path = os.path.join('modelo_tensorflow', 'keras_model.h5')
label_path = os.path.join('modelo_tensorflow', 'labels.txt')

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# --- Cargar modelo y etiquetas ---
try:
    modelo = tf.keras.models.load_model(model_path)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

try:
    with open(label_path, 'r') as f:
        class_names = [line.strip().split(' ', 1)[1] for line in f]
    print(f"Clases cargadas: {class_names}")
except Exception as e:
    print(f"Error al cargar las etiquetas: {e}")
    exit()

# --- Generador de datos con preprocesamiento ---
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Normalizar valores de píxeles entre 0 y 1
val_generator = val_datagen.flow_from_directory(
    ruta_validacion,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    classes=class_names  # Forzar el orden de las clases según labels.txt
)

# --- Predicciones ---
y_true = val_generator.classes  # Etiquetas verdaderas
y_pred = np.argmax(modelo.predict(val_generator, verbose=1), axis=1)  # Predicciones del modelo

# --- Resultados ---
print("\nMatriz de Confusión:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
