# metricas7.py

import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# --- Configuración ---
# Ruta de la carpeta de validación
ruta_validacion = os.path.join('..', 'datos', 'validacion')

# Ruta del modelo y etiquetas
model_path = os.path.join('modelo_tensorflow', 'keras_model.h5')
label_path = os.path.join('modelo_tensorflow', 'labels.txt')

# Tamaño de las imágenes esperadas por el modelo
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- Cargar el modelo ---
try:
    modelo = tf.keras.models.load_model(model_path)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# --- Cargar las etiquetas ---
try:
    with open(label_path, 'r') as f:
        clases = [line.strip().split(' ', 1)[1] for line in f]  # Asume formato "0 ClaseA", "1 ClaseB", etc.
    print(f"Clases cargadas: {clases}")
except Exception as e:
    print(f"Error al cargar las etiquetas desde {label_path}: {e}")
    exit()

# --- Preparar listas para almacenar etiquetas verdaderas y predicciones ---
y_true = []
y_pred = []

print(f"Procesando imágenes en la carpeta: {ruta_validacion}")

# --- Recorrer las carpetas de clases en el directorio de validación ---
if not os.path.isdir(ruta_validacion):
    print(f"Error: El directorio de validación '{ruta_validacion}' no existe.")
    exit()

# Obtener las clases reales de los nombres de las subcarpetas
actual_class_folders = sorted([d for d in os.listdir(ruta_validacion) if os.path.isdir(os.path.join(ruta_validacion, d))])

# Verificar si las clases de las carpetas coinciden con las etiquetas del modelo
if len(actual_class_folders) != len(clases):
    print(f"Advertencia: El número de clases en labels.txt ({len(clases)}) no coincide con el número de subcarpetas en {ruta_validacion} ({len(actual_class_folders)}).")
    exit()

# --- Procesar imágenes manualmente ---
for class_folder in actual_class_folders:
    true_label = class_folder
    class_path = os.path.join(ruta_validacion, class_folder)
    print(f"  Procesando clase: {true_label}")

    if not os.path.isdir(class_path):
        print(f"    Advertencia: '{class_path}' no es un directorio, saltando.")
        continue

    # Recorrer cada imagen dentro de la carpeta de la clase
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)

        # Ignorar archivos que no sean imágenes
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            print(f"    Saltando archivo no reconocido como imagen: {image_name}")
            continue

        try:
            # --- Cargar y preprocesar la imagen ---
            img = Image.open(image_path).convert('RGB')  # Asegurar que sea RGB
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # Redimensionar
            img_array = np.array(img).astype('float32') / 255.0  # Normalizar
            img_batch = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch

            # --- Realizar la predicción ---
            predictions = modelo.predict(img_batch, verbose=0)
            predicted_index = np.argmax(predictions[0])
            predicted_label = clases[predicted_index]

            # --- Almacenar resultados ---
            y_true.append(true_label)
            y_pred.append(predicted_label)

        except Exception as e:
            print(f"    Error procesando la imagen {image_name}: {e}")

# --- Calcular la Matriz de Confusión ---
print("\n--- Resultados ---")
if not y_true or not y_pred:
    print("No se procesaron imágenes o no se pudieron generar predicciones. No se puede calcular la matriz de confusión.")
else:
    cm = confusion_matrix(y_true, y_pred, labels=actual_class_folders)

    print("\nMatriz de Confusión:")
    print(cm)

    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, labels=actual_class_folders, zero_division=0))

