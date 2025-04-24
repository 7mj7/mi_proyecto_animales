# metricas4.py
# Este script proporciona una visualización de la matriz de confusión y un reporte de clasificación
# basado en las imágenes de validación y las predicciones del modelo.

import tensorflow as tf
import numpy as np
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuración ---
model_path = os.path.join('modelo_tensorflow', 'keras_model.h5') # Ruta a tu archivo .h5 exportado de Teachable Machine
label_path = os.path.join('modelo_tensorflow', 'labels.txt')   # Ruta a tu archivo labels.txt
validation_dir = os.path.join('..', 'datos', 'validacion') # Ruta a la carpeta con las imágenes de validación
IMG_HEIGHT = 224             # Altura de imagen esperada por el modelo (ajusta si es diferente)
IMG_WIDTH = 224              # Ancho de imagen esperado por el modelo (ajusta si es diferente)

# --- Cargar el modelo Keras ---
# Keras es una API de alto nivel para construir y entrenar modelos de deep learning [[6]].
# tf.keras es la implementación de TensorFlow de esta API [[6]].
try:
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# --- Cargar las etiquetas (nombres de las clases) ---
try:
    with open(label_path, 'r') as f:
        class_names = [line.strip().split(' ', 1)[1] for line in f] # Asume formato "0 ClaseA", "1 ClaseB", etc.
    print(f"Clases cargadas: {class_names}")
except Exception as e:
    print(f"Error al cargar las etiquetas desde {label_path}: {e}")
    print("Asegúrate de que el archivo labels.txt exista y tenga el formato correcto.")
    exit()

# --- Preparar listas para almacenar etiquetas verdaderas y predicciones ---
y_true = []
y_pred = []

print(f"Procesando imágenes en la carpeta: {validation_dir}")

# --- Recorrer las carpetas de clases en el directorio de validación ---
# La estructura de carpetas define las etiquetas verdaderas.
if not os.path.isdir(validation_dir):
    print(f"Error: El directorio de validación '{validation_dir}' no existe.")
    exit()

# Obtener las clases reales de los nombres de las subcarpetas
actual_class_folders = sorted([d for d in os.listdir(validation_dir) if os.path.isdir(os.path.join(validation_dir, d))])

# Verificar si las clases de las carpetas coinciden (al menos en número) con las etiquetas del modelo
if len(actual_class_folders) != len(class_names):
     print(f"Advertencia: El número de clases en labels.txt ({len(class_names)}) no coincide con el número de subcarpetas en {validation_dir} ({len(actual_class_folders)}).")
     print("Asegúrate de que cada clase tenga una subcarpeta correspondiente en 'validacion'.")
     # Podrías decidir salir aquí o continuar con las clases que coincidan.
     # Por simplicidad, continuaremos, pero la matriz podría ser incorrecta si faltan clases.

for class_folder in actual_class_folders:
    true_label = class_folder
    class_path = os.path.join(validation_dir, class_folder)
    print(f"  Procesando clase: {true_label}")

    if not os.path.isdir(class_path):
        print(f"    Advertencia: '{class_path}' no es un directorio, saltando.")
        continue

    # --- Recorrer cada imagen dentro de la carpeta de la clase ---
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)

        # Ignorar archivos que no sean imágenes (puedes añadir más extensiones si es necesario)
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            print(f"    Saltando archivo no reconocido como imagen: {image_name}")
            continue

        try:
            # --- Cargar y preprocesar la imagen ---
            img = Image.open(image_path).convert('RGB') # Asegurar que sea RGB
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))

            # Convertir imagen a array NumPy [[10]]
            img_array = np.array(img)

            # Normalizar la imagen (Teachable Machine usualmente normaliza a [-1, 1] o [0, 1])
            # Verifica cómo entrenaste tu modelo. Aquí asumimos [0, 1].
            img_array = img_array.astype('float32') / 255.0

            # Añadir una dimensión de batch (el modelo espera un lote de imágenes)
            # Shape cambia de (height, width, channels) a (1, height, width, channels)
            img_batch = np.expand_dims(img_array, axis=0)

            # --- Realizar la predicción ---
            # Keras permite hacer predicciones con el método predict() [[6]].
            predictions = model.predict(img_batch, verbose=0) # verbose=0 para no imprimir progreso por imagen

            # Obtener el índice de la clase con la mayor probabilidad
            predicted_index = np.argmax(predictions[0])

            # Obtener el nombre de la clase predicha usando el índice y las etiquetas cargadas
            predicted_label = class_names[predicted_index]

            # --- Almacenar resultados ---
            y_true.append(true_label)
            y_pred.append(predicted_label)

        except Exception as e:
            print(f"    Error procesando la imagen {image_name}: {e}")

# --- Calcular la Matriz de Confusión ---
# La matriz de confusión ayuda a entender el rendimiento del modelo, mostrando
# verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos [[3]].
print("\n--- Resultados ---")
if not y_true or not y_pred:
    print("No se procesaron imágenes o no se pudieron generar predicciones. No se puede calcular la matriz de confusión.")
else:
    # Usamos las carpetas encontradas como etiquetas para asegurar el orden correcto
    # Si alguna carpeta no estaba en labels.txt (o viceversa), esto puede dar error o resultados inesperados.
    # Es crucial que las carpetas en 'validacion' coincidan con las clases en 'labels.txt'.
    # Usamos 'actual_class_folders' como 'labels' para asegurar el orden de la matriz.
    cm = confusion_matrix(y_true, y_pred, labels=actual_class_folders)

    print("\nMatriz de Confusión:")
    print(cm)

    print("\nReporte de Clasificación:")
    # Asegúrate de que las etiquetas usadas aquí coincidan con las de la matriz
    print(classification_report(y_true, y_pred, labels=actual_class_folders, zero_division=0))

    # --- Visualizar la Matriz de Confusión (Opcional) ---
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actual_class_folders, yticklabels=actual_class_folders)
        plt.xlabel('Predicción')
        plt.ylabel('Valor Verdadero')
        plt.title('Matriz de Confusión')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png') # Guarda la imagen
        print("\nMatriz de confusión guardada como 'confusion_matrix.png'")
        # plt.show() # Descomenta si quieres mostrarla en una ventana
    except Exception as e:
        print(f"\nNo se pudo generar la visualización de la matriz de confusión: {e}")
        print("Asegúrate de tener instalado matplotlib y seaborn: pip install matplotlib seaborn")

