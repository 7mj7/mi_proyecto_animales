# metricas.py
# Este script genera una visualización de la matriz de confusión y un reporte de clasificación
# basado en las imágenes de validación y las predicciones del modelo.

import tensorflow as tf
import numpy as np
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración ---
ruta_modelo = os.path.join('modelo_tensorflow', 'keras_model.h5')  # Ruta al modelo .h5
ruta_etiquetas = os.path.join('modelo_tensorflow', 'labels.txt')   # Ruta al archivo labels.txt
directorio_validacion = os.path.join('..', 'datos', 'validacion')  # Ruta a las imágenes de validación
ALTURA_IMG = 224  # Altura esperada por el modelo
ANCHURA_IMG = 224  # Anchura esperada por el modelo

# --- Cargar el modelo ---
try:
    modelo = tf.keras.models.load_model(ruta_modelo)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# --- Cargar las etiquetas ---
try:
    with open(ruta_etiquetas, 'r') as archivo:
        nombres_clases = [linea.strip().split(' ', 1)[1] for linea in archivo]  # Asume formato "0 ClaseA", "1 ClaseB"
    print(f"Clases cargadas: {nombres_clases}")
except Exception as e:
    print(f"Error al cargar las etiquetas desde {ruta_etiquetas}: {e}")
    exit()

# --- Preparar listas para etiquetas verdaderas y predicciones ---
etiquetas_reales = []
etiquetas_predichas = []

print(f"Procesando imágenes en la carpeta: {directorio_validacion}")

# --- Validar la existencia del directorio de validación ---
if not os.path.isdir(directorio_validacion):
    print(f"Error: El directorio de validación '{directorio_validacion}' no existe.")
    exit()

# Obtener las clases reales desde las subcarpetas
carpetas_clases = sorted([d for d in os.listdir(directorio_validacion) if os.path.isdir(os.path.join(directorio_validacion, d))])

# Verificar si las clases coinciden en número con las etiquetas del modelo
if len(carpetas_clases) != len(nombres_clases):
    print(f"Advertencia: El número de clases en labels.txt ({len(nombres_clases)}) no coincide con las subcarpetas en {directorio_validacion} ({len(carpetas_clases)}).")

# --- Procesar imágenes por clase ---
for carpeta_clase in carpetas_clases:
    etiqueta_real = carpeta_clase
    ruta_clase = os.path.join(directorio_validacion, carpeta_clase)
    print(f"  Procesando clase: {etiqueta_real}")

    if not os.path.isdir(ruta_clase):
        print(f"    Advertencia: '{ruta_clase}' no es un directorio, saltando.")
        continue

    # Procesar cada imagen dentro de la carpeta
    for nombre_imagen in os.listdir(ruta_clase):
        ruta_imagen = os.path.join(ruta_clase, nombre_imagen)

        # Ignorar archivos que no sean imágenes
        if not nombre_imagen.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            print(f"    Saltando archivo no reconocido como imagen: {nombre_imagen}")
            continue

        try:
            # --- Cargar y preprocesar la imagen ---
            imagen = Image.open(ruta_imagen).convert('RGB')  # Convertir a RGB
            imagen = imagen.resize((ANCHURA_IMG, ALTURA_IMG))  # Redimensionar
            array_imagen = np.array(imagen).astype('float32') / 255.0  # Normalizar
            lote_imagen = np.expand_dims(array_imagen, axis=0)  # Añadir dimensión de batch

            # --- Realizar la predicción ---
            predicciones = modelo.predict(lote_imagen, verbose=0)
            indice_predicho = np.argmax(predicciones[0])
            etiqueta_predicha = nombres_clases[indice_predicho]

            # --- Almacenar resultados ---
            etiquetas_reales.append(etiqueta_real)
            etiquetas_predichas.append(etiqueta_predicha)

        except Exception as e:
            print(f"    Error procesando la imagen {nombre_imagen}: {e}")

# --- Calcular y mostrar resultados ---
print("\n--- Resultados ---")
if not etiquetas_reales or not etiquetas_predichas:
    print("No se procesaron imágenes o no se generaron predicciones. No se puede calcular la matriz de confusión.")
else:
    # Calcular matriz de confusión
    matriz_confusion = confusion_matrix(etiquetas_reales, etiquetas_predichas, labels=carpetas_clases)

    print("\nMatriz de Confusión:")
    print(matriz_confusion)

    #accuracy = np.trace(matriz_confusion) / np.sum(matriz_confusion)
    #print(f"Accuracy: {accuracy:.2f}")


    print("\nReporte de Clasificación:")
    print(classification_report(etiquetas_reales, etiquetas_predichas, labels=carpetas_clases, zero_division=0))

    # --- Visualizar la Matriz de Confusión ---
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=carpetas_clases, yticklabels=carpetas_clases)
        plt.xlabel('Predicción')
        plt.ylabel('Valor Verdadero')
        plt.title('Matriz de Confusión')
        plt.tight_layout()
        plt.savefig('matriz_confusion.png')  # Guardar la imagen
        print("\nMatriz de confusión guardada como 'matriz_confusion.png'")
    except Exception as e:
        print(f"\nNo se pudo generar la visualización de la matriz de confusión: {e}")
