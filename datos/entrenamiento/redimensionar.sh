#!/bin/bash

# Hay que instalar ImageMagick si no esta instalado
# sudo apt-get install imagemagick

# Redimensionar imágenes en todas las carpetas
for clase in gato perro otro; do
    cd "$clase"
    mogrify -resize 224x224! *.jpg *.png  # El '!' fuerza el tamaño exacto
    cd ..
done