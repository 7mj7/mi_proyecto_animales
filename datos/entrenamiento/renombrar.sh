#!/bin/bash

# Entrar a cada carpeta y renombrar imágenes
for clase in gato perro otro; do
    count=1
    cd "$clase"  # Entrar a la carpeta de la clase
    for img in *.*; do
        nuevo_nombre="${clase}_$(printf "%03d" $count).${img##*.}"  # Formato: gato_001.jpg
        mv -- "$img" "$nuevo_nombre"
        ((count++))
    done
    cd ..  # Volver a la raíz
done