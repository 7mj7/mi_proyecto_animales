
# Actividad 3 - Machine Learning

Este repositorio contiene los scripts necesarios para realizar la validación de modelos de clasificación entrenados con Teachable Machine. Los scripts permiten generar métricas clave como la matriz de confusión, precisión, recall, F1-score y exactitud.

## Requisitos previos

Tener instalado Python 3.10 y las siguientes herramientas:

- Python: Se recomienda usar pyenv para gestionar versiones de Python.
- Entorno virtual: Utiliza venv o virtualenv para aislar las dependencias del proyecto.
- Dependencias: Las librerías necesarias están listadas en el archivo requirements.txt.

## Preparación del entorno

Clonar el repositorio

```bash
git clone https://github.com/7mj7/mi_proyecto_animales.git
cd mi_proyecto_animales
```

Crear un entorno virtual

```bash
python -m venv mi_entorno_IA
source mi_entorno_IA/bin/activate  # En Linux/Mac
# En Windows:
# mi_entorno_IA\Scripts\activate
```

Instalar las dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

# Configuración del entorno (opcional)

Si necesitas instalar una versión específica de Python (por ejemplo, 3.10), puedes usar pyenv:

1. Instalar pyenv:

```bash
curl https://pyenv.run | bash
```

2. Añadir las siguientes líneas a tu archivo de configuración del shell

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

3. Instalar Python 3.10 con pyenv

```bash
pyenv install 3.10.12
pyenv virtualenv 3.10.12 mi_entorno_IA
pyenv activate mi_entorno_IA
```

# Ejecución de los scripts

El script *generar_metricas.py* calcula la matriz de consusión y las métricas modelo entrenado.

```bash
python generar_metricas.py 
```