"""
Aplicación Streamlit para el despliegue de modelos CNN
======================================================
Metodología: CRISP-ML(Q)
Modelos:     MNIST (Dígitos Manuscritos) y Fashion MNIST (Prendas de Vestir)
Framework:   Streamlit + TensorFlow/Keras + Pandas

Autor:       Eliab Ezziel Zamalloa Cayo
Versión:     1.0.0
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
# Importación en la cabecera para asegurar la estabilidad en Streamlit Cloud
from tensorflow.keras.models import load_model 

# ---------------------------------------------------------------------------
# Configuración de página (DEBE ser la primera llamada a Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clasificador CNN — CRISP-ML(Q)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Diccionarios de etiquetas
# ---------------------------------------------------------------------------
ETIQUETAS_MNIST = [str(i) for i in range(10)]

ETIQUETAS_FASHION_MNIST = [
    "Camiseta",   # 0
    "Pantalón",   # 1
    "Pulóver",    # 2
    "Vestido",    # 3
    "Abrigo",     # 4
    "Sandalia",   # 5
    "Camisa",     # 6
    "Zapatilla",  # 7
    "Bolso",      # 8
    "Botín",      # 9
]

RUTAS_MODELOS = {
    "MNIST (Dígitos)": "modelo_mnist.h5",
    "Fashion MNIST (Ropa)": "modelo_fashion.h5",
}


# ---------------------------------------------------------------------------
# Carga de modelos (cacheada en memoria)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo(ruta_modelo: str):
    """
    Carga un modelo Keras desde un archivo .h5 utilizando cache
    de recursos de Streamlit para evitar recargas en cada interacción.
    """
    try:
        modelo = load_model(ruta_modelo)
        return modelo
    except Exception as e:
        st.error(f"No se pudo cargar el modelo desde '{ruta_modelo}'.\n{str(e)}")
        return None


# ---------------------------------------------------------------------------
# Pipeline de preprocesamiento de imágenes
# ---------------------------------------------------------------------------
def preprocesar_imagen(imagen_subida):
    """
    Convierte la imagen subida por el usuario al formato exacto que espera
    el modelo CNN: (1, 28, 28, 1), en escala de grises, normalizada a [0, 1].
    """
    # Abrir imagen y convertir a escala de grises
    img = Image.open(imagen_subida).convert("L")  # 'L' = Luminance (grayscale)

    # Redimensionar a 28x28 píxeles (tamaño de entrenamiento)
    img = img.resize((28, 28))

    # Convertir a arreglo NumPy
    img_array = np.array(img, dtype=np.float32)

    # Normalizar: dividir entre 255.0 para escalar a [0, 1]
    img_array = img_array / 255.0

    # Ajustar dimensiones: (alto, ancho) -> (1, alto, ancho, 1)
    img_array = np.expand_dims(img_array, axis=0)   # añade batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # añade channel dimension

    return img_array


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def predecir(modelo, imagen_procesada, etiquetas):
    """
    Ejecuta la inferencia sobre la imagen preprocesada y retorna
    la clase predicha, la confianza y el vector completo de probabilidades.
    """
    try:
        probabilidades = modelo.predict(imagen_procesada, verbose=0)[0]
        clase_idx = int(np.argmax(probabilidades))
        confianza = float(probabilidades[clase_idx])
        clase_nombre = etiquetas[clase_idx]
        return clase_nombre, confianza, probabilidades
    except Exception as e:
        st.error(f"Error durante la predicción:\n{str(e)}")
        return None, None, None


# ---------------------------------------------------------------------------
# Componentes de la interfaz de usuario
# ---------------------------------------------------------------------------
def sidebar_controles():
    """Construye la barra lateral con controles, información y créditos."""
    with st.sidebar:
        # Actualizado con el logo oficial de TensorFlow
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg",
            width=60,
        )
        st.title("🧠 Clasificador CNN")
        st.markdown("---")

        # --- Selección del modelo ---
        modelo_seleccionado = st.selectbox(
            "Selecciona el modelo:",
            options=list(RUTAS_MODELOS.keys()),
            index=0,
        )

        # --- Sección de ayuda ---
        with st.expander("ℹ️ ¿Cómo usar?"):
            st.markdown(
                """
                1. Selecciona un modelo de la lista.
                2. Sube una imagen **.jpg** o **.png**.
                3. El modelo mostrará la clase predicha y las
                   probabilidades de todas las categorías.
                """
            )

        st.markdown("---")

        # --- Información del proyecto (CRISP-ML(Q)) ---
        st.subheader("📋 Metodología CRISP-ML(Q)")
        st.markdown(
            """
            | Fase | Descripción |
            |------|-------------|
            | **Business & Data Understanding** | Definición del problema de clasificación de imágenes. |
            | **Data Preparation** | Normalización, redimensión a 28×28 y codificación de canales. |
            | **Modeling** | Arquitectura CNN con capas Convolutional, Pooling y Dense. |
            | **Evaluation** | Validación con accuracy, pérdida y matriz de confusión. |
            | **Deployment** | Despliegue en Streamlit Cloud con interfaz interactiva. |
            | **Monitoring** | Seguimiento de predicciones y rendimiento del modelo. |
            """
        )

        st.markdown("---")

        # --- Créditos del desarrollador ---
        st.markdown(
            """
            **👨‍💻 Desarrollado por** *Estudiante de Ingeniería de Sistemas e Informática* *Universidad Continental*

            ---
            *Proyecto de producción — 2026*
            """
        )

        return modelo_seleccionado


def columna_imagen(imagen_subida):
    """Renderiza la imagen subida por el usuario en la columna izquierda."""
    st.subheader("📷 Imagen cargada")
    img = Image.open(imagen_subida).convert("L")
    st.image(
        img,
        caption="Imagen en escala de grises procesada (28×28)",
        use_container_width=False,
        width=200,
    )


def columna_resultados(clase_nombre, confianza, probabilidades, etiquetas):
    """Renderiza la predicción y el gráfico de confianza en la columna derecha."""
    st.subheader("📊 Resultado de la predicción")

    if clase_nombre is None:
        st.warning("No se pudo realizar la predicción. Verifica la imagen e inténtalo de nuevo.")
        return

    # --- Métrica principal ---
    st.metric(
        label="Clase predicha",
        value=clase_nombre,
        delta=f"{confianza:.2%} de certeza",
    )

    st.markdown("---")

    # --- Gráfico de barras horizontal con distribución Softmax ---
    st.markdown("**Distribución de probabilidad (Softmax) por clase**")

    df_prob = pd.DataFrame(
        {"Clase": etiquetas, "Probabilidad": probabilidades}
    ).sort_values("Probabilidad", ascending=True)

    st.bar_chart(df_prob, x="Clase", y="Probabilidad", horizontal=True)


# ---------------------------------------------------------------------------
# Cuerpo principal de la aplicación
# ---------------------------------------------------------------------------
def main():
    """Función principal que orquesta la interfaz y la lógica de predicción."""

    modelo_seleccionado = sidebar_controles()

    st.title("🧠 Clasificador de Imágenes con CNN")
    st.markdown(
        "Sube una imagen en escala de grises o a color (se convertirá automáticamente) "
        "para que el modelo de **Red Neuronal Convolucional** seleccionado realice "
        "una predicción con distribución completa de probabilidades."
    )

    st.markdown("---")

    # --- Carga del modelo ---
    ruta = RUTAS_MODELOS[modelo_seleccionado]
    modelo = cargar_modelo(ruta)

    if modelo is None:
        st.error(
            f"❌ El archivo del modelo **{ruta}** no se encuentra o no se pudo cargar. "
            "Asegúrate de que exista en el directorio raíz del repositorio."
        )
        st.stop()

    # --- Etiquetas según modelo ---
    if modelo_seleccionado == "MNIST (Dígitos)":
        etiquetas = ETIQUETAS_MNIST
    else:
        etiquetas = ETIQUETAS_FASHION_MNIST

    # --- Subida de imagen ---
    imagen_subida = st.file_uploader(
        "Selecciona una imagen (JPG / PNG):",
        type=["jpg", "jpeg", "png"],
    )

    if imagen_subida is None:
        st.info("⬆️ Sube una imagen para comenzar la clasificación.")
        st.stop()

    # --- Pipeline de preprocesamiento + predicción ---
    imagen_procesada = preprocesar_imagen(imagen_subida)
    clase_nombre, confianza, probabilidades = predecir(modelo, imagen_procesada, etiquetas)

    # --- Layout en dos columnas ---
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        # Rebobinar el puntero del archivo para que PIL pueda leerlo de nuevo sin errores
        imagen_subida.seek(0)
        columna_imagen(imagen_subida)

    with col_right:
        columna_resultados(clase_nombre, confianza, probabilidades, etiquetas)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()