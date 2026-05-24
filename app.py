"""
Aplicación Streamlit para el despliegue de modelos CNN
======================================================
Metodología: CRISP-ML(Q)
Modelos:     MNIST (Dígitos Manuscritos) y Fashion MNIST (Prendas de Vestir)
Framework:   Streamlit + TensorFlow/Keras + Pandas + Altair

Autor:       Eliab Ezziel Zamalloa Cayo
Versión:     1.1.0 (Con visualización de capas convolucionales)
"""

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import io
# Importación en la cabecera para asegurar la estabilidad en Streamlit Cloud
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D

# ---------------------------------------------------------------------------
# Configuración de página
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
    "Camiseta", "Pantalón", "Pulóver", "Vestido", "Abrigo", 
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"
]

RUTAS_MODELOS = {
    "MNIST (Dígitos)": "modelo_mnist.h5",
    "Fashion MNIST (Ropa)": "modelo_fashion.h5",
}

# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo(ruta_modelo: str):
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
    img = Image.open(imagen_subida).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def predecir(modelo, imagen_procesada, etiquetas):
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
# Visualización de Convoluciones
# ---------------------------------------------------------------------------
def visualizar_convoluciones(modelo, imagen_procesada):
    """Extrae y visualiza los mapas de características de la primera capa Conv2D"""
    st.markdown("---")
    st.subheader("👁️ Análisis Interno: ¿Qué está viendo la CNN?")
    st.write("Los siguientes **Mapas de Características** muestran la salida matemática de la primera capa convolucional de la Red Neuronal, revelando los bordes y texturas que la IA considera importantes.")

    # Buscar la primera capa Conv2D en el modelo
    capa_conv = None
    for layer in modelo.layers:
        if isinstance(layer, Conv2D):
            capa_conv = layer
            break

    if capa_conv is not None:
        # Crear un sub-modelo para extraer las activaciones hasta esa capa
        modelo_intermedio = Model(inputs=modelo.inputs, outputs=capa_conv.output)
        mapas = modelo_intermedio.predict(imagen_procesada, verbose=0)

        # Configurar la visualización de los primeros 6 filtros
        num_filtros = mapas.shape[-1]
        filtros_a_mostrar = min(num_filtros, 6)
        
        cols = st.columns(filtros_a_mostrar)
        for i in range(filtros_a_mostrar):
            mapa_img = mapas[0, :, :, i]
            
            # Normalizar la matriz a valores de píxeles [0, 255]
            mapa_img -= mapa_img.min()
            if mapa_img.max() > 0:
                mapa_img /= mapa_img.max()
            mapa_img *= 255
            mapa_img = np.clip(mapa_img, 0, 255).astype(np.uint8)

            # Mostrar la convolución en Streamlit
            with cols[i]:
                st.image(Image.fromarray(mapa_img).resize((100, 100)), caption=f"Filtro Conv {i+1}")
    else:
        st.info("El modelo actual no cuenta con capas Conv2D estándar reconocibles.")

# ---------------------------------------------------------------------------
# Componentes de UI
# ---------------------------------------------------------------------------
def sidebar_controles():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg", width=60)
        st.title("🧠 Clasificador CNN")
        st.markdown("---")
        modelo_seleccionado = st.selectbox("Selecciona el modelo:", options=list(RUTAS_MODELOS.keys()), index=0)
        
        with st.expander("ℹ️ ¿Cómo usar?"):
            st.markdown("""
                1. Selecciona un modelo.
                2. Sube una imagen **.jpg** o **.png**.
                3. Analiza las probabilidades y los filtros convolucionales en tiempo real.
                """)
        st.markdown("---")
        st.subheader("📋 Metodología CRISP-ML(Q)")
        st.markdown("""
            | Fase | Descripción |
            |------|-------------|
            | **Business & Data** | Definición del problema. |
            | **Data Prep** | Normalización 28×28. |
            | **Modeling** | CNN con capas Conv2D. |
            | **Evaluation** | Mapas de características. |
            | **Deployment** | Streamlit Cloud. |
            """)
        st.markdown("---")
        st.markdown("**👨‍💻 Desarrollado por MothCode**\n\n*Ingeniería de Sistemas e Informática - Universidad Continental*")
        return modelo_seleccionado

def columna_imagen(imagen_subida):
    st.subheader("📷 Imagen de entrada")
    img = Image.open(imagen_subida)
    st.image(img, caption="Imagen original lista para analizar", width=250)

def columna_resultados(clase_nombre, confianza, probabilidades, etiquetas):
    st.subheader("📊 Resultados de la IA")

    if clase_nombre is None:
        st.warning("Error en predicción.")
        return

    st.metric(label="Predicción Principal", value=clase_nombre, delta=f"{confianza:.2%} de certeza")
    st.markdown("---")

    # Preparar DataFrame con todas las probabilidades
    df_prob = pd.DataFrame({
        "Categoría": etiquetas,
        "Probabilidad": probabilidades,
        "Porcentaje Exacto": [f"{p*100:.2f}%" for p in probabilidades]
    }).sort_values("Probabilidad", ascending=False)

    st.markdown("**Desglose de Probabilidades (Todas las clases):**")
    
    # 1. Mostrar la tabla con los porcentajes exactos
    st.dataframe(df_prob[["Categoría", "Porcentaje Exacto"]], use_container_width=True, hide_index=True)

    # 2. Gráfico Horizontal usando Altair (Soporta versiones antiguas de Streamlit)
    chart = alt.Chart(df_prob).mark_bar().encode(
        x=alt.X('Probabilidad:Q', scale=alt.Scale(domain=[0, 1]), title="Nivel de Confianza"),
        y=alt.Y('Categoría:N', sort='-x', title=""),
        color=alt.condition(
            alt.datum.Categoría == clase_nombre,
            alt.value('#1f77b4'),  # Azul para el ganador
            alt.value('lightgray') # Gris para el resto
        )
    ).properties(height=250)
    
    st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------------------------
# Flujo Principal
# ---------------------------------------------------------------------------
def main():
    modelo_seleccionado = sidebar_controles()
    st.title("🧠 Dashboard Analítico: Redes Neuronales Convolucionales")
    st.markdown("Sistema de visión artificial entrenado bajo CRISP-ML(Q).")
    st.markdown("---")

    ruta = RUTAS_MODELOS[modelo_seleccionado]
    modelo = cargar_modelo(ruta)

    if modelo is None:
        st.error(f"❌ El modelo **{ruta}** no se encuentra.")
        st.stop()

    etiquetas = ETIQUETAS_MNIST if modelo_seleccionado == "MNIST (Dígitos)" else ETIQUETAS_FASHION_MNIST

    imagen_subida = st.file_uploader("Selecciona una imagen a evaluar:", type=["jpg", "jpeg", "png"])

    if imagen_subida is None:
        st.info("⬆️ Sube una imagen para comenzar.")
        st.stop()

    # Preprocesamiento y Predicción
    imagen_procesada = preprocesar_imagen(imagen_subida)
    clase_nombre, confianza, probabilidades = predecir(modelo, imagen_procesada, etiquetas)

    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        columna_imagen(imagen_subida)

    with col_right:
        columna_resultados(clase_nombre, confianza, probabilidades, etiquetas)

    # Llamar a la nueva función de visualización de convoluciones en la base de la pantalla
    visualizar_convoluciones(modelo, imagen_procesada)

if __name__ == "__main__":
    main()
