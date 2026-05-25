"""
Aplicación Streamlit para el despliegue de modelos CNN
======================================================
Metodología: CRISP-ML(Q)
Modelos:     MNIST (Dígitos Manuscritos) y Fashion MNIST (Prendas de Vestir)
Framework:   Streamlit + TensorFlow/Keras + Pandas + Altair

Autor:       Eliab Ezziel Zamalloa Cayo
Versión:     FINAL (Lógica estable 1.3.0 con UI Optimizada para Móviles)
"""

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import io
import urllib.request 
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D
from streamlit_drawable_canvas import st_canvas

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
def preprocesar_imagen(img):
    img = img.convert("L")
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
    st.markdown("---")
    st.subheader("👁️ Análisis Interno: ¿Qué está viendo la CNN?")
    st.write("Los siguientes **Mapas de Características** muestran la salida matemática de la primera capa convolucional de la Red Neuronal, revelando los bordes y texturas que la IA considera importantes.")

    capa_conv = None
    for layer in modelo.layers:
        if isinstance(layer, Conv2D):
            capa_conv = layer
            break

    if capa_conv is not None:
        modelo_intermedio = Model(inputs=modelo.inputs, outputs=capa_conv.output)
        mapas = modelo_intermedio.predict(imagen_procesada, verbose=0)

        num_filtros = mapas.shape[-1]
        filtros_a_mostrar = min(num_filtros, 6)
        
        cols = st.columns(filtros_a_mostrar)
        for i in range(filtros_a_mostrar):
            mapa_img = mapas[0, :, :, i]
            
            mapa_img -= mapa_img.min()
            if mapa_img.max() > 0:
                mapa_img /= mapa_img.max()
            mapa_img *= 255
            mapa_img = np.clip(mapa_img, 0, 255).astype(np.uint8)

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
                2. Usa un archivo local, una URL, la cámara o la pizarra.
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

def columna_imagen(img):
    st.subheader("📷 Imagen de entrada")
    st.image(img, caption="Imagen original lista para analizar", width=250)

def columna_resultados(clase_nombre, confianza, probabilidades, etiquetas):
    st.subheader("📊 Resultados de la IA")

    if clase_nombre is None:
        st.warning("Error en predicción.")
        return

    st.metric(label="Predicción Principal", value=clase_nombre, delta=f"{confianza:.2%} de certeza")
    st.markdown("---")

    df_prob = pd.DataFrame({
        "Categoría": etiquetas,
        "Probabilidad": probabilidades,
        "Porcentaje Exacto": [f"{p*100:.2f}%" for p in probabilidades]
    }).sort_values("Probabilidad", ascending=False)

    st.markdown("**Desglose de Probabilidades (Todas las clases):**")
    st.dataframe(df_prob[["Categoría", "Porcentaje Exacto"]], use_container_width=True, hide_index=True)

    chart = alt.Chart(df_prob).mark_bar().encode(
        x=alt.X('Probabilidad:Q', scale=alt.Scale(domain=[0, 1]), title="Nivel de Confianza"),
        y=alt.Y('Categoría:N', sort='-x', title=""),
        color=alt.condition(
            alt.datum.Categoría == clase_nombre,
            alt.value('#1f77b4'), 
            alt.value('lightgray')
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
    es_mnist = (modelo_seleccionado == "MNIST (Dígitos)")
    
    imagen_final = None 

    st.subheader("📥 Método de ingreso de datos")

    # =========================================================================
    # LÓGICA ESTABLE: Menú desplegable en lugar de pestañas rotas (Tabs)
    # =========================================================================
    opciones = ["📁 Subir archivo local", "🌐 Pegar URL"]
    if es_mnist:
        opciones.append("✍️ Dibujar en Pizarra")
    else:
        opciones.append("📸 Tomar Foto con Cámara")

    metodo = st.selectbox("Selecciona cómo quieres evaluar la imagen:", opciones)
    st.write("") # Espaciador

    if metodo == "📁 Subir archivo local":
        archivo_local = st.file_uploader("Arrastra tu imagen (JPG / PNG):", type=["jpg", "jpeg", "png"])
        if archivo_local is not None:
            imagen_final = Image.open(archivo_local)

    elif metodo == "🌐 Pegar URL":
        url_imagen = st.text_input("Ingresa el enlace directo a una imagen:")
        if url_imagen:
            try:
                req = urllib.request.Request(url_imagen, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    imagen_final = Image.open(io.BytesIO(response.read()))
            except Exception as e:
                st.error(f"⚠️ No se pudo descargar la imagen: {e}")

    elif metodo == "✍️ Dibujar en Pizarra":
        st.write("**Dibuja un número del 0 al 9 en el cuadro negro:**")
        
        # Centramos la pizarra para que se vea impecable en móviles
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            canvas_result = st_canvas(
                fill_color="#000000",
                stroke_width=20,       
                stroke_color="#FFFFFF",
                background_color="#000000", 
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas_estable" 
            )
            
            if st.button("Analizar Dibujo", use_container_width=True, type="primary"):
                # Verificación defensiva contra el NameError
                if canvas_result is not None and getattr(canvas_result, 'image_data', None) is not None:
                    imagen_final = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                else:
                    st.warning("⚠️ Dibuja algo en el recuadro antes de analizar.")

    elif metodo == "📸 Tomar Foto con Cámara":
        st.write("**Usa tu cámara para tomarle foto a una prenda de ropa:**")
        foto_camara = st.camera_input("Capturar Prenda")
        if foto_camara is not None:
            imagen_final = Image.open(foto_camara)
    # =========================================================================

    # -----------------------------------------------------------------------
    # Ejecución si hay una imagen lista
    # -----------------------------------------------------------------------
    if imagen_final is None:
        st.info("⬆️ Esperando imagen para comenzar el análisis...")
        st.stop()

    # Preprocesamiento y Predicción
    imagen_procesada = preprocesar_imagen(imagen_final)
    clase_nombre, confianza, probabilidades = predecir(modelo, imagen_procesada, etiquetas)

    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        columna_imagen(imagen_final)

    with col_right:
        columna_resultados(clase_nombre, confianza, probabilidades, etiquetas)

    visualizar_convoluciones(modelo, imagen_procesada)

if __name__ == "__main__":
    main()
