import streamlit as st
from utils import *

st.set_page_config(page_icon='🧬', page_title='Prediccion Diabetes')
estilo_boton()


# Menú principal
if "menu" not in st.session_state:
    st.session_state["menu"] = "inicio"  

if st.session_state["menu"] == "inicio":

    change_export_state('0')
    st.markdown(
        """<div style="background-color:#2C3E50;padding:10px;border-radius:10px;text-align:center;margin-bottom:20px;">
            <h2 style="color:white;">¡Aplicación para la Predicción de Diabetes!</h2>
            <h3 style="color:white;">Explora nuestras funcionalidades</h3></div>""",unsafe_allow_html=True)

    col1, _, col2 = st.columns([1.18, 0.01, 1.18])

    with col1:
        st.markdown(
            """
            <div style="background-color:#1F3A93;padding:1px;border-radius:10px;text-align:center;">
                <h3 style="color:white;">🔍Prediccion de diabetes</h3>
                <p style="color:white;">Modelo de machine learning capaz de detectar la diabetes de acuerdo a diversos factores de salud</p>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """<div style="background-color:#4169E1;padding:1px;border-radius:10px;text-align:center;">
                <h3 style="color:white;">📊Visualizacion de datos</h3><p style="color:white;">Proporciona una visión detallada de los datos usados para entrenar el modelo.</p></div>""", unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)

    _, col1, _, col2, _ = st.columns([0.6, 1, 1, 0.7, 0.4])

    with col1:
        if st.button("Acceder", key=1):
            cambiar_menu("prediccion")  # Cambia al menu de predicción

    with col2:
        if st.button("Acceder", key=2):
            cambiar_menu("visualizacion")  # Cambia al menu de visualizacion


# Menú de predicción
elif st.session_state["menu"] == "prediccion":
    title('🔍 Predicción Automática de Diabetes')

    _, c1, _ = st.columns([21, 611, 1])

    with c1:
            uploaded_files = st.file_uploader("Importe archivos .pkl", type='pkl',
                    key=3, accept_multiple_files=True)  # Permitir múltiples archivos
            
            if uploaded_files:
                for uploaded_file in uploaded_files:

                    try:
                        dato = leer_dato(uploaded_file)
                        xgb_model = cargar_modelo("sources/xgb_model.pkl")
                        prediccion = predecir(xgb_model, dato)

                        if prediccion == 'Diabetes':
                            prediccion_color = f'<span style="color: red;"><strong>{prediccion}</strong></span>'
                        else:
                            prediccion_color = f'<span style="color: green;"><strong>{prediccion}</strong></span>'

                        st.markdown(f'''<div style="background-color: #B0E0E6; padding: 10px; border-radius: 20px; color: #003366;">
                        <strong>Diagnóstico para {uploaded_file.name}: ㅤㅤ{prediccion_color}<strong></div>''', unsafe_allow_html=True)
                        st.markdown('<div style="margin-bottom: 1px;"></div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error procesando {uploaded_file.name}")

            else:
                st.info('Cargue uno o más archivos .pkl')

    st.write("<br>" * 3, unsafe_allow_html=True)

    # Digitacion manual
    if st.button("📝 Digitar datos"):
        cambiar_menu("digitacion")
    
        # Botón para regresar al menu principal
    if st.button("Volver al Menu Principal", key=4):
        cambiar_menu("inicio")


# Menú de digitacion
elif st.session_state["menu"] == "digitacion":
    title('📝 Digitar Datos para Predicción')

    # Campos para los cuadros de texto
    campos = ["Edad", "Genero", "Fuma", "BMIㅤㅤㅤㅤㅤㅤㅤㅤRango(18-30)", "Presion ArterialㅤㅤㅤRango(90-180)", 
              "GlucosaㅤㅤㅤㅤㅤㅤRango(70-200)", "PlaquetasㅤㅤRango(150mil-400mil)", "Creatinina sericaㅤㅤRango(0.6-1.3)"]

    # Diccionario para almacenar valores ingresados
    valores_ingresados = {}

    st.subheader("Ingrese los valores:")

    # Crear cuadricula para distribuir los campos de texto
    for i in range(0, len(campos), 3):  
        cols = st.columns(3)  
        for j, col in enumerate(cols):
            if i + j < len(campos):  
                campo = campos[i + j]
                with col:
                    if campo == "Genero":
                        genero = st.selectbox("Genero", options=["Hombre", "Mujer"], key=5)
                        valores_ingresados["Genero"] = 1 if genero == "Hombre" else 0
                    elif campo == "Fuma":
                        genero = st.selectbox("Fuma", options=["Si", "No"], key=6)
                        valores_ingresados["Fuma"] = 1 if genero == "Si" else 0
                    else:
                        valores_ingresados[campo] = st.text_input(campo, key=f"text_{campo}")

    c1, c2 = st.columns([15, 60])

    valores_lista = 0
    error = False
    prediccion_color = ''

    with c1:
        # Botón para enviar
        if st.button("Predecir", key=7):            
            try:
                change_export_state('0')
                valores_lista = list(valores_ingresados.values())
                valores_lista = [float(str(i).replace(',', '.')) for i in valores_lista] #En caso de recibir int o str con ','
                
                xgb_model = cargar_modelo("sources/xgb_model.pkl")
                prediccion = predecir_inputs(xgb_model, valores_lista)

                if prediccion == 'Diabetes':
                    prediccion_color = f'<span style="color: red;"><strong>{prediccion}</strong></span>'
                else:
                    prediccion_color = f'<span style="color: green;"><strong>{prediccion}</strong></span>'
            except:
                error = True
    
    if prediccion_color: 
        st.markdown(f'''<div style="background-color: #B0E0E6; padding: 10px; border-radius: 20px; color: #003366;">
                <strong>Diagnóstico: ㅤㅤ{prediccion_color}<strong></div>''', unsafe_allow_html=True)
        
    with c2:
        # Botón para exportar datos
        if st.button("Exportar datos", key=7.1):
            change_export_state('1')
            
        if get_export_state():
            try:
                valores_lista = list(valores_ingresados.values())
                valores_lista = [float(str(i).replace(',', '.')) for i in valores_lista] #En caso de recibir int o str con ','

                file_name = st.text_input("Escriba el nombre del archivo (sin extensión):")

                if file_name != "":
                    exportar_datos(file_name, valores_lista)
                    change_export_state('0')
            except:
                error = True   
                    
    if error:
        st.error("❌ Error: Ingrese un numero valido y/o complete todos los campos.")
        

    st.write("<br>" * 3, unsafe_allow_html=True)
    

    # Subida de archivo
    if st.button("📤 Subir archivo"):
        change_export_state('0')
        cambiar_menu("prediccion")

    # Botón para regresar al menu principal
    if st.button("Volver al Menú Principal", key=8):
        cambiar_menu("inicio")



# Menu de visualizacion
elif st.session_state["menu"] == "visualizacion":

    title('📊 Visualizacion de datos')

    info_box_wait = st.info('Generando grafico...')
    visualizar()
    info_box_wait.empty()

    if st.button("Devolver"):
        cambiar_menu("inicio")