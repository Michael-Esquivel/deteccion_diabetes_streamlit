from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import dropbox
import pickle


with open('token.txt', "r", encoding="utf-8") as file:
    TOKEN = file.read()
dbx = dropbox.Dropbox(TOKEN)

#dbx = dropbox.Dropbox(st.secrets['TOKEN'])

try:
    with open('sources/datos_procesados_dx.pkl', "wb") as f:
        try:
            metadata, res = dbx.files_download(path='/datos_procesados.pkl')
            f.write(res.content)
            
            df = pd.read_pickle('sources/datos_procesados_dx.pkl')
            print('Archivo extraido exitosamente del DataLake')
        except:
            print('Token vencido, se usara archivo local')
            df = pd.read_pickle('sources/datos_procesados.pkl')
except:
    ...    


with open('sources/minmax_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


def cambiar_menu(menu):
    st.session_state["menu"] = menu

def leer_dato(upload_file):
    dato = pickle.loads(upload_file.getvalue())
    
    return dato

def cargar_modelo(model_path):
    with open(model_path, 'rb') as file:
        xgb_model = pickle.load(file)

    return xgb_model

def predecir(modelo, dato):
    return 'Diabetes' if modelo.predict(dato) else 'No tiene diabetes'

def predecir_inputs(modelo, dato):    
    array = np.array([dato])
    array[:,[0,3,4,5,6,7]] = scaler.transform(array[:,[0,3,4,5,6,7]])

    return 'Diabetes' if modelo.predict(array).round(2) else 'No tiene diabetes'
    
def exportar_datos(name, dato):    
    array = np.array([dato])
    array[:,[0,3,4,5,6,7]] = scaler.transform(array[:,[0,3,4,5,6,7]])

    
    # Botón para generar archivo descargable
    st.download_button(
            label="Descargar archivo .pkl",
            data= pickle.dumps(array),
            file_name=f"{name}.pkl",
            mime="application/octet-stream"
        )
      

def get_export_state(): #Verificar si el boton 'Exportar datos' de 'Digitar datos' se oprimio
    with open('sources/export_state.txt', 'r') as f:
        return int(f.read())

def change_export_state(value):
    with open('sources/export_state.txt', 'w') as f:
        f.write(value)

def title(titulo):
    st.markdown(
        f"""
        <div style="background-color:#00008B;padding:10px;border-radius:10px;text-align:center;margin-bottom:20px;">
            <h2 style="color:white;">{titulo}</h2>
        </div>""",unsafe_allow_html=True)


def estilo_boton():
    st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #008CBA; /* Azul */
        color: white;
        border: 2px solid #005f73; /* Borde */
        border-radius: 8px; /* Bordes redondeados */
        padding: 12px 28px;
        font-size: 18px;
        box-shadow: 3px 3px 6px rgba(0, 0, 0, 0.2); /* Sombra */
        transition: transform 0.2s; /* Efecto al pasar el mouse */
    }
    div.stButton > button:hover {
        transform: scale(1.05); /* Agrandar un poco al pasar el mouse */
        background-color: #0078a5; /* Color más oscuro */
    }
    </style>
    """,
    unsafe_allow_html=True)



def visualizar():
    # =============================================
    # GRAFICO 1: Tasa de Diabetes por Rango de Edad
    # =============================================
    bins = np.arange(0, 1.1, 0.2)
    labels = [f"{int(bins[i] * 100)}-{int(bins[i+1] * 100)}" for i in range(len(bins) - 1)]
    df['age_bin'] = pd.cut(df['age'], bins=bins, right=False, labels=labels)

    # Tasa de diabetes por rango de edad
    age_diabetes_rate = df.groupby('age_bin')['condition_Diabetic'].mean().reset_index()

    # Crear Grafico
    fig_age_diabetes = px.bar(
        age_diabetes_rate,
        x='age_bin',
        y='condition_Diabetic',
        title='Tasa de Diabetes por Rango de Edad',
        labels={'age_bin': 'Rango de Edad', 'condition_Diabetic': 'Proporción Diabéticos'},
        text='condition_Diabetic'
    )

    fig_age_diabetes.update_traces(texttemplate='%{text:.2%}', 
                                   textposition='outside', hovertemplate='%{x}<br>Proporción: %{y:.2%}<extra></extra>')
    fig_age_diabetes.update_layout(
        xaxis_title='Rango de Edad',
        yaxis_title='Proporción de Diabéticos',
        yaxis_tickformat='.0%',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        width=10000,
        height=500
    )


    # =============================================
    # GRAFICO 2: Prevalencia de Diabetes por Género
    # =============================================
    gender_rate = df.groupby('gender')['condition_Diabetic'].mean().reset_index()

    fig_gender_diabetes = px.bar(
        gender_rate,
        x='gender',
        y='condition_Diabetic',
        color='gender',
        title='Prevalencia de Diabetes por Género',
        labels={'condition_Diabetic': 'Proporción Diabéticos'},
        text='condition_Diabetic'
    )

    fig_gender_diabetes.update_traces(texttemplate='%{text:.2%}',
                                      textposition='outside', hovertemplate='%{x}<br>Proporción: %{y:.2%}<extra></extra>')
    fig_gender_diabetes.update_layout(
        xaxis_title='Género',
        yaxis_title='Proporción de Diabéticos',
        yaxis_tickformat='.0%',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        width=1000,  # Ancho del gráfico
        height=600   # Alto del gráfico
    )

    fig_gender_diabetes.update_xaxes(
        categoryorder='array', 
        tickvals=[0, 1], 
        ticktext=['Mujer', 'Hombre']
    )


    # ===================================
    # GRAFICO 3: Umbrales de Variables
    # ===================================
    def plot_threshold_analysis_single(df, variable, bins):
        df_temp = df.copy()

        if variable == 'bmi':
                bins = np.linspace(0, 1, 4)  #
                labels = [f"{int(10 + i * (45 - 10) / 3)}-{int(10 + (i + 1) * (45 - 10) / 3)}" for i in range(3)]
        elif variable == 'glucose_levels':
            bins = np.linspace(0, 1, 5)  
            labels = [f"{int(50 + i * (210 - 50) / 4)}-{int(50 + (i + 1) * (210 - 50) / 4)}" for i in range(4)]
        elif variable == 'blood_pressure':
            bins = np.linspace(0, 1, 4)  
            labels = [f"{int(80 + i * (130 - 80) / 3)}-{int(80 + (i + 1) * (130 - 80) / 3)}" for i in range(3)]
        elif variable == 'platelets':
            bins = np.linspace(0, 1, 4)  
            labels = [f"{int(755 + i * (850000 - 755) / 3)}-{int(755 + (i + 1) * (850000 - 755) / 3)}" for i in range(3)]
        else:
            bins = np.linspace(0, 1, 4)  
            labels = [f"{int(0 + i * (10 - 0) / 3)}-{int(0 + (i + 1) * (10 - 0) / 3)}" for i in range(3)]


        # Categorias de intervalos
        df_temp['bin'] = pd.cut(df_temp[variable], bins=bins, right=False, labels=labels)
        
        
        # Tasa promedio de diabetes por intervalo
        rate = df_temp.groupby('bin')['condition_Diabetic'].mean().reset_index()
        
        variable = {'bmi': 'BMI', 'glucose_levels': 'Niveles de glucosa', 
                'platelets':'Plaquetas', 'serum_creatinine':'Creatinina_serica' }.get(variable, 'Presion arterial')
        
        fig = px.bar(
            rate,
            x='bin',
            y='condition_Diabetic',
            text=[f"{val:.2%}" for val in rate['condition_Diabetic']],  
            title=f'Proporción de Diabetes por Intervalos de {variable}',
            labels={'bin': 'Intervalos', 'condition_Diabetic': 'Proporción Diabéticos'}
        )
        
        fig.update_traces(textposition='outside', marker_color='lightgreen')
        fig.update_layout(
            xaxis_title=f'Intervalos de {variable}',
            yaxis_title='Proporción Diabéticos',
            yaxis_tickformat='.0%',
            width=1500,
            height=500 
        )
        
        return fig


    # Bins para analisis de umbrales
    variables = {
        'bmi': np.linspace(0, 1, 3),
        'glucose_levels': np.linspace(0, 1, 5),
        'blood_pressure': np.linspace(0, 1, 3),
        'platelets': np.linspace(0, 1, 3),
        'serum_creatinine': np.linspace(0, 1, 3)
    }


    # ===================================
    # GRAFICO 4: Matriz de Correlaciones
    # ===================================
    diab = df[df['condition_Diabetic'] == True]
    no_diab = df[df['condition_Diabetic'] == False]

    vars_of_interest = ['bmi', 'blood_pressure', 'glucose_levels']
    corr_diab = diab[vars_of_interest].corr()
    corr_no_diab = no_diab[vars_of_interest].corr()

    fig_corr = make_subplots(rows=1, cols=2, subplot_titles=("Diabéticos", "No Diabéticos"), horizontal_spacing=0.25)

    # Heatmap Diabeticos
    fig_corr.add_trace(
        go.Heatmap(
            z=corr_diab.values,
            x=corr_diab.columns,
            y=corr_diab.index,
            colorscale='Blues',
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(title='Correlación')
        ),
        row=1, col=1
    )

    for i in range(corr_diab.shape[0]):
        for j in range(corr_diab.shape[1]):
            fig_corr.add_annotation(
                text=f"{corr_diab.iloc[i, j]:.2f}",
                x=corr_diab.columns[j],
                y=corr_diab.index[i],
                xref="x1", yref="y1",
                showarrow=False,
                font=dict(color="white")
            )

    # Heatmap No Diabeticos
    fig_corr.add_trace(
        go.Heatmap(
            z=corr_no_diab.values,
            x=corr_no_diab.columns,
            y=corr_no_diab.index,
            colorscale='Blues',
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(title='Correlación')
        ),
        row=1, col=2
    )

    for i in range(corr_no_diab.shape[0]):
        for j in range(corr_no_diab.shape[1]):
            fig_corr.add_annotation(
                text=f"{corr_no_diab.iloc[i, j]:.2f}",
                x=corr_no_diab.columns[j],
                y=corr_no_diab.index[i],
                xref="x2", yref="y2",
                showarrow=False,
                font=dict(color="white")
            )

    fig_corr.update_layout(
        title='Matriz de Correlaciones con Números',
        width=500,
        height=450,
    )


    # =============================================
    # GRAFICO 5: Variables Fisiológicas (Boxplots)
    # =============================================
    df_long = df[['bmi', 'blood_pressure', 'glucose_levels', 'condition_Diabetic']].melt(
        id_vars='condition_Diabetic', 
        var_name='Variable', 
        value_name='Valor'
    )

    fig_vars = px.box(
        df_long, 
        x='Variable', 
        y='Valor',
        color='condition_Diabetic',
        title="Distribución de Variables según Condición Diabética",
        labels={'Variable': 'Variables', 'Valor': 'Valores', 'condition_Diabetic': 'Diabetes'},
        color_discrete_map={True: 'red', False: 'blue'},
        width=1000,
        height=500
    )

    fig_vars.for_each_trace(lambda t: t.update(name='Si' if t.name == 'True' else 'No'))

    fig_vars.update_xaxes(
        categoryorder='array', 
        tickvals=[0, 1, 2], 
        ticktext=['bmi', 'blood_pressure', 'glucose_levels']
    )



    # =========================
    # INTERFAZ STREAMLIT
    # =========================
    opciones = {
        "Tasa de Diabetes por Rango de Edad": fig_age_diabetes,
        "Prevalencia de Diabetes por Género": fig_gender_diabetes,
        "Umbrales Específicos de Indicadores Clínicos": None,
        "Matriz de Correlaciones (Diabéticos vs No Diabéticos)": fig_corr,
        "Distribución de Variables Fisiológicas": fig_vars
    }

    opcion_seleccionada = st.selectbox("Seleccione el grafico a visualizar:", list(opciones.keys()))

    # Mostrar grafico seleccionado
    if opcion_seleccionada == "Umbrales Específicos de Indicadores Clínicos":
        # Selectbox para seleccionar la variable
        variable_seleccionada = st.selectbox("Seleccione la variable:", list(variables.keys()))
        
        bins = variables[variable_seleccionada]
        fig_umbral = plot_threshold_analysis_single(df, variable_seleccionada, bins)
        
        st.plotly_chart(fig_umbral)
    else:
        st.plotly_chart(opciones[opcion_seleccionada], use_container_width=True)