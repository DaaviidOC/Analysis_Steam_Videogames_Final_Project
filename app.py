import fire
import streamlit as st
import matplotlib as plt
import base64
import numpy as np
import pandas as pd
import seaborn as sns
import plotly_express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from streamlit_option_menu import option_menu
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import base64
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import requests
from bs4 import BeautifulSoup
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import re
import streamlit.components.v1 as components
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from streamlit.components.v1 import iframe

st.set_page_config(page_title="STEAM | DAVID OFICIAL",
        layout="centered",
        page_icon="🎮",
        )

# Definición de la función y añadimos el fondo de la página
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Llamamos a la función
add_bg_from_local("imgs/steamfondo2.jpg")



st.image("imgs/steam_banner.jpg")

# Definir los títulos de tus opciones de menú
menu_options = [
    "Inicio 🏠", "Análisis inicial 📊", "Gráficos avanzados 📈", "Minería ⛏️",
    "Serie temporal 📅", "Predicción de éxito ✔", "Predicción de precio 💵", "Power BI 〽️"
]

# Crear una función para generar botones en filas
def generate_menu_in_rows(options, per_row):
    if 'selected_menu_option' not in st.session_state:
        st.session_state['selected_menu_option'] = options[0]  # Establecer "Inicio" como predeterminado
    
    num_rows = len(options) // per_row + (1 if len(options) % per_row else 0)
    for row in range(num_rows):
        cols = st.columns(per_row)
        for idx, col in enumerate(cols):
            option_index = row * per_row + idx
            if option_index < len(options):
                with col:
                    button_key = f"button_{options[option_index]}"
                    if st.button(options[option_index], key=button_key):
                        st.session_state['selected_menu_option'] = options[option_index]

# En el cuerpo principal del script, se llama a la función sin necesidad de capturar el valor de retorno
generate_menu_in_rows(menu_options, per_row=4)

# Para mostrar el contenido, verificamos el estado de la sesión
selected_menu_option = st.session_state['selected_menu_option']

# CONTENT INICIO
if selected_menu_option =="Inicio 🏠":
    
    # ¿QUÉ ES STEAM?
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
                <h2 style='text-align: center;  
                background-color: #131720; 
                font-family: Roboto;
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                '>¿QUÉ ES STEAM?</h2>""" ,unsafe_allow_html=True) 
    
    st.markdown(
    """
    <div style="border-bottom-left-radius: 10px; 
    border-bottom-right-radius: 10px; 
    padding: 20px; 
    text-align: justify; 
    background-color: #131720; 
    color: #FFFFFF; 
    font-family: Roboto;">
    Steam  es una plataforma de distribución digital de videojuegos desarrollada por Valve Corporation. Fue lanzada en septiembre de 2003 como una forma para Valve de proveer actualizaciones automáticas a sus juegos, pero finalmente se amplió para incluir juegos de terceros. Steam ofrece protección contra piratería, servidores de emparejamiento, transmisiones de vídeo y servicios de redes sociales. </br></br>

    También proporciona al usuario la instalación y la actualización automática de juegos y características de comunidad como grupos y listas de amigos, guardado en la nube, voz en el juego y funcionalidad de chat. Se utiliza tanto por pequeños desarrolladores independientes como grandes corporaciones de software para la distribución de videojuegos y material multimedia relacionado.

    Para poder disfrutar de todos estos servicios, es necesario estar registrado en el servicio mediante la creación de una cuenta gratuita, a la que se vinculan los videojuegos comprados por el jugador. Estos juegos pueden ser tanto los juegos que se ofrecen para la compra en la propia plataforma como ciertos juegos comprados en tiendas físicas.
    </div>
    </br></br>
    """, unsafe_allow_html=True)
    
    # PRODUCTOS MÁS FAMOSOS
    st.markdown("""
                <h2 style='text-align: center;  
                background-color: #131720; 
                font-family: Roboto;
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                '>PRODUCTOS MÁS FAMOSOS</h2>
                <h3 style='text-align: center;  
                background-color: #131720; 
                font-family: Roboto;
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                '>STEAM CONTROLLER</h3>""" ,unsafe_allow_html=True) 
    
    # Primero, convierte la imagen a base64
    with open("imgs/steam_controller.jpg", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()

    # Luego, incluye la imagen en tu bloque de Markdown usando una etiqueta <img>
    st.markdown(f"""
    <div style="border-bottom-left-radius: 10px; 
    border-bottom-right-radius: 10px; 
    padding: 20px; 
    text-align: justify; 
    background-color: #131720; 
    color: #FFFFFF; 
    font-family: Roboto;">
    <img src="data:image/jpeg;base64,{b64_string}" 
    style="border-radius: 10px; 
    display: block; 
    margin-left: auto; 
    margin-right: auto; 
    padding-bottom: 20px">
    Steam Controller te permite jugar tu colección de juegos de Steam al completo en tu TV, incluso los diseñados sin compatibilidad con mando. Contando con trackpads duales, respuesta háptica en HD, gatillos de doble pulsación, botones traseros y plantillas de controles totalmente personalizables, el Steam Controller ofrece un nuevo nivel de control preciso. Encuentra tu configuración favorita en la Comunidad Steam, o crea y comparte la tuya propia.
    
    </div>
    </br></br>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""
                <h3 style='text-align: center;  
                background-color: #131720; 
                font-family: Roboto;
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                '>STEAM DECK</h3>""" ,unsafe_allow_html=True) 
    
    # Primero, convierte la imagen a base64
    with open("imgs/steamdeck.jpg", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()

    # Luego, incluye la imagen en tu bloque de Markdown usando una etiqueta <img>
    st.markdown(f"""
    <div style="border-bottom-left-radius: 10px; 
    border-bottom-right-radius: 10px; 
    padding: 20px; 
    text-align: justify; 
    background-color: #131720; 
    color: #FFFFFF; 
    font-family: Roboto;">
    <img src="data:image/jpeg;base64,{b64_string}" 
    style="border-radius: 10px; 
    display: block; 
    margin-left: auto; 
    margin-right: auto; 
    padding-bottom: 20px;
    width: 450px;
    height: 300px;">
    La Steam Deck es un dispositivo de juego para PC portátiles potentes diseñados para brindar comodidad y una experiencia similar a la de una consola. Steam Deck tiene una interfaz intuitiva diseñada expresamente para sus controles de mando. Tanto el software como el sistema operativo se crearon a medida para Steam Deck, lo cual hace que este dispositivo sea la forma más fácil de empezar a jugar en PC.
    
    </div>
    </br></br>
    """, unsafe_allow_html=True)

    st.markdown("""
                <h3 style='text-align: center;  
                background-color: #131720; 
                font-family: Roboto;
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                '>VALVE INDEX</h3>""" ,unsafe_allow_html=True) 
    
    # Primero, convierte la imagen a base64
    with open("imgs/valveindex.jpg", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()

    # Luego, incluye la imagen en tu bloque de Markdown usando una etiqueta <img>
    st.markdown(f"""
    <div style="border-bottom-left-radius: 10px; 
    border-bottom-right-radius: 10px; 
    padding: 20px; 
    text-align: justify; 
    background-color: #131720; 
    color: #FFFFFF; 
    font-family: Roboto;">
    <img src="data:image/jpeg;base64,{b64_string}" 
    style="border-radius: 10px; 
    display: block; 
    margin-left: auto; 
    margin-right: auto; 
    padding-bottom: 20px;
    width: 450px;
    height: 300px;">
    Valve Index es un sistema de realidad virtual diseñado por Valve Corporation, empresa responsable de la plataforma de distribución de videojuegos Steam.​ El sistema fue anunciado oficialmente a finales de abril de 2019, y salió a la venta el junio del mismo año a un precio inicial de 999 dólares. 
    </div>
    </br>
    """, unsafe_allow_html=True)
    
    # INTRO SIDEBAR
    
    intro_sidebar = False
    if(st.button("Valve")):
        intro_sidebar = True
        if intro_sidebar:
            with st.sidebar:
                st.sidebar.image('imgs/Valve_logo.svg')
                st.write("Valve Corporation, también conocido como Valve Software, es una empresa estadounidense desarrolladora de videojuegos. Se hizo mundialmente famosa por su primer juego, Half-Life y por una modificación de este, Counter-Strike.")
                st.write("Otros de sus logros más famosos son la creación del motor de videojuego Source, utilizado en gran parte de sus videojuegos, incluyendo Half-Life 2, Portal, Team Fortress 2, Left 4 Dead, Left 4 Dead 2 y Dota 2 y la creación de la plataforma digital de videojuegos Steam. Las oficinas centrales de Valve Software se encuentran en Bellevue, Washington, Estados Unidos.")
                st.write("Valve fue fundada en 1996 por Gabe Newell y Mike Harrington. Ambos eran trabajadores del gigante informático Microsoft y habían trabajado en sus sistemas operativos Windows y OS/2 antes de pasarse al mundo de los videojuegos.")
                st.sidebar.image('imgs/valve.jpg')
        if st.button("Ocultar info de Valve"):
            intro_sidebar = False

# CONTENT ANÁLISIS INICIAL
if selected_menu_option =="Análisis inicial 📊":
    
    st.image("graphs/3.png", width=700)      
    st.image("graphs/1.png", width=700)
    st.image("graphs/2.png", width=700)
    st.image("graphs/4.png", width=700)
    st.image("graphs/5.png", width=700)
    st.image("graphs/6.png", width=700)
    st.image("graphs/7.png", width=700)
    
    # SIDEBAR
    with st.sidebar:
            st.title("Microsoft actualizará Windows 11 con una función de IA para mejorar el rendimiento de los juegos")
            st.sidebar.image('imgs/windows.jpg')
            st.header("Será una herramienta de reescalado y suavizado de texturas similar a NVIDIA DLSS o AMD FSR y llegará para darnos unos FPS extra en los juegos que sean compatibles")
            st.write("Mientras esperamos a que Microsoft nos dé noticias sobre la nueva estrategia de Xbox esta misma semana, los de Redmnond trabajan en paralelo en muchas otras cosas, algunas relacionadas también con el mundo de los videojuegos como una nueva herramienta que va a llegar próximamente a windows 11.")
            st.write("Y es que la próxima gran actualización para el sistema operativo, la 24H2 llegará cargada de nuevas funciones basadas en IA, muchas de ellas pensadas para dar más posibilidades a Copilot aunque, entre todas, encontramos una nueva tecnología basada IA que servirá para mejorar el rendimiento de ciertos juegos.")
            st.write("Esta función se trata de Auto Super Resolution y ya ha podido verse en la última versión beta de Windows 11.")
            st.markdown("[NOTÍCIA COMPLETA](https://vandal.elespanol.com/noticia/1350769070/microsoft-actualizara-windows-11-con-una-funcion-de-ia-para-mejorar-el-rendimiento-de-los-juegos/)", unsafe_allow_html=True)
    
    
    
if selected_menu_option =="Gráficos avanzados 📈":

    HtmlFile = open("graphs/7.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read() 
    style = """
    <style>
        body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden;
            background-color: #111111;
        }
    </style>
    """
    source_code_with_style = style + source_code
    components.html(source_code_with_style, width=700, height=500)
    
    HtmlFile = open("graphs/8.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read() 
    style = """
    <style>
        body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden;
            background-color: #111111;
        }
    </style>
    """
    source_code_with_style = style + source_code
    components.html(source_code_with_style, width=700, height=500)
    
    HtmlFile = open("graphs/9.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read() 
    style = """
    <style>
        body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden;
            background-color: #111111;
        }
    </style>
    """
    source_code_with_style = style + source_code
    components.html(source_code_with_style, width=700, height=500)
    
    HtmlFile = open("graphs/10.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read() 
    style = """
    <style>
        body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden;
            background-color: #111111;
        }
    </style>
    """
    source_code_with_style = style + source_code
    components.html(source_code_with_style, width=700, height=500)
    
     
    HtmlFile = open("graphs/12.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read() 
    style = """
    <style>
        body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden;
            background-color: #111111;
        }
    </style>
    """
    source_code_with_style = style + source_code
    components.html(source_code_with_style, width=700, height=500)
     
    HtmlFile = open("graphs/13.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read() 
    style = """
    <style>
        body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden;
            background-color: #111111;
        }
    </style>
    """
    source_code_with_style = style + source_code
    components.html(source_code_with_style, width=700, height=500)
    
    HtmlFile = open("graphs/14.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read() 
    style = """
    <style>
        body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden;
            background-color: #111111;
        }
    </style>
    """
    source_code_with_style = style + source_code
    components.html(source_code_with_style, width=700, height=500)
    
if selected_menu_option == "Minería ⛏️":
    st.markdown("""<br><br>""", unsafe_allow_html=True)
    st.image("imgs/your_wordcloud2.png", width=700)
    
    
if selected_menu_option == "Predicción de éxito ✔":
    with st.sidebar:
            st.title("Quédate gratis para siempre RPG Maker XP en Steam si lo reclamas antes del 19 de febrero")
            st.sidebar.image('imgs/noticias1.jpg')
            st.header("Festival de RPG Maker en Seam. La versión del motor RPG Maker XP para crear juegos de rol gratis para siempre")
            st.write("La creación de juegos nunca ha sido tan sencilla como ahora gracias a varias herramientas. Una de las más conocidas y difundidas es el motor RPG Maker, que celebra su día el 15 de febrero. El actual responsable de RPG Naker, Gotcha  Gotcha Games, ha organizado un festival en Steam con descuentos en las distintas versiones del motor y RPG Maker XP gratis si se compra hasta el 19 de  febrero.")
            st.write("RPG Maker XP salió en 2005 en Steam y con él se puede crear un juego de rol sin tener formaicón previa. Incluye un conjunto básico de gráficos y sonidos que se pueden alterar y también se pueden importar otros extremos.")
            st.markdown("[NOTÍCIA COMPLETA](https://vandal.elespanol.com/noticia/1350769090/quedate-gratis-para-siempre-rpg-maker-xp-en-steam-si-lo-reclamas-antes-del-19-de-febrero/)", unsafe_allow_html=True)
            
    
    videogames_data = pd.read_csv('data/videogames_eda.csv')
    videogames_data = videogames_data.loc[videogames_data['release_year'] > 2000]
    
    # Obtenemos listado de desarrolladores únicos
    developers_list = videogames_data['developer'].unique().tolist()
    
    # Ordenamos el lsitado
    developers_list.sort()
    
        # Cargar el modelo entrenado
    with open('models/classsification_success.pkl', 'rb') as file:
        model = pickle.load(file)

    # Crear el formulario para recoger entradas del usuario
    with st.form("prediction_form"):
        st.write("Introduce los datos del videojuego:")
        positive = st.number_input('Reviews positivas', min_value=0)
        negative = st.number_input('Reviews negativas', min_value=0)
        average_forever = st.number_input('Media de tiempo jugado total', min_value=0)
        price = st.number_input('Precio', min_value=0.0)
        
        # Usar un selectbox para el campo 'developer'
        developer = st.selectbox('Desarrollador', developers_list)
        
        # Botón de envío del formulario
        submit_button = st.form_submit_button(label="Predict")
    
        
    if submit_button:
        data = pd.DataFrame([[positive, negative, average_forever, price, developer]],
                            columns=['positive', 'negative', 'average_forever', 'price', 'developer'])
        
        # Hacer la predicción
        prediction = model.predict(data)
        
        # Mostrar el resultado
        if prediction[0] == 1:
            st.success("El videojuego será exitoso!")
        else:
            st.error("El videojuego podría no ser exitoso.")
            
    # Aplicamos CSS personalizado para cambiar el color de fondo del formulario
    st.markdown("""
    <style>
    [data-testid="stForm"] {
        background-color: #131720;
    }
    </style>
    """, unsafe_allow_html=True)
    

if selected_menu_option =="Serie temporal 📅":

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
                <h3 style='text-align: center;  
                background-color: #131720; 
                font-family: Roboto;
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                '>Revisamos el promedio de los precios de los videojuegos por año</h3>""" ,unsafe_allow_html=True) 
    
    
    HtmlFile = open("graphs/tu_gráfico.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read() 
    style = """
    <style>
        body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden;
            background-color: #111111;
        }
    </style>
    """
    source_code_with_style = style + source_code
    components.html(source_code_with_style, width=700, height=500)
    
    st.markdown("""
                <h3 style='text-align: center;  
                background-color: #131720; 
                font-family: Roboto;
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                '>Revisamos estacionariedad</h3>""" ,unsafe_allow_html=True) 
    
    st.image("graphs/serie1.png", width=700)
    
    st.markdown("""
                <h3 style='text-align: center;  
                background-color: #131720; 
                font-family: Roboto;
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                '>Diferenciamos</h3>""" ,unsafe_allow_html=True) 
    
    st.image("graphs/serie2.png", width=700)
    
    st.markdown("""
                <h3 style='text-align: center;  
                background-color: #131720; 
                font-family: Roboto;
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                '>Entrenamos modelo ARIMA y revisamos diagnósticos</h3>""" ,unsafe_allow_html=True) 
    
    st.image("graphs/serie3.png", width=700)
    
    st.markdown("""
                <h3 style='text-align: center;  
                background-color: #131720; 
                font-family: Roboto;
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                '>Graficamos precios promedios de los videojuegos por año</h3>""" ,unsafe_allow_html=True) 
    
    
    
    HtmlFile = open("graphs/stemporal.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read() 
    style = """
    <style>
        body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden;
            background-color: #111111;
        }
    </style>
    """
    source_code_with_style = style + source_code
    components.html(source_code_with_style, width=700, height=500)
    
    
    
    HtmlFile = open("graphs/stemporal2.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read() 
    style = """
    <style>
        body {
            background-color: #111111;
        }
    </style>
    """
    source_code_with_style = style + source_code
    components.html(source_code_with_style, width=700, height=500)


if selected_menu_option == "Predicción de precio 💵":
    with st.sidebar:
        st.title("Hogwarts Legacy repite como juego más vendido durante la última semana en Reino Unido")
        st.sidebar.image('imgs/hogwarts.jpg')
        st.header("Fue el juego más vendido en 2023 y todavía se mantiene como uno de los éxitos de la temporada.")
        st.write("Hogwarts Legacy vuelve al primer puesto de ventas semanales físicas en Reino Unido, desplazando a Suicide Suad: Kill the Justice League -anterior número uno- al tercer puesto.")
        st.write("El RPG de acción ha cumplido un año desde su debut y en este tiempo se ha convertido en el más vendido de 2023.")
        st.markdown("[NOTICIA COMPLETA](https://vandal.elespanol.com/noticia/1350769071/hogwarts-legacy-repite-como-juego-mas-vendido-durante-la-ultima-semana-en-reino-unido/)", unsafe_allow_html=True)
    
    # Cargar el modelo entrenado
    with open('models/regression_price.pkl', 'rb') as file:
        model = pickle.load(file)
    
    videogames_data = pd.read_csv('data/videogames_eda.csv')
    videogames_data = videogames_data.loc[videogames_data['release_year'] > 2000]
    
    # Obtenemos listado de desarrolladores únicos
    developers_list = videogames_data['developer'].unique().tolist()
    developers_list.sort()
    
    # Definir las columnas numéricas que queremos escalar basadas en tu entrenamiento
    variables_to_scale = ['positive', 'negative', 'average_forever', 'owners_max', 'release_year']
    
    scaler = StandardScaler()
    videogames_data[variables_to_scale] = scaler.fit_transform(videogames_data[variables_to_scale])

    
    # Crear el formulario para recoger entradas del usuario
    with st.form("prediction_form"):
        st.write("Introduce los datos del videojuego:")
        positive = st.number_input('Reviews positivas', min_value=0)
        negative = st.number_input('Reviews negativas', min_value=0)
        average_forever = st.number_input('Media total de tiempo jugado', min_value=0)
        owners_max = st.number_input('Nº de propietarios máximo', min_value=0)
        release_year = st.number_input('Año de lanzamiento', min_value=2000, max_value=2026, step=1)
        developer = st.selectbox('Desarrollador', developers_list)
        submit_button = st.form_submit_button("Predicción")
        
    
    if submit_button:
        input_df = pd.DataFrame([[positive, negative, average_forever, owners_max, release_year]],
                            columns=['positive', 'negative', 'average_forever', 'owners_max', 'release_year'])
    
        input_df[developer] = 1
        
        # Asegurar que todas las otras columnas de 'developer' estén presentes y tengan valor 0
        for dev in developers_list:
            if dev not in input_df.columns:
                input_df[dev] = 0

        # Escalar las variables numéricas
        input_df[variables_to_scale] = scaler.transform(input_df[variables_to_scale])
    
         # Asegurar el orden correcto de las columnas como se entrenó el modelo
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
        # Hacer la predicción
        prediction = model.predict(input_df)
        
        # Mostrar el resultado
        st.write(f'El precio predicho del videojuego es: €{prediction[0]:.2f}')
    
    st.markdown("""
    <style>
    [data-testid="stForm"] {
        background-color: #131720;
    }
    </style>
    """, unsafe_allow_html=True)

if selected_menu_option =="Power BI 〽️":
    
    st.markdown("""<br>""", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
                <h2 style='text-align: center;  
                background-color: #131720; 
                font-family: Roboto;
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                '>PANEL POWER BI</h2>""" ,unsafe_allow_html=True) 
    
    st.markdown("""
        <div "style=margin: auto;
        background-color: #131720;">
            <iframe title="videogames" width="100%" height="500" background-color: #131720 src="https://app.powerbi.com/view?r=eyJrIjoiNjdjMWQxZmUtMzJmOS00NzlhLWIyMTUtMzY5Mjg5ZDMxNTEyIiwidCI6IjhhZWJkZGI2LTM0MTgtNDNhMS1hMjU1LWI5NjQxODZlY2M2NCIsImMiOjl9" frameborder="0" allowFullScreen="true" &navContentPaneEnabled=false>
            </iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )

    
  
    
    
