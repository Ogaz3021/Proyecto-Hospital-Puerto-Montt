# --- IMPORTS NECESARIOS ---
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import Fullscreen, MarkerCluster
from branca.element import Element
import os
import logging
from datetime import datetime  
import numpy as np 

# Configuración de logging 
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- Se necesita una 'secret_key' para que funcionen los mensajes flash ---
app.secret_key = 'tu_llave_secreta_aleatoria_aqui'
# -------------------------------------------------------------------------------


# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---

# Rutas a los archivos
GEOMETRY_PATH = 'llanquihue_comunas.geojson'
DATA_PATH = 'BASE FINAL CON CLUSTER.csv'

# Coordenadas del Hospital de Puerto Montt
HPM_LOCATION = (-41.44763238267444, -72.95692300523349)

df_original = pd.DataFrame()
gdf_comunas = None

# Colores y títulos para la leyenda de Severidad
COLORES_SEVERIDAD = {
    'Mayor': 'red',
    'Moderada': 'orange',
    'Menor': 'green'
}

# Columnas para los filtros de Amputación (usadas en el frontend)
COLUMNAS_AMPUTACION = [
    'AMP_DEDO_MANO', 'AMP_PULGAR', 'AMP_DEDO_PIE', 'AMP_A_NIVEL_PIE',
    'DESART_TOBILLO', 'AMP_NIVEL_MALEOLO', 'AMP_DEBAJO_RODILLA',
    'DESART_RODILLA', 'AMP_ENCIMA_RODILLA'
]

# --- Funciones de Carga de Datos ---

def load_data(csv_file_path):
    global app
    try:
        # Usamos read_csv con separador ;
        df = pd.read_csv(csv_file_path, sep=';')
    except FileNotFoundError:
        app.logger.error(f"Error: No se encontró el archivo '{csv_file_path}'.")
        return pd.DataFrame()
    
    # *** BLOQUE DE RENOMBRADO CRUCIAL ***
    # Renombra las columnas del CSV (izquierda) a los nombres que usa la aplicación (derecha)
    df = df.rename(columns={
        'Sexo(Desc)': 'Sexo (Desc)',
        'UltimaEdadRegistrada': 'Ultima Edad Registrada',
        'UltimaRegistroSeveridad': 'Ultima registro severidad',
        'tiempo(minutos)': 'tiempo (minutos)'
    })
    # ****************************************

    # Filtro inicial de comunas (por si los datos contienen otras provincias)
    comunas_llanquihue = ['Puerto Montt', 'Calbuco', 'Cochamó', 'Fresia', 'Frutillar', 
                          'Llanquihue', 'Los Muermos', 'Maullín', 'Puerto Varas']
    
    df = df[df['Comuna'].isin(comunas_llanquihue)].copy()
    
    # Aseguramos la limpieza de coordenadas
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df.dropna(subset=['lat', 'lng'], inplace=True)
    
    # IMPORTANTE: Asegurar que la columna de edad sea numérica
    df['Ultima Edad Registrada'] = pd.to_numeric(df['Ultima Edad Registrada'], errors='coerce')
    df.dropna(subset=['Ultima Edad Registrada'], inplace=True)
    
    return df

def load_shapes(geojson_path):
    global app
    try:
        gdf_llanquihue = gpd.read_file(geojson_path)
        gdf_llanquihue = gdf_llanquihue.to_crs(epsg=4326)
        
        # --- LÓGICA DE RENOMBRADO PARA TOOLTIP ---
        nombre_columna_comuna = None
        if 'Comuna' in gdf_llanquihue.columns:
            nombre_columna_comuna = 'Comuna'
        elif 'COMUNA' in gdf_llanquihue.columns:
            nombre_columna_comuna = 'COMUNA'
        elif 'NAME' in gdf_llanquihue.columns:
            nombre_columna_comuna = 'NAME'

        if nombre_columna_comuna:
            # La columna 'Comuna_Corregida' es la que usa el tooltip en generate_map_html
            gdf_llanquihue['Comuna_Corregida'] = gdf_llanquihue[nombre_columna_comuna]
        else:
            gdf_llanquihue['Comuna_Corregida'] = 'Comuna Desconocida'
            app.logger.warning("No se encontró columna de comuna ('Comuna', 'COMUNA', o 'NAME') en el GeoJSON.")
        # --- FIN LÓGICA ---

        return gdf_llanquihue
    except Exception as e:
        app.logger.error(f"Error al cargar el GeoJSON ('{geojson_path}'): {e}")
        return None

# --- EJECUCIÓN DE CARGA GLOBAL ---
df_original = load_data(DATA_PATH)
gdf_comunas = load_shapes(GEOMETRY_PATH)
# -----------------------------------


# --- 2. FUNCIONES DEL BACKEND ---

def add_severity_legend(mapa):
    """Añade una leyenda HTML/CSS al mapa de Folium."""
    
    legend_html = """
    <div style="position: fixed; 
                 top: 20px; right: 20px; width: 150px; height: 120px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white; opacity:0.9; padding: 10px;">
        <b>Última Severidad</b><br>
    """

    for severity, color in COLORES_SEVERIDAD.items():
        legend_html += f"""
          <i style="background:{color}; border-radius:50%; display:inline-block; width:10px; height:10px; margin-right:5px;"></i> {severity}<br>
        """
    legend_html += "</div>"

    mapa.get_root().html.add_child(folium.Element(legend_html))


def generate_map_html(df_filtrado, HPM_LOCATION):
    """Genera el mapa de Folium con los datos ya filtrados."""
    global df_original, gdf_comunas
    
    if df_filtrado.empty:
        lat_center = HPM_LOCATION[0]
        lng_center = HPM_LOCATION[1]
        
        mapa = folium.Map(location=[lat_center, lng_center], zoom_start=9)
        folium.Marker(location=[lat_center, lng_center], popup='No hay datos filtrados').add_to(mapa)
        
        folium.Marker(
            location=HPM_LOCATION,
            popup="Hospital Puerto Montt (HPM)",
            tooltip="Hospital Puerto Montt",
            icon=folium.Icon(color='blue', icon='hospital', prefix='fa')
        ).add_to(mapa)
        
        add_severity_legend(mapa)
        return mapa._repr_html_()

    lat_media = df_filtrado['lat'].mean()
    lng_media = df_filtrado['lng'].mean()

    mapa = folium.Map(location=[lat_media, lng_media], zoom_start=9, tiles="OpenStreetMap")
    Fullscreen().add_to(mapa)
    
    # Añadir GeoJSON de Comunas
    if gdf_comunas is not None:
        style_comunas = {'fillColor': '#222222', 'color': '#FFFFFF', 'weight': 1.5, 'fillOpacity': 0.1}
        folium.GeoJson(
            gdf_comunas,
            name='Bordes Comunales',
            style_function=lambda x: style_comunas,
            # Tooltip ya configurado para mostrar 'Comuna_Corregida'
            tooltip=folium.GeoJsonTooltip(fields=['Comuna_Corregida'], aliases=['Comuna:']),
        ).add_to(mapa)
    
    # Añadir Marcador del Hospital de Puerto Montt (HPM)
    folium.Marker(
        location=HPM_LOCATION,
        popup="Hospital Puerto Montt (HPM)",
        tooltip="Hospital Puerto Montt",
        icon=folium.Icon(color='blue', icon='hospital', prefix='fa')
    ).add_to(mapa)


    for _, row in df_filtrado.iterrows():
        popup_html = f"""
        <b>Código PPD:</b> {row['Codigo']}<br>
        <b>Comuna:</b> {row['Comuna']}<br>
        <b>Sexo:</b> {row['Sexo (Desc)']}<br>
        <b>Edad:</b> {row['Ultima Edad Registrada']}<br>
        <b>Severidad:</b> {row['Ultima registro severidad']}<br>
        <hr>
        <b>Tiempo a HPM:</b> {row['tiempo (minutos)']:.1f} min<br>
        <b>Distancia:</b> {row['km']:.1f} km<br>
        <b>Total Amputaciones:</b> {row['Total_Amputaciones']}<br>
        <b>Grupo:</b> {row['Grupo']}
        """
        popup = folium.Popup(popup_html, max_width=300)
        
        severidad = row['Ultima registro severidad']
        color = COLORES_SEVERIDAD.get(severidad, 'gray')
        
        marcador = folium.Marker(
            location=[row['lat'], row['lng']],
            popup=popup,
            tooltip=f"PPD: {row['Codigo']} ({severidad})",
            icon=folium.Icon(color=color, icon='user', prefix='fa')
        )
        
        marcador.add_to(mapa)  

    add_severity_legend(mapa)

    return mapa._repr_html_()


# --- 3. RUTAS DE FLASK ---

@app.route('/')
@app.route('/inicio')
def mostrar_inicio_y_diagrama():
    """Muestra la página de inicio/diagrama de arquitectura."""
    return render_template('diagrama_arquitectura.html')

@app.route('/dashboard') 
def dashboard():
    """Ruta del Dashboard: Muestra la interfaz HTML con los filtros."""
    global df_original
    if df_original.empty:
        return "<h1>Error 500: No se pudieron cargar los datos del CSV. Revise el archivo.</h1>", 500

    opciones_comuna = sorted(df_original['Comuna'].unique().tolist())
    opciones_sexo = sorted(df_original['Sexo (Desc)'].unique().tolist())
    opciones_grupo = sorted(df_original['Grupo'].unique().tolist())
    
    opciones_amputacion = COLUMNAS_AMPUTACION
    opciones_edad_operadores = ['<', '<=', '>', '>=', '=']
    
    return render_template(
        'index.html', 
        opciones_comuna=opciones_comuna, 
        opciones_sexo=opciones_sexo,
        opciones_grupo=opciones_grupo,
        opciones_amputacion=opciones_amputacion,
        opciones_edad_operadores=opciones_edad_operadores
    )


@app.route('/get_filtered_map')
def get_filtered_map():
    # ... (código de filtrado, no modificado) ...
    global df_original
    df_filtrado = df_original.copy()
    
    # 1. FILTROS BÁSICOS
    comunas_seleccionadas = request.args.getlist('comuna')
    if comunas_seleccionadas:
        df_filtrado = df_filtrado[df_filtrado['Comuna'].isin(comunas_seleccionadas)]

    sexos_seleccionados = request.args.getlist('sexo')
    if sexos_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['Sexo (Desc)'].isin(sexos_seleccionados)]

    grupos_seleccionados = request.args.getlist('grupo')
    if grupos_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['Grupo'].isin(grupos_seleccionados)]

    total_amp_str = request.args.get('total_amputaciones')
    if total_amp_str and total_amp_str.isdigit():
        total_amp_val = int(total_amp_str)
        df_filtrado = df_filtrado[df_filtrado['Total_Amputaciones'] == total_amp_val]
        
    codigo_str = request.args.get('codigo_ppd') 
    if codigo_str:
        df_filtrado = df_filtrado[
            df_filtrado['Codigo'].astype(str).str.contains(codigo_str.strip(), case=False, na=False)
        ]
        
    edad_valor_str = request.args.get('edad_valor')
    edad_operador = request.args.get('edad_operador') 
    
    columna_edad = 'Ultima Edad Registrada'
    
    if edad_valor_str and edad_operador:
        try:
            edad_valor = float(edad_valor_str) 
            
            if edad_operador == '<':
                df_filtrado = df_filtrado[df_filtrado[columna_edad] < edad_valor]
            elif edad_operador == '<=':
                df_filtrado = df_filtrado[df_filtrado[columna_edad] <= edad_valor]
            elif edad_operador == '>':
                df_filtrado = df_filtrado[df_filtrado[columna_edad] > edad_valor]
            elif edad_operador == '>=':
                df_filtrado = df_filtrado[df_filtrado[columna_edad] >= edad_valor]
            elif edad_operador == '=':
                df_filtrado = df_filtrado[np.isclose(df_filtrado[columna_edad], edad_valor)] 
                
        except ValueError:
            pass

    # 3. FILTRO POR TIPO DE AMPUTACIÓN
    amputaciones_seleccionadas = request.args.getlist('amputacion')
    amputacion_modo = request.args.get('amputacion_modo', 'OR')

    if amputaciones_seleccionadas:
        columnas_validas = [col for col in amputaciones_seleccionadas if col in COLUMNAS_AMPUTACION]
        
        if columnas_validas:
            if amputacion_modo == 'OR':
                condition = df_filtrado[columnas_validas].gt(0).any(axis=1) 
                df_filtrado = df_filtrado[condition]
            
            elif amputacion_modo == 'AND':
                condition = df_filtrado[columnas_validas].gt(0).all(axis=1)
                df_filtrado = df_filtrado[condition]

    
    map_html = generate_map_html(df_filtrado, HPM_LOCATION)
    return map_html

# --- 4. RUTA PARA SUGERENCIAS (MODIFICADA) ---

@app.route('/enviar_sugerencia', methods=['POST'])
def enviar_sugerencia():
    """
    Recibe la sugerencia y redirige a la página de origen, 
    usando el campo 'next_route' del formulario.
    """
    
    sugerencia = request.form.get('sugerencia_texto')
    # NUEVO: Obtener la ruta de destino (dashboard o mostrar_inicio_y_diagrama)
    next_route = request.form.get('next_route')
    
    # Validamos la ruta de destino (por seguridad y control)
    valid_routes = ['mostrar_inicio_y_diagrama', 'dashboard']
    # Si la ruta no es válida, usamos el dashboard como destino por defecto
    target_route = next_route if next_route in valid_routes else 'dashboard'
    
    if sugerencia and sugerencia.strip(): 
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open('sugerencias.txt', 'a', encoding='utf-8') as f:
                f.write(f"--- Sugerencia recibida el: {timestamp} ---\n")
                f.write(sugerencia + "\n")
                f.write("="*50 + "\n\n")
            
            flash('¡Sugerencia enviada correctamente!', 'success')
        
        except Exception as e:
            app.logger.error(f"Error al guardar sugerencia: {e}")
            flash('Error al guardar la sugerencia. Inténtelo más tarde.', 'error')
    
    else:
        flash('La sugerencia no puede estar vacía.', 'warning')
    
    # Redirigir al usuario de vuelta a la página de origen (Inicio o Dashboard)
    return redirect(url_for(target_route)) 
# ---------------------------------------------


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)