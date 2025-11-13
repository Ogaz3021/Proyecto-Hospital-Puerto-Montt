# --- IMPORTS NECESARIOS ---
from flask import Flask, render_template, request
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import Fullscreen, MarkerCluster
from branca.element import Element
import os
import logging

# Configuración de logging (Para ver errores en la terminal)
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---

# Rutas a los archivos
GEOMETRY_PATH = 'llanquihue_comunas.geojson'
DATA_PATH = 'BASE FINAL CON CLUSTER.csv'

df_original = pd.DataFrame()
gdf_comunas = None

# Colores de Severidad
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
    
    return df

def load_shapes(geojson_path):
    global app
    try:
        gdf_llanquihue = gpd.read_file(geojson_path)
        gdf_llanquihue = gdf_llanquihue.to_crs(epsg=4326)
        
        # Renombrar si la columna de la comuna se llama 'COMUNA' en el GeoJSON
        if 'COMUNA' in gdf_llanquihue.columns:
            gdf_llanquihue = gdf_llanquihue.rename(columns={'COMUNA': 'Comuna_Corregida'})

        return gdf_llanquihue
    except Exception as e:
        app.logger.error(f"Error al cargar el GeoJSON ('{geojson_path}'): {e}")
        return None

# --- EJECUCIÓN DE CARGA GLOBAL ---
# Estas líneas se ejecutan al iniciar el script y cargan los datos en variables globales.
df_original = load_data(DATA_PATH)
gdf_comunas = load_shapes(GEOMETRY_PATH)
# -----------------------------------


# --- 2. FUNCIONES DEL BACKEND ---

def generate_map_html(df_filtrado):
    """Genera el mapa de Folium con los datos ya filtrados.
    
    *** MODIFICACIÓN: Los marcadores se añaden directamente al mapa (no agrupados). ***
    """
    global df_original, gdf_comunas
    
    # Manejar caso de DataFrame filtrado vacío
    if df_filtrado.empty:
        # Usa df_original para centrar el mapa si es posible
        lat_center = df_original['lat'].mean() if not df_original.empty and 'lat' in df_original.columns else -41.3
        lng_center = df_original['lng'].mean() if not df_original.empty and 'lng' in df_original.columns else -73.0
        
        mapa = folium.Map(location=[lat_center, lng_center], zoom_start=9)
        folium.Marker(location=[lat_center, lng_center], popup='No hay datos filtrados').add_to(mapa)
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
            tooltip=folium.GeoJsonTooltip(fields=['Comuna_Corregida'], aliases=['Comuna:']),
        ).add_to(mapa)

    # El MarkerCluster se omite para mostrar puntos individuales.

    for _, row in df_filtrado.iterrows():
        # *** POPUP COMPLETO (Se mantiene con nombres renombrados) ***
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
        
        # AÑADIR DIRECTAMENTE AL MAPA: Esto asegura que el punto NO se agrupe.
        marcador.add_to(mapa) 

    return mapa._repr_html_()


# --- 3. RUTAS DE FLASK ---

@app.route('/')
def index():
    """Ruta principal: Muestra la interfaz HTML con los filtros."""
    global df_original
    if df_original.empty:
        # Se muestra un error si el CSV no se pudo cargar
        return "<h1>Error 500: No se pudieron cargar los datos del CSV. Revise el archivo.</h1>", 500

    # Uso de nombres renombrados para obtener opciones de filtro
    opciones_comuna = sorted(df_original['Comuna'].unique().tolist())
    opciones_sexo = sorted(df_original['Sexo (Desc)'].unique().tolist())
    opciones_grupo = sorted(df_original['Grupo'].unique().tolist())
    
    opciones_amputacion = COLUMNAS_AMPUTACION
    
    return render_template(
        'index.html', 
        opciones_comuna=opciones_comuna, 
        opciones_sexo=opciones_sexo,
        opciones_grupo=opciones_grupo,
        opciones_amputacion=opciones_amputacion
    )


@app.route('/get_filtered_map')
def get_filtered_map():
    """Ruta API: Recibe parámetros de filtro y devuelve el mapa HTML filtrado."""
    global df_original
    df_filtrado = df_original.copy()
    
    # FILTRO POR COMUNA
    comunas_seleccionadas = request.args.getlist('comuna')
    if comunas_seleccionadas:
        df_filtrado = df_filtrado[df_filtrado['Comuna'].isin(comunas_seleccionadas)]

    # FILTRO POR SEXO
    sexos_seleccionados = request.args.getlist('sexo')
    if sexos_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['Sexo (Desc)'].isin(sexos_seleccionados)]

    # FILTRO POR GRUPO
    grupos_seleccionados = request.args.getlist('grupo')
    if grupos_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['Grupo'].isin(grupos_seleccionados)]

    # FILTRO POR TOTAL AMPUTACIONES
    total_amp_str = request.args.get('total_amputaciones')
    if total_amp_str and total_amp_str.isdigit():
        total_amp_val = int(total_amp_str)
        df_filtrado = df_filtrado[df_filtrado['Total_Amputaciones'] == total_amp_val]


    # FILTRO POR TIPO DE AMPUTACIÓN
    amputaciones_seleccionadas = request.args.getlist('amputacion')
    if amputaciones_seleccionadas:
        condition = df_filtrado[amputaciones_seleccionadas].any(axis=1)
        df_filtrado = df_filtrado[condition]

    map_html = generate_map_html(df_filtrado)
    return map_html


if __name__ == '__main__':
    # Línea activada que mantiene el servidor Flask corriendo.
    app.run(host='0.0.0.0', port=5000, debug=True)
