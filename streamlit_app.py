import streamlit as st
import folium
from streamlit_folium import st_folium
#from geopy.geocoders import Nominatim
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import plotly.express as px
import networkx as nx
import xlrd
from streamlit_folium import folium_static

#from folium import GeoJson, GeoJsonTooltip

import requests
import time
import branca.colormap as cm# 8. Create a linear color scale for grade_abs
from opencage.geocoder import OpenCageGeocode
from folium.plugins import Draw
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


# Load OpenCage API key from Streamlit secrets
OPENCAGE_KEY = st.secrets["opencage"]["api_key"]

# Initialize the geocoder
geocoder = OpenCageGeocode(OPENCAGE_KEY)

#geolocator = Nominatim(user_agent="Navigator")

@st.cache_data(show_spinner=False, show_time = False)
def geocode_address(address):
    # geolocator = Nominatim(user_agent="Navigator")
    # return geolocator.geocode(address)
    results = geocoder.geocode(address)
    lat = results[0]['geometry']['lat']
    lon = results[0]['geometry']['lng']
    return lat, lon

#@st.cache_data(show_spinner=True, show_time = True)
# def get_osm_features(location, tags, dist):
#     return ox.features_from_point(location, tags=tags, dist=dist)

@st.cache_data(show_spinner=False, show_time = False)
def load_pie_index(sheet):
    df = pd.read_excel("OSM features.xls", sheet_name=sheet)
    df = df.dropna(subset=["key", "value"])
    df["key"] = df["key"].astype(str).str.strip()
    df["value"] = df["value"].astype(str).str.strip()
    return df



def clip_to_circle(gdf, location, radius):
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    proj_crs = gdf.estimate_utm_crs()
    gdf_proj = gdf.to_crs(proj_crs)
    center = Point(location[1], location[0])
    circle = gpd.GeoSeries([center], crs=4326).to_crs(proj_crs).buffer(radius)
    return gpd.clip(gdf_proj, circle).to_crs(4326)




def melt_tags(gdf, tag_keys):
    # keep onlt keys that exist in gdf
    tag_keys = [k for k in tag_keys if k in gdf.columns]
    if not tag_keys:
        raise ValueError("None of the provided tag_keys exist in the GeoDataFrame.")

    melted = (
        gdf[tag_keys]
        .stack()
        .reset_index()
        .rename(columns={"level_2": "key", 0: "value"})
    )
    melted = melted.merge(gdf.reset_index()[["id", "geometry"]], on="id")
    melted = melted.drop(columns="element")
    melted = gpd.GeoDataFrame(melted, geometry="geometry", crs=gdf.crs)
    return melted

@st.cache_data(show_spinner=False, show_time = False)
def process_elevations(location, radius):
    """
    Fetches elevation data and computes edge grades for a walking network around a given point.

    Args:
        location (tuple): (latitude, longitude) of the center point.
        radius (float): Distance in meters for fetching the street network.

    Returns:
        nodes (GeoDataFrame): Node data with elevation.
        edges (GeoDataFrame): Edge data with absolute grade.
    """
    
    # 1. Retrieve street network
    G = ox.graph_from_point(location, dist=radius, network_type='walk')
    
    # 2. Extract nodes
    nodes, _ = ox.graph_to_gdfs(G)
    
    # 3. Prepare node coordinates and batching
    coords = list(zip(nodes.y, nodes.x))
    batch_size = 100
    elevations = []
    #total_batches = (len(coords) + batch_size - 1) // batch_size

    # 4. Fetch elevations in batches
    for i in range(0, len(coords), batch_size):
        batch = coords[i:i+batch_size]
        locations = "|".join([f"{lat},{lon}" for lat, lon in batch])
        url = f"https://api.opentopodata.org/v1/srtm90m?locations={locations}"
        response = requests.get(url)
        
        if response.status_code == 200:
            results = response.json().get('results', [])
            elevations.extend([res.get('elevation', None) for res in results])
        else:
            elevations.extend([None] * len(batch))
        
        time.sleep(1)  # API rate limiting
    
    # 5. Append elevation data
    nodes['elevation'] = pd.to_numeric(elevations, errors='coerce')
    median_elevation = nodes['elevation'].median()
    nodes['elevation'].fillna(median_elevation, inplace=True)

    # 6. Push elevations back to graph
    for node_id, elev in zip(nodes.index, nodes['elevation']):
        G.nodes[node_id]['elevation'] = elev

    # 7. Compute edge grades
    # G = ox.add_edge_grades(G, add_absolute=True)
    # _, edges = ox.graph_to_gdfs(G)
    
    G = ox.add_edge_grades(G, add_absolute=True)
    edges = ox.graph_to_gdfs(G, nodes=False)
    #grades = edges['grade_abs'].dropna()

    return nodes, edges

# def show_map():
#     #folium_static(st.session_state.map)
#     #map_data = st_folium(st.session_state.map, key="main_map", height=600, returned_objects=["last_clicked"])
#     #return map_data
#     st_folium(st.session_state.map, key="main_map", height=600, returned_objects=["last_clicked"])

# def click_button():
#     st.session_state.clicked = True

@st.cache_data(show_spinner=False, show_time = False) 
def get_landuse_data(location, radius, tags):

    #all_features = get_osm_features(lat, lon, tags0, POI_radius)
    all_features = ox.features_from_point(center_point=location, tags=tags, dist=radius)
    #transform to long format
    all_features=melt_tags(all_features, tags.keys())
    
    # Pie chart data ------------------------------------------------------------------------------------------------------
    # get only polygons
    polygon_features = all_features[all_features.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    
    clipped=clip_to_circle(gdf=polygon_features, location=location, radius=radius)
    
    #Copmute square meter area per key and value-----------------------
    # Project to metric CRS for accurate areas
    proj_crs = clipped.estimate_utm_crs()
    clipped = clipped.to_crs(proj_crs)
    clipped["area_m2"] = clipped.geometry.area
    
    pie_data0 = clipped.merge(pie_index, on=["key", "value"], how="left")
    #pie_data0['pie_cat'] =pie_data0['pie_cat'].fillna('other')
    pie_data0 =pie_data0[pie_data0['pie_cat'].notna()] #remove polygons that are not in the pie index
    
    #sum areas
   
    return pie_data0

#@st.cache_data cannot hash
def aggregate_landuse_data(landuse_data):
    pie_data = landuse_data.groupby(["pie_cat"]).agg(
        total_area_m2 = ("area_m2", "sum"),
        values_included=("value", lambda x: ", ".join(sorted(x.unique())))).reset_index() #concantenate all values within the pie_category
    
    pie_data["values_included"] = (pie_data["values_included"].str.replace("_", " ")) #remove underscores from the column (for the popup)
    
    return pie_data

@st.cache_data(show_spinner=False, show_time = False)
def get_POIs(location, radius, poi_tags):
    ms_poi = ox.features_from_point(center_point=location, tags=poi_tags, dist=radius)
    poi_data = melt_tags(ms_poi, poi_tags.keys()).reset_index().merge(ms_poi.reset_index()[["id", "name"]], on="id").merge(ms_index[["Category", "Multiselect", "key", "value", "color", "icon"]], on=["key", "value"])
    poi_data.loc[poi_data['name'].isna(), 'name']="Unnamed"
    poi_data = poi_data.drop_duplicates(subset=["name", "Multiselect"])
    
    return poi_data

#@st.cache_data cannot hash
def available_POIs(location, radius, poi_data):
        G = ox.graph_from_point(location, dist=radius, network_type='walk')
        home_node = ox.nearest_nodes(G, X=location[1], Y=location[0])
        
        #change crs to compute centroids of the polygons
        p3857 = poi_data.to_crs(epsg=3857) 
        p3857['centroide'] = p3857.geometry.centroid
        p3857=p3857.set_geometry("centroide")

        p4326=p3857.to_crs(epsg=4326)

        results = []
        
        for cat in selected_poi:
           
            filtered = p4326[p4326["Multiselect"] == cat]
            if filtered.empty:
                results.append({"Point of interest": cat,
                                "Present": "❌",
                                "Name of nearest": None,
                                "Walking distance (m)": None})
                continue
        
            # map each POI geometry to nearest node
            filtered = filtered.copy()
           
            filtered["node"] = filtered.geometry.apply(lambda geom: ox.nearest_nodes(G, geom.x, geom.y))
            # compute walk distance for each
            filtered["walk_dist_m"] = filtered["node"].apply( lambda target_node: nx.shortest_path_length(G, home_node, target_node, weight='length'))
            
            # pick nearest by walking
            nearest = filtered.to_crs(epsg=4326).loc[filtered["walk_dist_m"].idxmin()]
            
            results.append({"Point of interest": cat,
                            "Present": "✅",
                            "Name of nearest": nearest["name"],
                            "Walking distance (m)": round(nearest["walk_dist_m"])
                           })
        results_df=pd.DataFrame(results)
        return results_df

def process_topography(nodes):
    # Extract point coordinates and elevation
    points = nodes[['x', 'y']].values
    values = nodes['elevation'].values
    
    # Bounding box of your node data
    lng_min, lng_max = nodes['x'].min(), nodes['x'].max()
    lat_min, lat_max = nodes['y'].min(), nodes['y'].max()
    
    grid_x, grid_y = np.mgrid[lng_min:lng_max:400j,lat_min:lat_max:400j]

    # Cubic interpolation
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    
    # Fill NaN gaps with nearest-neighbor interpolation
    grid_z_nn = griddata(points, values, (grid_x, grid_y), method='nearest')
    grid_z = np.where(np.isnan(grid_z), grid_z_nn, grid_z)

    n_levels = 12
    levels = np.linspace(values.min(), values.max(), n_levels)
    
    fig, ax = plt.subplots(figsize=(6,6))
    cs = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='Blues')
    
    polygons = []
    
    # cs.allsegs is a list of lists containing polygon coordinates for each level
    for level_idx, seglist in enumerate(cs.allsegs):
        elev_value = levels[level_idx]
    
        for seg in seglist:
            try:
                poly = Polygon(seg)
                if poly.is_valid and poly.area > 0:
                    polygons.append({
                        "geometry": poly,
                        "elev": float(elev_value)
                    })
            except:
                pass

    gdf = gpd.GeoDataFrame(polygons, crs="EPSG:4326")
    
    

    return gdf

@st.dialog("Please, wait...",dismissible=False, on_dismiss="ignore")
def progress_dialog():
        
    #Base map ---------------------------------------------------------------------
    with st.spinner("Get elevation data...", show_time=True):
        
        #st.session_state.location = geocode_address(st.session_state.address)
        
        st.session_state.map = folium.Map(location=st.session_state.location, zoom_start=14)         
        # Add address marker
        folium.Marker(st.session_state.location, popup=st.session_state.address, icon=folium.Icon(color='red', icon='home')).add_to(st.session_state.map)
        folium.Circle(
            location=st.session_state.location,
            radius=st.session_state.POI_radius,  # in meters
            color='black',       
            fill=False,
            weight=2.5            
            ).add_to(st.session_state.map)
         
    
        # Elevation map layer --------------------------------------------------------
       
        
        st.session_state.nodes, st.session_state.edges = process_elevations(st.session_state.location, st.session_state.POI_radius)

    st.write("✅ Get elevation data")
    
    with st.spinner("Process and plot elevation data...", show_time=True):
        elevation_layer = folium.FeatureGroup(name="Street steepness")
        
        colormap = cm.LinearColormap(["yellow","orange",'red', 'purple', 'blue'], vmin=0, vmax=0.15)
        colormap.caption = 'Street steepness'
        
        #Add edges as polylines with color based on grade
        for _, row in st.session_state.edges.iterrows():
            coords = [(y, x) for x, y in row.geometry.coords]
            color = colormap(row['grade_abs'])
            folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(elevation_layer)
        
       
        colormap.add_to(st.session_state.map)
        elevation_layer.add_to(st.session_state.map)
        
        # Topography layer----------------------------------------------------
        st.session_state.topography_gdf = process_topography(st.session_state.nodes)
        
        topography_layer = folium.FeatureGroup(name="Elevation")
        colormap_tpg = cm.linear.Blues_09.scale(st.session_state.nodes['elevation'].values.min(), st.session_state.nodes['elevation'].values.max())
        
        for _, row in st.session_state.topography_gdf.iterrows():
            color = colormap_tpg(row['elev'])
        
            folium.GeoJson(
                data=row['geometry'].__geo_interface__,
                style_function=lambda feature, color=color: {
                    "fillColor": color,
                    "color": color,
                    "weight": 0,
                    "fillOpacity": 0.3
                },
                tooltip=f"Elevation: {row['elev']:.1f} m" 
            ).add_to(topography_layer)
        
        topography_layer.add_to(st.session_state.map)
        colormap_tpg.add_to(st.session_state.map)
    st.write("✅ Process and plot elevation data")
    
    
     #pie chart------------------------------------------------------------------------------------
    with st.spinner("Get and process land use data...", show_time=True):
    
        
        st.session_state.landuse_data = get_landuse_data(location = st.session_state.location,
                                                         radius = st.session_state.POI_radius,
                                                         tags = tags0)
      
        st.session_state.piechart = px.pie(
            aggregate_landuse_data(st.session_state.landuse_data),
            names="pie_cat",
            values="total_area_m2",
            hover_data=["values_included"],
            color='pie_cat',
            color_discrete_map=color_lookup,
            hole=.5)
        st.session_state.piechart.update_traces(
            textinfo="percent+label",
            pull=[0.05]*len(aggregate_landuse_data(st.session_state.landuse_data)),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} m²<br>%{customdata}")
        
        
    
    
        #Land use map layed --------------------------------------------------------------
        landuse_layer = folium.FeatureGroup(name="Land use distribution")
            
        folium.GeoJson(
            data=st.session_state.landuse_data,  # All data at once
            style_function=lambda feature: {
                "fillColor": color_lookup.get(feature["properties"]["pie_cat"]),
                "color": "black",
                "weight": 0.3,
                "fillOpacity": 0.5,
            },
            popup=folium.GeoJsonPopup(
                fields=["pie_cat", "key", "value"],
                aliases=["In pie chart", "OSM key", "OSM value"]
            )
        ).add_to(landuse_layer)
        
        landuse_layer.add_to(st.session_state.map)
    
    st.write("✅ Get land use data")

    # POI map layer ----------------------------------------------------------------------------------------
    if selected_poi:
        with st.spinner("Get points of interest...", show_time=True):
            
            st.session_state.poi_data = get_POIs(location = st.session_state.location,
                                                 radius = st.session_state.POI_radius,
                                                 poi_tags = st.session_state.poi_tags)
            poi_layer = folium.FeatureGroup(name="Points of Interest")
                
            for idx, row in st.session_state.poi_data.iterrows():
                lon_, lat_ = row.geometry.centroid.xy
                folium.Marker(
                    location=[lat_[0], lon_[0]],
                    popup= f"<div style='font-size:12px; font-family:Arial; white-space:nowrap;'><b>{row.get('Category','N/A').capitalize()}: </b>{row.get('Multiselect')}<br>{row.get('name', 'Unnamed')}",
                    icon=folium.Icon(
                        color=row['color'],
                        icon=row['icon'].replace("fa-", "") if str(row['icon']).startswith("fa-") else row['icon'],
                        prefix="fa" if str(row['icon']).startswith("fa-") else None
                    )
                    
                    ).add_to(poi_layer)
                     
            poi_layer.add_to(st.session_state.map)
    
            #Available PoI: ---------------------------------------------------------------------------------
            st.session_state.nearest_poi =available_POIs(location = st.session_state.location,
                                                       radius = st.session_state.POI_radius,
                                                       poi_data = st.session_state.poi_data)    
        st.write("✅ Get points of interest")

    folium.LayerControl().add_to(st.session_state.map)
    
    st.rerun()    
    
# def show_pie_chart():
#     st.plotly_chart(st.session_state.piechart,
#                     use_container_width=True,
#                     key="landuse_pie",
#                     config = {'height': fig_height})
    
#get pie index 
pie_index = load_pie_index("pie_index")

#Create a color to category mapping
unique_cats = pie_index["pie_cat"].unique()
palette = px.colors.qualitative.Light24
color_lookup = {
    cat: palette[i % len(palette)]
    for i, cat in enumerate(sorted(unique_cats))
}

color_lookup.get(pie_index['pie_cat'][1], "gray")

ms_index  = load_pie_index("Multiselect")
ms_index = ms_index[ms_index['Multiselect'].notna()]

ms_cats = ms_index['Category'].unique()


fig_height=700
# -- Set page config
apptitle = 'Navigator'
st.set_page_config(page_title=apptitle,
                   layout="wide",
                   initial_sidebar_state="collapsed")

# st.title("Relocation Navigator")
# st.write("App started at:", time.time())

# Initialize session state variables if they don't exist
if "location" not in st.session_state:
    st.session_state.location = [59.33, 18.0656]
# if "map" not in st.session_state:
#     st.session_state.map = None
if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'nodes' not in st.session_state:
    st.session_state.nodes = None
if 'edges' not in st.session_state:
    st.session_state.edges = None
if 'landuse_data' not in st.session_state:
    st.session_state.landuse_data = None
if 'map' not in st.session_state:
    st.session_state.map = None # st.session_state.map = folium.Map(location=(59.33, 18.06), zoom_start=11)
if 'piechart' not in st.session_state:
    st.session_state.piechart = None
if 'poi_data' not in st.session_state:
    st.session_state.poi_data = None
if 'poi_tags' not in st.session_state:
    st.session_state.poi_tags = None
if 'nearest_poi' not in st.session_state:
    st.session_state.nearest_poi = None
if "topography_gdf" not in st.session_state:
    st.session_state.topography_gdf = None
if "topography_colormap" not in st.session_state:
    st.session_state.topography_colormap = None                 
    
#Built environment feautres for the pie chart
tags0 = {
    'landuse': True,   # True → all landuse values
    'natural': True,   # all natural features
    'leisure': True,    # all leisure features
    'amenity':True,
   # 'shop':True,
    'building': True,
}


 # /* No padding */
 # div[data-testid = 'stMainBlockContainer']{padding: 0rem 0rem 0rem 1rem;} 
   # /* Add left margin to the tabs */
   # /* div[data-baseweb="tab-list"] {margin-left: 50px !important;}*/
   
   # /* Add left padding to the tab content panels */
   # */    div[data-baseweb="tab-panel"] { padding-left: 50px !important;}*/
st.markdown(
    """
    <style>
    
    div[data-testid = 'stMainBlockContainer']{padding-left: 1rem;}
    

   
    div.stButtonGroup {
        display: flex !important;       /* set label to be on the same line as buttons */
        align-items: top;            /* vertical align label and pills */
        gap: 10px;                      /* space between label and buttons */
    }
    
    div.stButtonGroup label {
        white-space: nowrap !important; /* prevent label from breaking */
        flex-shrink: 0;                 /* don’t allow the label to shrink */
        margin-bottom: 0 !important;
    }
    
    div.stButtonGroup label div[data-testid='stMarkdownContainer'] p {
        font-weight: bold !important;
        margin: 0;  /* optional: remove default margin */
    }
     /* App title in header bar */
    header:before {
        content: 'Relocation Navigator';
        font-size: 28px;
        font-weight: 600;
        margin-right: 20px;
        margin-left: 20px;
        color: #2d2d2e;
        white-space: nowrap;
    }
    
    header {
    background-color: #c3cee3 !important;  /* your color */
    height: 3.5rem !important;             /* adjust height if needed */
}
    header .stAppToolbar {
    background-color: transparent !important;
}
    
    </style>
    """,
    unsafe_allow_html=True
)

intro_text = " Relocation Navigator helps you explore neighborhoods by providing and visualizing information about the local streets, land use, and nearby points of interest. Home seekers, cyclists and pedestrians can get an overview of an unknown neighborhood to inform relocation or travel decisions.Open Street Map data is used to visualize land use patterns and to find amenities like schools, public transport, shops, leisure spots, etc. The app shows walking distances to selected points of interest. Street steepness and shortest distance to key amenities can indicate level of accessibility. "
 
tab_intro, tab_map = st.tabs(["Introduction", "Explore"])

with tab_intro:
    #st.markdown(textwrap.fill(text=intro_text, width=50),unsafe_allow_html=True)
    st.markdown("""
    Relocation Navigator helps you explore neighborhoods by providing and visualizing information about the local \n\n
    streets, land use, and nearby points of interest. Home seekers, cyclists and pedestrians can get an overview \n\n
    of an unknown neighborhood to inform relocation or travel decisions.Open Street Map data is used to visualize \n\n
    land use patterns and to find amenities like schools, public transport, shops, leisure spots, etc. The app shows  \n\n
    walking distances to selected points of interest. Street steepness and shortest distance to key amenities can \n\n
    indicate level of accessibility.
    """)
    st.markdown("---")
    st.markdown("""
    **Features include:**
    - **Street Network & Elevation:** Visualize street slopes (grades) color-coded by steepness.  
    - **Land Use Distribution:** See the proportion of parks, nature, residential, commercial,idustrial and other land uses.  
    - **Points of Interest:** Search for nearby amenities such as schools, shops, leisure spots, or any other POIs.  
    - **Accessibility & Distance:** Find the nearest selected points of interest and walking distances.  

    **How to use:**
    1. Enter an address in the left panel.  
    2. Select the radius around the address to explore.  
    3. Choose which points of interest to include. (optional)  
    4. Check the "Show land use distribution" box if you want additional land use info.  
    5. Click **Go!** to generate the map, elevation layer, and POI information.  
    6. Wait for fetching, processing and visualizing the data. Depending on the location, and the size of the radius, this may take a while

    The map shows:
    - Streets colored by slope (steepness)  
    - Land use polygons corresponding to the pie chart of land use distribution  
    - Markers for selected points of interest  
    """)
    

# Main --------------------------------------------------------

with tab_map:
    cont_input = st.container()
    col_address, col_features = cont_input.columns(spec= [0.4, 0.6], gap="small", border=True)
    with col_address:
        st.text_input("Enter an address (or choose on the map below):", value ="Skaldevägen 60", key="address")
        st.slider('Radius of interest (m)', min_value=100, max_value=2000, value=500, key="POI_radius")
        
        #input map
        input_map = folium.Map(location=st.session_state.location, zoom_start=10)
        
        # Create feature group that will contain the marker
        editable = folium.FeatureGroup(name="Editable")
        input_map.add_child(editable)
        
        
        
        # Draw(
        #     feature_group=editable,
        #     draw_options={           # disable ALL drawing tools
        #         "polyline": False,
        #         "polygon": False,
        #         "circle": False,
        #         "rectangle": False,
        #         "circlemarker": False,
        #         "marker": False
        #     },
        #     edit_options={           # enable editing of existing layers
        #         "edit": True,
        #         "remove": False      # you can set True if deletion should be allowed
        #     }
        # ).add_to(input_map)
        
        
        
        if st.session_state.address is not None:
             st.session_state.location = geocode_address(st.session_state.address)
    
             
        # # Check if the user moved the marker => Update location
        # if selected["all_drawings"] is not None:  # there is at least one feature
        #     marker_coords = selected["all_drawings"][0]["geometry"]["coordinates"]
        #    st.session_state.location = [marker_coords[1],marker_coords[0]]
        # #else if session.state.address is TRUE => geocode address from that    
        # elif st.session_state.address is not None:
        #     st.session_state.location = geocode_address(st.session_state.address)
        # Add a marker to be dragged
        folium.Marker(
            st.session_state.location
           # tooltip="Move to the desired location by using the Edit tool to the left"
        ).add_to(editable)
        
        selected = st_folium(input_map,
                             key= "map_input",
                             height=300,
                             feature_group_to_add=editable
                            # returned_objects=["all_drawings"]
                             )
        st.write(f"Marker coordinates: {st.session_state.location}")
        
        
        
        # st.html("""
        #         <p style='font-size: 0.8rem; color: #555;'>
        #             <em>*Larger radius and busy areas will take longer loading times as there is more data to process.</em>
        #         </p>
        #         """)
                
        # st.html("""
        #         <p style='font-size: 0.8rem; color: #555;'>
        #             <em>*Note that the results rely on OpenStreetMap data, which may contain errors.</em>
        #         </p>
        #         """)
    with col_features:
        st.write("Select points of interest you'd like to have in the area")
        
        selected_poi = []
    
        # Loop through all categories
        for category in ms_index["Category"].unique():
            options = ms_index.loc[ms_index["Category"] == category, "Multiselect"].dropna().unique().tolist()
            
            # Dynamically generate a pills input for each category
            selected = st.pills(
                label=category,
                options=options,
                key=f"poi_{category.replace(' ', '_')}_input",  # unique key
                selection_mode="multi"
            )
            
            # Store the selected values
            selected_poi.extend(selected)
            
    if selected_poi:
        st.session_state.poi_tags=ms_index[ms_index['Multiselect'].isin(selected_poi)][["key", "value"]].groupby("key")["value"].apply(list).to_dict()
    
    
    # go_input = st.button("Go!", on_click=click_button)
    #st.write("Button value:", go_input)
    # If user enters an address => find latitude and longitude
    if st.button("Go!"): #st.session_state.clicked:
        if st.session_state.address:
            
            progress_dialog()
            #with st.status("Processing, please wait...", expanded=True) as status:
                
            

        else:
            st.error("Address not found!")
            

    #Outputs---------------------------------------------------------------------------------            
    col1,col2 = st.columns(2, gap="small", border=True)             
    col3,col4 = st.columns(2, gap="small", border=True)              
    
    with col1:
       st.subheader("Map")
       st.write("Given an address, here you can see land use patterns, street steepness, and where your points of interest are located. The area colors correspond to the pie chart categories.")
       #st.write(st.session_state.location)
       #st_folium(st.session_state.map, width=700, height=500)
       with st.popover("Steepness reference values"):
            st.markdown("""
                - **0–2%**: Very flat street, easy to walk or bike  
                - **2–5%**: Slight incline, barely noticeable  
                - **5–8%**: Moderate slope, noticeable uphill effort  
                - **8–12%**: Steep street, challenging for bikes or long walks  
                - **>12%**: Very steep, strenuous; may be difficult for vehicles and bicycles
            
                """)
       #st.session_state.folium_map
       #st.session_state.map_click= st_folium(st.session_state.map, key="main_map", height=600, returned_objects=["last_clicked"])
       if st.session_state.map is not None:
           folium_static(st.session_state.map)
       
       
    with col2:
        st.subheader("Land use distribution")
        st.write("Here you can see land use distribution across different categories. If you want to remove a category from the pie chart, click on it in the legend.")
        if st.session_state.piechart:
            st.plotly_chart(st.session_state.piechart,
                           use_container_width=True,
                           key="landuse_pie",
                           config = {'height': fig_height})
        else: 
            st.image("sample_piechart.png", caption="Example pie chart. Provide an address to replace.")
    with col3:
        st.subheader("Available points of interest")
        if st.session_state.nearest_poi is not None and not st.session_state.nearest_poi.empty:
            st.dataframe(st.session_state.nearest_poi)
    
    
    # if selected_poi:
    #     st.session_state.poi_data
            # else:
            #     st.error("Address not found!")