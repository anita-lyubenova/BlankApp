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

def show_map():
    folium_static(st.session_state.map)

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
                                "Present": "No",
                                "Name of nearest": None,
                                "Distance to nearest (m)": None})
                continue
        
            # map each POI geometry to nearest node
            filtered = filtered.copy()
           
            filtered["node"] = filtered.geometry.apply(lambda geom: ox.nearest_nodes(G, geom.x, geom.y))
            # compute walk distance for each
            filtered["walk_dist_m"] = filtered["node"].apply( lambda target_node: nx.shortest_path_length(G, home_node, target_node, weight='length'))
            
            # pick nearest by walking
            nearest = filtered.to_crs(epsg=4326).loc[filtered["walk_dist_m"].idxmin()]
            
            results.append({"Point of interest": cat,
                            "Present": "Yes",
                            "Name of nearest": nearest["name"],
                            "Distance to nearest (m)": round(nearest["walk_dist_m"])
                           })
        results_df=pd.DataFrame(results)
        return results_df
    
@st.dialog("Please, wait...",dismissible=False, on_dismiss="ignore")
def progress_dialog():
        
    #Base map ---------------------------------------------------------------------
    with st.spinner("Get elevation data...", show_time=True):
        
        st.session_state.location = geocode_address(st.session_state.address)
        
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
    st.write("✅ Get elevation data")
    
    
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
    st.session_state.location = None
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
    st.session_state.map = st.session_state.map = folium.Map(location=(59.33, 18.06), zoom_start=11)
if 'piechart' not in st.session_state:
    st.session_state.piechart = None
if 'poi_data' not in st.session_state:
    st.session_state.poi_data = None
if 'poi_tags' not in st.session_state:
    st.session_state.poi_tags = None
if 'nearest_poi' not in st.session_state:
    st.session_state.nearest_poi = None
                 
    
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
    col_address, col_features = cont_input.columns(spec= [0.3, 0.7], gap="small", border=True)
    with col_address:
        st.text_input("Enter an address (or choose on the map below):", value ="Skaldevägen 60", key="address")
        st.slider('Radius of interest (m)', min_value=100, max_value=2000, value=500, key="POI_radius")
        
        st.html("""
                <p style='font-size: 0.8rem; color: #555;'>
                    <em>*Larger radius and busy areas will take longer loading times as there is more data to process.</em>
                </p>
                """)
                
        st.html("""
                <p style='font-size: 0.8rem; color: #555;'>
                    <em>*Note that the results depend on the quality of OpenStreetMap data, which may contain errors.</em>
                </p>
                """)
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
       show_map()
       if st.session_state.nearest_poi is not None and not st.session_state.nearest_poi.empty:
           st.dataframe(st.session_state.nearest_poi)
       
    with col2:
        st.subheader("Land use distribution")
        st.write("Here you can see land use distribution across different categories. If you want to remove a category from the pie chart, click on it in the legend.")
        if st.session_state.location:
            st.plotly_chart(st.session_state.piechart,
                           use_container_width=True,
                           key="landuse_pie",
                           config = {'height': fig_height})
        else: 
            st.image("sample_piechart.png", caption="Example pie chart. Provide an address to replace.")
        
    
    # if selected_poi:
    #     st.session_state.poi_data
            # else:
            #     st.error("Address not found!")