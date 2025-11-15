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

@st.cache_data(show_spinner=True, show_time = True)
def geocode_address(address):
    # geolocator = Nominatim(user_agent="Navigator")
    # return geolocator.geocode(address)
    results = geocoder.geocode(address)
    lat = results[0]['geometry']['lat']
    lon = results[0]['geometry']['lng']
    return lat, lon

@st.cache_data(show_spinner=True, show_time = True)
def get_osm_features(lat, lon, tags, dist):
    return ox.features_from_point((lat, lon), tags=tags, dist=dist)

@st.cache_data
def load_pie_index(sheet):
    df = pd.read_excel("OSM features.xls", sheet_name=sheet)
    df = df.dropna(subset=["key", "value"])
    df["key"] = df["key"].astype(str).str.strip()
    df["value"] = df["value"].astype(str).str.strip()
    return df



def clip_to_circle(gdf, lat, lon, radius):
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    proj_crs = gdf.estimate_utm_crs()
    gdf_proj = gdf.to_crs(proj_crs)
    center = Point(lon, lat)
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

st.title("Relocation Navigator")





st.markdown(
    """
    <style>
    /* No padding */
    div[data-testid = 'stMainBlockContainer']{padding: 0rem 0rem 0rem 1rem;} 
    
   
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
    /* Add left margin to the tabs */
    div[data-baseweb="tab-list"] {margin-left: 50px !important;}
    
    /* Add left padding to the tab content panels */
        div[data-baseweb="tab-panel"] { padding-left: 50px !important;}
    
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
        address = st.text_input("Enter an address:", value ="Skaldevägen 60")
        POI_radius=st.slider('Show PoIs within:', min_value=100, max_value=2000, value=500)
        
        no_landuse_input = st.checkbox("Show land use distribution (might take more time)", value =True)
    
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
        poi_tags=ms_index[ms_index['Multiselect'].isin(selected_poi)][["key", "value"]].groupby("key")["value"].apply(list).to_dict()
    
    #Built environment: get POIs within 500m
    tags0 = {
        'landuse': True,   # True → all landuse values
        'natural': True,   # all natural features
        'leisure': True,    # all leisure features
        'amenity':True,
       # 'shop':True,
        'building': True,
    }
    
    
    # If user enters an address => find latitude and longitude
    if st.button("Go!"):
        
        if address:
            
            location = geocode_address(address)
    
            if location:
                lat, lon = location.latitude, location.longitude
                st.write(f"Coordinates: {lat}, {lon}")
                
                #Map --------------------------------------------------------------
                
                m = folium.Map(location=[lat, lon], zoom_start=14)         
                # Add address marker
                folium.Marker([lat, lon], popup=address, icon=folium.Icon(color='red', icon='home')).add_to(m)
                folium.Circle(
                    location=[lat, lon],
                    radius=POI_radius,  # in meters
                    color='black',       
                    fill=False,
                    weight=2.5            
                    ).add_to(m)
                 
                if no_landuse_input:
                    all_features = get_osm_features(lat, lon, tags0, POI_radius)
                    #transform to long format
                    all_features=melt_tags(all_features, tags0.keys())
                    
                    # Pie chart data ------------------------------------------------------------------------------------------------------
                    # get only polygons
                    polygon_features = all_features[all_features.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                    
                    clipped=clip_to_circle(gdf=polygon_features, lat=lat, lon=lon, radius=POI_radius)
                    
                    #Copmute square meter area per key and value-----------------------
                    # Project to metric CRS for accurate areas
                    proj_crs = clipped.estimate_utm_crs()
                    clipped = clipped.to_crs(proj_crs)
                    clipped["area_m2"] = clipped.geometry.area
                    
                    pie_data0 = clipped.merge(pie_index, on=["key", "value"], how="left")
                    #pie_data0['pie_cat'] =pie_data0['pie_cat'].fillna('other')
                    pie_data0 =pie_data0[pie_data0['pie_cat'].notna()] #remove polygons that are not in the pie index
                    
                    pie_data = pie_data0.groupby(["pie_cat"]).agg(
                        total_area_m2 = ("area_m2", "sum"),
                        values_included=("value", lambda x: ", ".join(sorted(x.unique())))).reset_index() #concantenate all values within the pie_category
        
                    pie_data["values_included"] = (pie_data["values_included"].str.replace("_", " ")) #remove underscores from the column (for the popup)
                    
                    #pie chart----------------------------------------------------
                    fig = px.pie(
                        pie_data,
                        names="pie_cat",
                        values="total_area_m2",
                        hover_data=["values_included"],
                        color='pie_cat',
                        color_discrete_map=color_lookup,
                        hole=.5)
                    fig.update_traces(
                        textinfo="percent+label",
                        pull=[0.05]*len(pie_data),
                        hovertemplate="<b>%{label}</b><br>%{value:,.0f} m²<br>%{customdata}")
                    
                    fig.update_layout(height=fig_height)
                    
                    ##Land use layer
                    keys = pie_data.sort_values("total_area_m2", ascending=False)["pie_cat"].unique()
                
                    landuse_layer = folium.FeatureGroup(name="Land use distribution")
                    
                    folium.GeoJson(
                        data=pie_data0,  # All data at once
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
                    
                    landuse_layer.add_to(m)
                
                # Elevation data ----------------------------------------------------------------------------------------
                # 1. Get the street network (nodes + edges)
                G = ox.graph_from_point((lat, lon), dist=POI_radius, network_type='walk')
                
                # 2. Extract nodes only
                nodes, edges = ox.graph_to_gdfs(G)
                
                
                # # 3. Prepare node coordinates
                coords = list(zip(nodes.y, nodes.x))
                batch_size = 100  # OpenTopoData can only take limited locations per request
                batches = [coords[i:i+batch_size] for i in range(0, len(coords), batch_size)]
                elevations = []
                
                progress_text = "Fetching elevation data..."
                progress_elevation= st.progress(0, text=progress_text)
                total_batches = (len(coords) + batch_size - 1) // batch_size  # ceil division
                
                # 4. Query the OpenTopoData API in batches
                for i in range(0, len(coords), batch_size):
                    batch = coords[i:i+batch_size]
                    locations = "|".join([f"{lat},{lon}" for lat, lon in batch])
                    url = f"https://api.opentopodata.org/v1/srtm90m?locations={locations}"
                    r = requests.get(url)
                    if r.status_code == 200:
                        results = r.json().get('results', [])
                        elevations.extend([r.get('elevation', None) for r in results])
                    else:
                        elevations.extend([None]*len(batch))
                    time.sleep(1)  # avoid rate limit
                    progress_pct = int((i+1)/total_batches ) #int(((i+1)/total_batches) * 100)
                    progress_elevation.progress(progress_pct, text=f"{progress_text} ({progress_pct}%)")
                    
                progress_elevation.empty()
                
                
                # 5. Add node elevations
                nodes["elevation"] = elevations
                
                # Replace None or NaN with median (fallback)
                nodes["elevation"] = pd.to_numeric(nodes["elevation"], errors="coerce")
                median_elev = nodes["elevation"].median()
                nodes["elevation"].fillna(median_elev, inplace=True)
                
                # 5. Push node elevations back to the graph
                for node_id, elev in zip(nodes.index, nodes["elevation"]):
                    G.nodes[node_id]["elevation"] = elev
                
                # 6. Compute edge grades (uses node elevations)
                G = ox.add_edge_grades(G, add_absolute=True)
                edges = ox.graph_to_gdfs(G, nodes=False)
                grades = edges['grade_abs'].dropna()  # remove any NaN 
    
                #m_elev = folium.Map(location=[lat, lon], zoom_start=14) 
                elevation_layer = folium.FeatureGroup(name="Street steepness")
                
                max_grade = 0.15 #edges['grade_abs'].max()
                colormap = cm.LinearColormap(["yellow","orange",'red', 'purple', 'blue'], vmin=0, vmax=max_grade)
                colormap.caption = 'Street Grade (%)'
                
                # 10. Add edges as polylines with color based on grade
                for _, row in edges.iterrows():
                    coords = [(y, x) for x, y in row.geometry.coords]
                    color = colormap(row['grade_abs'])
                    folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(elevation_layer)
                
                # 11. Add the color scale
                colormap.add_to(m)
                elevation_layer.add_to(m)
                
                if selected_poi:
                    ms_poi = get_osm_features(lat, lon, poi_tags, POI_radius)
                    poi_data = melt_tags(ms_poi, poi_tags.keys()).reset_index().merge(ms_poi.reset_index()[["id", "name"]], on="id").merge(ms_index[["Category", "Multiselect", "key", "value", "color", "icon"]], on=["key", "value"])
                    poi_data.loc[poi_data['name'].isna(), 'name']="Unnamed"
                     
                    #add layer with PoI markers to map                              
                    poi_layer = folium.FeatureGroup(name="Points of Interest")
                    
                    for idx, row in poi_data.iterrows():
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
                             
                    poi_layer.add_to(m)
                    
                    #Available PoI: ---------------------------------------------------------------------------------
                    G = ox.graph_from_point((lat, lon), dist=POI_radius, network_type='walk')
                    home_node = ox.nearest_nodes(G, lon, lat)
                    
                    #change crs to compute centroids of the polygons
                    p3857 = poi_data.to_crs(epsg=3857) 
                    p3857['centroide'] = p3857.geometry.centroid
                    p3857=p3857.set_geometry("centroide")
    
                    p4326=p3857.to_crs(epsg=4326)
    
                    results = []
                    
                    for cat in selected_poi:
                       
                        filtered = p4326[p4326["Multiselect"] == cat]
                        if filtered.empty:
                            results.append({"Category": cat, "Present": "No", "Name of nearest": None, "Distance to nearest (m)": None})
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
                    resdf=pd.DataFrame(results)
                    
                
               
                
                
                
                folium.LayerControl().add_to(m)
                
                col1,col2 = st.columns(2, gap="small", border=True)    
                
                with col1:
                    # st.subheader("Map")
                    # st.write("Here you can see land use patterns, elevation profile and where your points of interest are located")
                    st_folium(m,use_container_width=True)
                    
                    with st.popover("Degree reference values"):
                        st.markdown("""
                            - **0–2%**: Very flat street, easy to walk or bike  
                            - **2–5%**: Slight incline, barely noticeable  
                            - **5–8%**: Moderate slope, noticeable uphill effort  
                            - **8–12%**: Steep street, challenging for bikes or long walks  
                            - **>12%**: Very steep, strenuous; may be difficult for vehicles, bicycles, or accessibility
                            """)
                    st.header("Nearest points of interest")
                            
                    if 'resdf' in locals() and not resdf.empty:
                        st.dataframe(resdf, key="nearest_pois")
                    else:
                        st.info("No Points of interest selected.")
                    
                    
                with col2:
                    st.subheader("Land use distribution")
                    
                    if 'fig' in locals():
                        st.plotly_chart(fig,
                                        use_container_width=True,
                                        key="landuse_pie",
                                        config = {'height': fig_height})
                    else:
                        st.info("No landuse characteristics were obtained. If you want to see the landuse distribution, mark the checkbox")
                   
                    
               
                
                    
    
    
            else:
                st.error("Address not found!")