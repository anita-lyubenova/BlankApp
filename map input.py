

import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw


st.title("Map input")


input_map = folium.Map(location=[59.33, 18.0656], zoom_start=10)
# Create feature group that will contain the marker
editable = folium.FeatureGroup(name="Editable")
input_map.add_child(editable)

# Add a marker to be dragged
folium.Marker(
    [59.33, 18.0656],
    tooltip="Place on the desired location by using the Edit tool to the left"
).add_to(editable)

Draw(
    feature_group=editable,
    draw_options={           # disable ALL drawing tools
        "polyline": False,
        "polygon": False,
        "circle": False,
        "rectangle": False,
        "circlemarker": False,
        "marker": False
    },
    edit_options={           # enable editing of existing layers
        "edit": True,
        "remove": False      # you can set True if deletion should be allowed
    }
).add_to(input_map)

selected = st_folium(input_map, width=700, height=500, returned_objects=["all_drawings"])

# Check if the user moved the marker
if selected and "all_drawings" in selected:
    drawings = selected["all_drawings"]
    if drawings:  # there is at least one feature
        # For a single marker, get its coordinates
        marker_coords = drawings[0]["geometry"]["coordinates"]  # [lon, lat]
        st.write(f"Marker coordinates: {marker_coords}")