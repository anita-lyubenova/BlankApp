import streamlit as st
import leafmap.foliumap as leafmap  # Uses ipyleaflet backend, not folium

st.title("Click map to get coordinates")

# Create map
m = leafmap.Map(center=[40, -3], zoom=5)

# Enable click capture
m.add_basemap("HYBRID")

# Store clicks in session state
if "clicked" not in st.session_state:
    st.session_state.clicked = None

def handle_map_click(**kwargs):
    lat = kwargs.get("lat")
    lon = kwargs.get("lng")
    st.session_state.clicked = (lat, lon)
    # Add marker
    m.add_marker(location=(lat, lon))

m.on_interaction(handle_map_click)

m.to_streamlit(height=600)

if st.session_state.clicked:
    st.write("Clicked coordinates:", st.session_state.clicked)