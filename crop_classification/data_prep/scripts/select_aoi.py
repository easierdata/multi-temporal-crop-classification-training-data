import json
from pathlib import Path
import sys
import geopandas as gpd
import folium
import folium.plugins

# used to add the src directory to the Python path, making
# it possible to import modules from that directory.
module_path = Path(__file__).parent.parent.parent.resolve().as_posix()
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass

# Default name for the HTML file
HTML_FILE_NAME = "Select-AOI.html"


def main() -> None:
    """
    Main function to create an interactive map for selecting areas of interest (AOI) from GeoJSON files.

    This function performs the following steps:
    1. Loads two GeoJSON files containing bounding box data.
    2. Creates a folium GeoJson layer from the loaded data.
    3. Converts the GeoJson layer to a GeoDataFrame for further processing.
    4. Fits the map to the bounds of the GeoDataFrame.
    5. Creates a folium map centered around the mean coordinates of the GeoDataFrame.
    6. Adds the GeoJson layer and various plugins (LayerControl, Draw, Fullscreen) to the map.
    7. Adds the Turf.js library to the map for spatial operations.
    8. Saves the map to an HTML file and reads the content.
    9. Adds custom JavaScript to handle polygon drawing and feature selection.
    10. Exports the selected features to a GeoJSON file upon drawing completion.
    Returns:
        None
    """
    # Load the json file
    with open(PREGEN_BB_CHIP_5070_PAYLOAD, "r") as f:
        chip_bbox_5070 = json.load(f)

    # Load the json file
    with open(PREGEN_BB_CHIP_PAYLOAD, "r") as f:
        chip_bbox = json.load(f)

    chips_bbox_layer = folium.GeoJson(
        data=chip_bbox,
        name="Bounding box chips",
        style_function=lambda feature: {
            "fillColor": "lightgreen",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5,
        },
    )
    # Create GPD from foluim layer so it can be used for intersection
    chips_bbox_gpd = gpd.GeoDataFrame.from_features(chip_bbox["features"])

    # Fit map to the bounds of the GeoDataFrame
    bounds = chips_bbox_gpd.total_bounds
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    # Create a folium map centered around the mean coordinates of your GeoDataFrame
    m = folium.Map([center_y, center_x], zoom_start=5)

    # Add the layer to the map
    chips_bbox_layer.add_to(m)

    m.add_child(folium.LayerControl())
    m.add_child(
        folium.plugins.Draw(
            export=True,
            draw_options={
                "polyline": False,
                "rectangle": True,
                "polygon": True,
                "circle": False,
                "marker": False,
                "circlemarker": False,
            },
            show_geometry_on_click=False,
            edit_options={"edit": True, "remove": True},
        )
    )
    m.add_child(folium.plugins.Fullscreen())

    # Get the unique identifier of the folium map and the GeoJSON layer
    map_id = m.get_name()
    chips_bbox_layer_id = chips_bbox_layer.get_name()

    # Add the Turf.js library to the map
    m.get_root().header.add_child(
        folium.JavascriptLink("https://unpkg.com/@turf/turf/turf.min.js")
    )

    # save the map and read in the HTML content
    m.save(HTML_FILE_NAME)
    # Read the generated HTML file
    with open(HTML_FILE_NAME, "r") as f:
        html_content = f.read()

    # Add custom JavaScript to handle the polygon drawing and feature selection
    custom_js = """
        function handleDrawCreated(e) {
            var layer = e.layer;
            var type = e.layerType;

            var drawnPolygon = layer.toGeoJSON();
            var selectedFeatures = [];

            geojson_bbbox_layer.eachLayer(function (featureLayer) {
                feat = featureLayer.toGeoJSON();
                feat_collection = turf.featureCollection([drawnPolygon, feat]);
                if (turf.intersect(feat_collection)) {
                    selectedFeatures.push(feat);
                }
            });

            var selectedGeoJson = {
                "type": "FeatureCollection",
                "features": selectedFeatures
            };

            console.log("The number of selected features: ", selectedFeatures.length);
            // Export the selected features to a GeoJSON file
            var blob = new Blob([JSON.stringify(selectedGeoJson)], { type: 'application/json' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'selected_features.geojson';
            a.click();

        }
        map.on('draw:created', handleDrawCreated);
    """
    # Replace the placeholders with the actual map and layer IDs
    custom_js = custom_js.replace("Select-AOI", f"{map_id}")
    custom_js = custom_js.replace("geojson_bbbox_layer", f"{chips_bbox_layer_id}")

    # Find the insertion point at the end of the HTML and add the custom JavaScript
    insertion_point = "\n</script>\n</html>".format(m.get_name())
    html_content = html_content.replace(
        insertion_point, custom_js + "\n" + insertion_point
    )

    # Write the modified HTML content back to the file
    with open(HTML_FILE_NAME, "w") as f:
        f.write(html_content)
