from pyproj import Proj, transform
import math
import pickle
import folium
def get_geom_size():
    # Initialize the WGS84 (lat/lon) and UTM projection
    proj_wgs84 = Proj(init='epsg:4326')  # WGS84
    proj_utm = Proj(proj="utm", zone=17, ellps="WGS84", south=False)  # UTM zone for Miami area (zone 17)

    # Coordinates of the rectangle in lat/lon
    coordinates = [
        (25.7581072, -80.3738942),
        (25.7581072, -80.3734494),
        (25.7583659, -80.3734494),
        (25.7583659, -80.3738942)
    ]

    # Convert lat/lon to UTM
    utm_coordinates = [transform(proj_wgs84, proj_utm, lon, lat) for lat, lon in coordinates]

    # Extract UTM easting and northing separately
    eastings = [coord[0] for coord in utm_coordinates]
    northings = [coord[1] for coord in utm_coordinates]

    # Calculate the width (difference in easting) and height (difference in northing)
    width = math.dist([eastings[0], northings[0]], [eastings[1], northings[1]])
    height = math.dist([eastings[1], northings[1]], [eastings[2], northings[2]])

    print(f"Width of the rectangle: {width} meters")
    print(f"Height of the rectangle: {height} meters")


def transform_path(metric_path, metric_bounds, lat_lon_bounds):
    # Unpack bounds
    x_min, y_min, x_max, y_max = metric_bounds
    lat_min, lon_min, lat_max, lon_max = lat_lon_bounds

    # Function to normalize metric coordinates
    def normalize(x, y):
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)
        return x_norm, y_norm

    # Function to scale normalized coordinates to lat-lon domain
    def scale_to_lat_lon(x_norm, y_norm):
        lat = lat_min + y_norm * (lat_max - lat_min)
        lon = lon_min + x_norm * (lon_max - lon_min)
        return lat, lon

    # Transform the path
    lat_lon_path = []
    for (x, y) in metric_path:
        x_norm, y_norm = normalize(x, y)
        lat, lon = scale_to_lat_lon(x_norm, y_norm)
        lat_lon_path.append((lat, lon))

    return lat_lon_path


def get_trajectories(file_path):
    with open(file_path, 'rb') as file:
        loaded_history_robot_states = pickle.load(file)
    return loaded_history_robot_states


if __name__ == '__main__':
    seed = 3560  # max distance: 4.607
    filepath = f"/home/airlab/PycharmProjects/LCD_RIG/outputs/{seed}/temp_data/distributed/ak_team3_path_v1.pkl"
    trajs = get_trajectories(filepath)
    # Example usage
    metric_bounds = (-14, -14, 14, 14)
    lat_lon_bounds = (25.7581072, -80.3734494, 25.7583659, -80.3738942)

    map_center = [(lat_lon_bounds[0] + lat_lon_bounds[2]) / 2, (lat_lon_bounds[1] + lat_lon_bounds[3]) / 2]
    map = folium.Map(location=map_center, zoom_start=100)
    colors = ["red", "blue", "green"]
    for i, (robotName, metric_path) in enumerate(trajs.items()):
        lat_lon_path = transform_path(metric_path, metric_bounds, lat_lon_bounds)
        # Create a map centered around the middle of the lat-lon bounds


        # Add the path to the map
        folium.PolyLine(lat_lon_path, color=colors[i], weight=2.5, opacity=1).add_to(map)

        # # Add markers for each point in the path
        # for lat, lon in lat_lon_path:
        #     folium.Marker([lat, lon]).add_to(map)

        # Save the map to an HTML file
    map.save(f"path_map.html")
    #
    # # Display the map
    # map


