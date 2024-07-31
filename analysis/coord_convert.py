from pyproj import Proj, transform
import math

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
