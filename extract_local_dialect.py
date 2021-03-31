
import cv2
import math
import numpy as np
from aux_funcs_local_dialect import *

print("Organize points by distance from center")
points_by_radius = get_points_by_radius(["purple/purple_1.jpg", "purple/purple_2.jpg", "purple/purple_3.jpg"]) 
print("Calculate distances between points")
distances = make_distances(points_by_radius)
print("Creating local dialect")
clusters = local_dialect(distances)
local_dialect_rgb = get_image_from_pixels(clusters, 224)
cv2.imwrite("purple_dialect.jpg", local_dialect_rgb)
