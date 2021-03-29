
import cv2
import math
import numpy as np
from aux_funcs_local_dialect import *

# Script to get local_dialect (array of 224**2 of length of a photo)

print("Organize points by distance from center")
points_by_radius = get_points_by_radius("pic.jpg")
print("Calculate distances between points")
distances = make_distances(points_by_radius)
print("Creating local dialect")
clusters = local_dialect(distances)

if len(clusters) != 0:
	l = 50
	L = [l for i in range(224)]
	a = []
	b = []

	local_dialect = []
	for i in range(224):
	    new_row = []
	    for j in range(224):
	        a = clusters[i*224 + j][0]
	        b = clusters[i*224 + j][1]
	        new_row.append((l, a, b))
	    local_dialect.append(new_row)

	local_dialect_lab = np.float32(local_dialect)
	local_dialect_rgb = cv2.cvtColor(local_dialect_lab, cv2.COLOR_LAB2BGR)
	local_dialect_rgb = (255 * local_dialect_rgb).astype("uint8")
	cv2.imwrite("testingLD.jpg", local_dialect_rgb)
