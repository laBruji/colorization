import cv2
import math
import numpy as np
import sys

def extract_channels_ab(photo):
    '''
    Takes string with filename of image.
    Returns its a and b channels in the Lab color space
    '''
    image = cv2.imread(photo)
    width = 175
    height = 175
    desired_size = (width, height)
    resized = cv2.resize(image, desired_size)
    scaled = resized.astype("float32") / 255.0
    lab_repr = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    # Get the channels
    a = cv2.split(lab_repr)[1]
    b = cv2.split(lab_repr)[2]
    return (a, b)


def get_points_by_radius(photos):
    '''
    Takes list of strings with filename of images and returns a dictionary in which:
        keys: floats starting in 0.0 and increasing by 0.5.
            Represent distances from the origin of the ab color space
        values: set of points that have a distance from the origin shorter
            than key and greater than key-0.5
    '''
    # Points arranged by distance from the origin in the
    # ab color space
    pts_arranged_by_distance = {}   

    for photo in photos:
        a, b = extract_channels_ab(photo, 25)
        
        # Calculate distance from the origin in the ab color space
        # for each pixel
        for h in range(len(a)):
            for w in range(len(a[0])):
                x = a[h][w]
                y = b[h][w]
                dist_from_center = math.dist((x,y), (0,0))
                key = math.ceil(dist_from_center * 2) / 2
                if key not in pts_arranged_by_distance:
                    pts_arranged_by_distance[key] = set()
                pts_arranged_by_distance[key].add(tuple((x,y)))

    return pts_arranged_by_distance

def make_distances(points_by_radius, threshold = 0.5):
    '''
    Takes a dictionary of points arranged by their distance from the origin
    and a threshold.
    Returns a dictionary that maps each point to their adjacent points
    Two points are adjacent if the distance between them is less than threshold
    '''
    distances = {}
    # Calculate distance between points in the same layer and the next
    for r in points_by_radius:
        current_points = points_by_radius[r]
        if len(current_points) < 1000 and r + 0.5 in points_by_radius:
            current_points |= points_by_radius[r+0.5]
        
        current_points = list(current_points)
        # calculate distances between them
        for p in range(len(current_points)-1):
            for q in range(p, len(current_points)):
                p1 = current_points[p]
                p2 = current_points[q]
                if p1 not in distances:
                    distances[p1] = {p1: 0.0}
                if p2 not in distances.keys():
                    distances[p2] = {p2: 0.0}
                distance = math.dist(p1,p2)
                if distance < threshold:
                    distances[p1][p2] = distance
                    distances[p2][p1] = distance
    return distances


def make_connected_component(distances, node, radius):
    """ Return the connected component containing the point 'node' """

    agenda = [node]
    visited = set()

    while agenda:
        curr_node = agenda.pop()
        visited.add(curr_node)

        for child in distances[curr_node]:
            if not child in visited and distances[curr_node][child] < radius:
                agenda.append(child)

    return visited
    

def form_clusters(distances, cluster_radius):
    '''
    Takes a dictionary that maps each point to an adjacent one and
    the target radius of the cluster
    Returns clusters of points of radius cluster_radius
    '''
    clusters = []
    visited = set()
    for point in distances:
        if not point in visited:
            cluster = make_connected_component(distances, point, cluster_radius)
            visited |= cluster
            clusters.append(cluster)
    
    return clusters

def binarySearch(distances, left, right, target, percent_error):
    '''
    Performs binary search for the best radius of the clusters
    '''
    epsilon = target * percent_error; #10% of target

    if right < left:
        return None  # Maybe raise an exception
    elif len(distances) < target:
        return form_clusters(distances, left)
    else:
        mid = (left + right) / 2
        clusters = form_clusters(distances, mid)
        if 0 < len(clusters) - target < epsilon:
            return clusters
        elif len(clusters) < target:
            return binarySearch(distances, left, mid, target, percent_error)
        elif len(clusters) > target:
            return binarySearch(distances, mid, right, target, percent_error)

def local_dialect(distances, target=224**2, percent_error=0.1):
    '''
    Takes a dictionary that maps each point to an adjacent one
    Returns an array of target values which contains the most
    significant colors from points in the ab color space
    '''
    clusters = binarySearch(distances, 0, 0.7, target, percent_error)

    if not clusters:
        return []
    else:
        # delete the first 'len(clusters) - target' elements from clusters
        # when ordered by the size of the sets it contains
        cluster_averages = []
        to_delete = len(clusters) - target
        clusters.sort(key = len)
        clusters = clusters[to_delete:]

        for cluster in clusters:
            #cluster is a set of points (a,b)
            average_a = 0
            average_b = 0
            for point in cluster:
                average_a += point[0]
                average_b += point[1]
            average_a /= len(cluster)
            average_b /= len(cluster)
            cluster_averages.append((average_a, average_b))
        return cluster_averages

def get_image_from_pixels(clusters, target):
    '''
    Takes a list of pixels and its target dimensions to 
    yield an image in rgb space
    '''
    if len(clusters) != 0:
        l = 50
        L = [l for i in range(target)]
        a = []
        b = []

        if target**2 > len(clusters):
            target = math.floor(math.sqrt(len(clusters)))

        local_dialect = []
        for i in range(target):
            new_row = []
            for j in range(target):
                a = clusters[i*target + j][0]
                b = clusters[i*target + j][1]
                new_row.append((l, a, b))
            local_dialect.append(new_row)

        local_dialect_lab = np.float32(local_dialect)
        local_dialect_rgb = cv2.cvtColor(local_dialect_lab, cv2.COLOR_LAB2BGR)
        local_dialect_rgb = (255 * local_dialect_rgb).astype("uint8")
        return local_dialect_rgb
