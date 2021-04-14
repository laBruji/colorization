import os
import pickle
import cv2
import math
import numpy as np
import itertools

MAX_THRESHOLD = 1


def extract_ab_channels_from_image(photo):
    """
    Extracts the a and b channels given the image
    :param photo: image
    :return: image's a and b channels in the Lab color space
    """
    width, height = 175, 175
    desired_size = (width, height)
    resized = cv2.resize(photo, desired_size)
    scaled = resized.astype("float32") / 255.0
    lab_repr = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    # Get the channels
    a = cv2.split(lab_repr)[1]
    b = cv2.split(lab_repr)[2]
    return a, b


def extract_ab_pixels_from_clusters(clusters):
    """
    Extracts pixels of the form (a, b) in LAB color space
    and its associated frequency given clusters of pixels
    :param clusters: list of pixel information in the form (a,b,frequency)
    :return: dictionary that maps a pixel (a,b) to its frequency
    """
    result = {}

    for i in range(len(clusters)):
        a = clusters[i][0]
        b = clusters[i][1]
        frequency = clusters[i][2]
        result[(a, b)] = frequency
    return result


def compute_key(x, y):
    """
    Given a point in the ab space, returns the region it belongs to
    """
    GRID_SIZE = 0.7
    return math.floor(x / GRID_SIZE), math.floor(y / GRID_SIZE)


def get_points_by_regions(photos):
    """
    Arranges ab pixels of photos according to the region
    in the a,b space where they are
    :return: dictionary of regions mapped to points inside
            that region
    """
    pts_arranged_by_regions = {}

    for photo in photos:
        a, b = extract_ab_channels_from_image(photo)

        for y in range(len(a)):
            for x in range(len(a[0])):
                key = compute_key(a[y][x], b[y][x])
                if key not in pts_arranged_by_regions:
                    pts_arranged_by_regions[key] = set()
                pts_arranged_by_regions[key].add((a[y][x], b[y][x]))

    return pts_arranged_by_regions


def get_neighboring_regions(region):
    """
    Get regions surrounding region
    """
    x, y = region
    return itertools.product([x - 1, x, x + 1], [y - 1, y, y + 1])


def make_distances_by_regions(points_by_regions, threshold=MAX_THRESHOLD):
    """
    Determines distances between pixels of adjacent regions
    :param points_by_regions: dictionary of regions mapped to points inside
                            that region
    :param threshold: maximum distance for pixels to be considered "close"
    :return: dictionary that maps each point to another dictionary containing
            its adjacent points as keys, and the distance between them as values
    """
    distances = {}
    visited_points = set()

    # Calculate distance between points in the neighbor regions
    for region in points_by_regions:
        nr = get_neighboring_regions(region)
        points = set() | points_by_regions[region]
        neighboring_points = set() | points_by_regions[region]

        for r in nr:
            if r in points_by_regions:
                neighboring_points |= points_by_regions[r]

        neighboring_points -= visited_points

        _points = np.asarray(list(points))
        for p in points:
            # distances between all the points
            deltas = _points - p
            dist = np.einsum('ij,ij->i', deltas, deltas)
            distances[p] = {}
            for i in range(len(_points)):
                if (d := dist[i]) < threshold:
                    distances[p][tuple(_points[i])] = d

    return distances


def make_connected_component(distances, node, radius):
    '''
    Makes the connected component containing the point 'node'
    :param distances: dictionary that maps each point to another dictionary containing
            its adjacent points as keys, and the distance between them as values
    :param node: node to place in connected component
    :param radius: maximum distance to consider two nodes as part of the same
                connected component
    :return: set of points in the connected component containing the point 'node'
    '''
    agenda = [node]
    visited = set()
    i = 0
    while agenda:
        i += 1
        curr_node = agenda.pop()
        visited.add(curr_node)

        for child in distances[curr_node]:
            if child not in visited and distances[curr_node][child] < radius:
                agenda.append(child)

    return visited


def form_clusters(distances, cluster_radius):
    """
    Separates pixels into clusters of radius cluster_radius
    :param distances: dictionary that maps each point to another dictionary containing
            its adjacent points as keys, and the distance between them as values
    :param cluster_radius: maximum size of clusters
    :return: list of clusters of points of radius cluster_radius
    """
    clusters = []
    visited = set()
    for point in distances:
        if not point in visited:
            cluster = make_connected_component(distances, point, cluster_radius)
            visited |= cluster
            clusters.append(cluster)

    print(len(clusters))
    return clusters


def binary_search(distances, left, right, target, percent_error):
    """
    Performs binary search for the best radius of clusters
    :param distances: dictionary that maps each point to another dictionary containing
            its adjacent points as keys, and the distance between them as values
    :param left: lower limit for radius
    :param right: upper limit for radius
    :param target: number of clusters we want to form
    :param percent_error: margin of error for number of clusters
    :return: list of approximately 'target' number of clusters
    """
    epsilon = target * percent_error;  # 10% of target

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
            return binary_search(distances, left, mid, target, percent_error)
        elif len(clusters) > target:
            return binary_search(distances, mid, right, target, percent_error)


def local_dialect(distances, target=224 ** 2, percent_error=0.1):
    """
    Calculates pixels and their significance for the color dialect
    :param distances: dictionary that maps each point to another dictionary containing
            its adjacent points as keys, and the distance between them as values
    :param target: size of the dialect
    :param percent_error: margin of error on the size of the target
    :return: returns an array of pixel values and its frequencies in the form [(a,b,frequency)]
    """
    clusters = binary_search(distances, 0, MAX_THRESHOLD, target, percent_error)

    if not clusters:
        return []
    else:
        # delete the first 'len(clusters) - target' elements from clusters
        # when ordered by the size of the sets it contains
        cluster_averages = []
        to_delete = len(clusters) - target
        clusters.sort(key=len)
        clusters = clusters[to_delete:]

        for cluster in clusters:
            # cluster is a set of points (a,b)
            cluster = np.asarray(list(cluster))
            average = np.sum(cluster, axis=0) / len(cluster)
            cluster_averages.append((average[0], average[1], len(cluster)))

        return cluster_averages


def get_image_from_pixels(clusters, target):
    """
    Get image of the pixels in clusters for visualization
    :param clusters: list of pixels
    :param target: target dimensions of expected image
    :return: image in rgb space
    """
    if len(clusters) != 0:
        l = 50
        L = [l for i in range(target)]
        a = []
        b = []

        if target ** 2 > len(clusters):
            target = math.floor(math.sqrt(len(clusters)))

        local_dialect = []
        for i in range(target):
            new_row = []
            for j in range(target):
                a = clusters[i * target + j][0]
                b = clusters[i * target + j][1]
                new_row.append((l, a, b))
            local_dialect.append(new_row)

        local_dialect_lab = np.float32(local_dialect)
        local_dialect_rgb = cv2.cvtColor(local_dialect_lab, cv2.COLOR_LAB2BGR)
        local_dialect_rgb = (255 * local_dialect_rgb).astype("uint8")
        return local_dialect_rgb


def load_images_from_folder(folder):
    """
    Load images from folder
    :param folder: string with folder path
    :return: list of images in folder
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def create_color_dialect(location):
    """
    Creates color dialect given a location and saves it
    for visualization purposes.
    :param location: string with location of the dialect
    :return: list of clusters in the dialect
    """
    print("Extract images to create local dialect")
    images = load_images_from_folder(location)

    print("Organize points by regions")
    points_by_regions = get_points_by_regions(images)

    print("Calculate distances between points")
    distances = make_distances_by_regions(points_by_regions)

    print("Creating local dialect")
    clusters = local_dialect(distances)

    print("Save created local dialect")
    filename = (f'{location}.pkl')
    open_file = open(filename, "wb")
    pickle.dump(clusters, open_file)
    open_file.close()

    print("Save image of local dialect")
    local_dialect_rgb = get_image_from_pixels(clusters, 224)
    cv2.imwrite(f'Dialects/{location}_dialect.jpg', local_dialect_rgb)
    return clusters


def create_color_dialect_image(image):
    """
    Creates color dialect given an image
    :param image: string with filename of image
    :return: list of clusters in the dialect
    """
    print("Organize points by regions")
    points_by_regions = get_points_by_regions([image])

    print("Calculate distances between points")
    distances = make_distances_by_regions(points_by_regions)

    print("Creating local dialect")
    clusters = local_dialect(distances)

    return clusters


def distance_to_closest_point(node, nodes):
    """
    Calculates distance from node to closest point
    in nodes
    :param node: tuple of a and b values
    :param nodes: list of tuples of a and b values
    :return: tuple of minimum distance and the index to closest node
    """
    nodes = np.asarray(nodes)
    deltas = nodes - node
    distance = np.einsum('ij, ij->i', deltas, deltas)
    return min(distance), np.argmin(distance)
