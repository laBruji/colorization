import os
import pickle
import cv2
import math
import numpy as np
import itertools

MAX_THRESHOLD = 1.5


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


def compute_key(a, b, GRID_SIZE):
    """
    Given a point in the ab space, returns the region it belongs to
    :param a: float, a coordinate of ab color space
    :param b: float, b coordinate of ab color space
    :return: region of the point
    """
    # GRID_SIZE = 1
    return math.floor(a / GRID_SIZE), math.floor(b / GRID_SIZE)


def extract_ab_channels_from_image(image):
    """
    Extracts the a and b channels given the image
    :param image: image
    :return: image's a and b channels in the Lab color space
    """
    width, height = 175, 175
    desired_size = (width, height)
    resized = cv2.resize(image, desired_size)
    scaled = resized.astype("float32") / 255.0
    lab_repr = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    # Get the channels
    a = cv2.split(lab_repr)[1]
    b = cv2.split(lab_repr)[2]
    return a, b


def get_colors_by_regions(images, grid_size):
    """
    Get pixels of images arranged by regions of 1x1
    in the ab color space (a and b range from -128 to 127)
    :param images: a list of images
    :return: dictionary of regions mapped to the colors in that region
    """
    pts_arranged_by_regions = {}

    for image in images:
        a, b = extract_ab_channels_from_image(image)

        for y in range(len(a)):
            for x in range(len(a[0])):
                key = compute_key(a[y][x], b[y][x], grid_size)
                if key not in pts_arranged_by_regions:
                    pts_arranged_by_regions[key] = set()
                pts_arranged_by_regions[key].add((a[y][x], b[y][x]))

    return pts_arranged_by_regions


def get_neighboring_regions(region):
    """
    Get regions surrounding region
    :param region: a tuple of the form float x, float y
    :return: itertools.product of region and its surrounding regions
    """
    x, y = region
    return itertools.product([x - 1, x, x + 1], [y - 1, y, y + 1])


def get_neighboring_points(region, colors_by_regions):
    """
    Get points in region and in neighboring regions
    :param colors_by_regions: dictionary of regions mapped to points inside
                            that region
    :param region: a tuple of the form float x, float y
    :return: set of points in region and in neighboring regions
    """
    nr = get_neighboring_regions(region)
    neighboring_points = set()

    for r in nr:
        if r in colors_by_regions:
            neighboring_points |= colors_by_regions[r]

    return neighboring_points


def make_distances_by_regions(colors_by_regions, threshold=MAX_THRESHOLD):
    """
    Determines distances between pixels of adjacent regions
    :param colors_by_regions: dictionary of regions mapped to points inside
                            that region
    :param threshold: maximum distance for pixels to be considered "close"
    :return: dictionary that maps each point to another dictionary containing
            its adjacent points as keys, and the distance between them as values
            {(1,2): {(1,2):0}}
    """
    distances = {}

    # Calculate distance between points in the neighbor regions
    for region in colors_by_regions:
        regions = {region}
        nr = get_neighboring_regions(region)
        for r in nr:
            if r in colors_by_regions.keys():
                regions |= {r}

        _points = np.asarray(list(regions))
        for r in regions:
            deltas = _points - r
            dist = np.einsum('ij,ij->i', deltas, deltas)
            distances[r] = {}
            for i in range(len(_points)):
                if (d := dist[i]) < threshold:
                    distances[r][tuple(_points[i])] = d

    return distances


def _make_distances_by_regions(colors_by_regions, threshold=MAX_THRESHOLD):
    """
    Determines distances between pixels of adjacent regions
    :param colors_by_regions: dictionary of regions mapped to points inside
                            that region
    :param threshold: maximum distance for pixels to be considered "close"
    :return: dictionary that maps each point to another dictionary containing
            its adjacent points as keys, and the distance between them as values
            {(1,2): {(1,2):0, #of points in region}}
    """
    distances = {}

    # Calculate distance between points in the neighbor regions
    for region in colors_by_regions:
        distances[region] = {}
        distances[region][region] = (0.0, len(region))

    return distances


def make_connected_component(distances, node, radius):
    """
    Makes the connected component containing the point 'node'
    :param distances: dictionary that maps each point to another dictionary containing
            its adjacent points as keys, and the distance between them as values
    :param node: node to place in connected component
    :param radius: maximum distance to consider two nodes as part of the same
                connected component
    :return: set of points in the connected component containing the point 'node'
    """
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
        if point not in visited:
            cluster = make_connected_component(distances, point, cluster_radius)
            visited |= cluster
            clusters.append(cluster)

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
    clusters = form_clusters(distances, 2.0)
    return clusters
    # epsilon = target * percent_error  # 10% of target
    #
    # if right < left:
    #     return None  # Maybe raise an exception
    # elif len(distances) < target:
    #     return form_clusters(distances, left)
    # else:
    #     mid = (left + right) / 2
    #     clusters = form_clusters(distances, mid)
    #     if 0 < len(clusters) - target < epsilon:
    #         return clusters
    #     elif len(clusters) < target:
    #         return binary_search(distances, left, mid, target, percent_error)
    #     elif len(clusters) > target:
    #         return binary_search(distances, mid, right, target, percent_error)


def get_representative_colors_and_their_frequencies(distances, target=224 ** 2, percent_error=0.1):
    """
    Get approximately 'target' colors and their significance for the color dialect
    :param distances: dictionary that maps each point to another dictionary containing
            its adjacent points as keys, and the distance between them as values
    :param target: number of representative colors expected
    :param percent_error: margin of error on the size of the target
    :return: returns an array of pixel values and its frequencies in the form [(a,b,frequency)]
    """
    # clusters = binary_search(distances, 0, 0.7, target, percent_error)
    clusters = form_clusters(distances, 2.0)
    if not clusters:
        return []
    else:
        # delete the first 'len(clusters) - target' elements from clusters
        # when ordered by the size of the sets it contains
        cluster_averages = []
        if target == 224**2:
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
        lum = 50

        if target ** 2 > len(clusters):
            target = math.floor(math.sqrt(len(clusters)))

        local_dialect = []
        for i in range(target):
            new_row = []
            for j in range(target):
                a = clusters[i * target + j][0]
                b = clusters[i * target + j][1]
                new_row.append((lum, a, b))
            local_dialect.append(new_row)

        local_dialect_lab = np.float32(local_dialect)
        local_dialect_rgb = cv2.cvtColor(local_dialect_lab, cv2.COLOR_LAB2BGR)
        local_dialect_rgb = (255 * local_dialect_rgb).astype("uint8")
        local_dialect_resized = cv2.resize(local_dialect_rgb, (target, target))
        return local_dialect_resized


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
    colors_by_regions = get_colors_by_regions(images, 1)

    print("Calculate distances between points")
    distances = make_distances_by_regions(colors_by_regions)

    print("Creating local dialect")
    clusters = get_representative_colors_and_their_frequencies(distances)

    print("Save created local dialect")
    filename = f'Dialects/{location}.pkl'
    open_file = open(filename, "wb")
    pickle.dump(clusters, open_file)
    open_file.close()

    print("Save image of local dialect")
    local_dialect_rgb = get_image_from_pixels(clusters, 224)
    cv2.imwrite(f'Dialects/{location}_dialect.jpg', local_dialect_rgb)
    return clusters


def get_most_common_colors(image_filename):
    """
    Get color information (color and its frequency) of most common colors in image
    :param image_filename: string of image_filename
    :return: a list of tuples of the form (float a, float b, int c)
    """
    image = cv2.imread(image_filename)
    image = cv2.resize(image, (200, 200))
    colors_by_regions = get_colors_by_regions([image], 0.5)
    clusters = []
    for region, colors in colors_by_regions.items():
        clusters.append((region[0], region[1], len(colors)))
    return clusters
