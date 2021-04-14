from os import path
from aux_funcs_local_dialect import *


def evaluate_colorization(image, location):
    """
    Evaluate a colorized image according to its location
    :param image: string representing filename of the image
    :param location: string representing image's location
    :return: tuple of the average distance between local dialect
            and image dialect, and average difference in the
            frequencies of each color
    """
    print("Start evaluation")
    filename = f'{location}.pkl'

    if not path.exists(filename):  # create color dialect if it has not been calculated
        print("Local dialect not found. Creating local dialect")
        clusters_location = create_color_dialect(location)
    else:
        open_file = open(filename, "rb")
        clusters_location = pickle.load(open_file)
        open_file.close()

    location_freq_dict = extract_ab_pixels_from_clusters(clusters_location)
    dialect_pixels = list(location_freq_dict.keys())

    # Create color dialect of image
    clusters_image = create_color_dialect_image(image)
    image_freq_dict = extract_ab_pixels_from_clusters(clusters_image)
    image_dialect_pixels = list(image_freq_dict.keys())

    print("Calculating error")
    sum_distances_closest_points = 0
    sum_difference_frequencies = 0
    for pix in image_dialect_pixels:
        min_distance, closest_point_index = distance_to_closest_point(pix, dialect_pixels)
        sum_distances_closest_points += min_distance

        pix_freq = image_freq_dict[pix]
        closest_point_freq = location_freq_dict[dialect_pixels[closest_point_index]]

        sum_difference_frequencies += abs(pix_freq - closest_point_freq)

    average_distance_to_closest_point = sum_distances_closest_points / len(image_dialect_pixels)
    average_difference_frequencies = sum_difference_frequencies / len(image_dialect_pixels)

    return average_distance_to_closest_point, average_difference_frequencies
