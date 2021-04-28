import pickle
from os import path
from aux_funcs_local_dialect import create_color_dialect, get_most_common_colors
import numpy as np


def get_colors_from_dialect(local_dialect):
    """
    Get colors from local dialect
    :param local_dialect: a list of tuples of the form (float a, float b, int c)
    :return: a list of tuples of the form (float a, float b)
    """
    colors = []

    for color_info in local_dialect:
        colors.append((color_info[0], color_info[1]))
    return colors


def distance_from_color_to_local_dialect(color, local_dialect):
    """
    Calculate distance from color to all colors in local dialect
    :param color: a tuple of floats with the form (a,b)
    :param local_dialect: a list of tuples of the form (float a, float b, int c)
    :return: a numpy array of distances between color and colors in local dialect
    """
    all_colors = np.asarray(get_colors_from_dialect(local_dialect))
    differences = all_colors - color
    distances = np.einsum('ij, ij->i', differences, differences)
    return distances


def find_closest_color(color, local_dialect):
    """
    Find closest color to 'color' in local_dialect
    :param color: a tuple of floats with the form (a,b)
    :param local_dialect: a list of tuples of the form (float a, float b, int c)
    :return: an integer, the index of the closest color to 'color' in local_dialect
            and a float that represents the distance between them
    """
    distances = distance_from_color_to_local_dialect(color, local_dialect)
    return np.argmin(distances),  min(distances)


def evaluate_one_color(color, color_frequency, local_dialect):
    """
    Calculate error between 'color' and the closest color in local_dialect
    :param color: a tuple of floats in the form (a,b)
    :param color_frequency: integer
    :param local_dialect: a list of tuples of the form (float a, float b, int c)
    :return: a float representing error of the color in given local_dialect
    """
    index_closest_color, distance_to_closest_color = find_closest_color(color, local_dialect)
    # closest_color = local_dialect[index_closest_color][0], local_dialect[index_closest_color][1]
    frequency = local_dialect[index_closest_color][2]
    freq_difference = abs(color_frequency - frequency)
    error = distance_to_closest_color * freq_difference
    return error


def evaluate_colors(colors_info, local_dialect):
    """
    Calculate error between 'colors' and their corresponding closest colors in
    local_dialect
    :param colors_info: a list of tuples of the form (float a, float b, int c)
    :param local_dialect: a list of tuples of the form (float a, float b, int c)
    :return: a float representing the error of the colors in the given local_dialect
    """
    sum_errors = 0
    for info in colors_info:
        color = info[0], info[1]
        frequency = info[2]
        sum_errors += evaluate_one_color(color, frequency, local_dialect)

    return sum_errors / len(colors_info)


def evaluate_one_image(image, local_dialect):
    """
    Calculate the error between the colors of one image and its local dialect
    :param image: string, filename
    :param local_dialect: a list of tuples of the form (float a, float b, int c)
    :return: float representing error in image
    """
    colors_info = get_most_common_colors(image)
    error = evaluate_colors(colors_info, local_dialect)
    return error


def get_local_dialect(location):
    """
    Get local dialect of location
    :param location: a string
    :return: a list of tuples of the form (float a, float b, int c)
    """
    filename = f'Dialects/{location}.pkl'
    if not path.exists(filename):  # create color dialect if it has not been calculated
        print("Local dialect not found. Creating local dialect")
        local_dialect = create_color_dialect(location)
    else:
        open_file = open(filename, "rb")
        local_dialect = pickle.load(open_file)
        open_file.close()

    return local_dialect


def evaluate_images_colorization_in_location(images_filenames, location):
    """
    Calculate errors in images colorization according to the given location
    :param images_filenames: a list of string with images filenames
    :param location: a string with the location
    :return: a list of floats representing the errors in colors
    """
    local_dialect = get_local_dialect(location)

    errors = []
    for image in images_filenames:
        print(image)
        error = evaluate_one_image(image, local_dialect)
        errors.append(error)

    return errors

