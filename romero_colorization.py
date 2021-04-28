import cv2

from aux_funcs_colorization import get_image_and_lab, get_LAB_channels, reconstructing_image
from aux_funcs_evaluate import find_closest_color
from aux_funcs_local_dialect import compute_key


def recolorize_pixel(pixel, closest_color, distance, frequency):
    """
    Change pixel if it is close enough to closest_color
    :param pixel: tuple of floats in the form (a, b)
    :param closest_color: tuple of floats in the form (a,b)
    :param distance: float
    :return: tuple of floats in the form (a,b)
    """
    new_pixel = pixel
    if distance < 2.0 and frequency > 20:
        new_pixel = closest_color
    return new_pixel


def recolorize_one_image(image, new_filename, local_dialect):
    """
    Improves colorization of image based on its zhang colorization
    :param image: string, filename
    :param new_filename: where the image will be saved
    :param local_dialect: list of tuples in the form (float a, float b, int c)
    """
    image, lab = get_image_and_lab(image)
    L, a, b = get_LAB_channels(lab, target=100)
    new_a, new_b = a, b
    memo = {}

    for i in range(len(a)):
        for j in range(len(a[0])):
            pixel = compute_key(a[i][j], b[i][j], 0.5)
            if pixel not in memo:
                idx, distance = find_closest_color(pixel, local_dialect)
                closest_color = local_dialect[idx][0], local_dialect[idx][1]
                frequency = local_dialect[idx][2]
                new_a[i][j], new_b[i][j] = recolorize_pixel(pixel, closest_color, distance, frequency)
                memo[pixel] = (new_a[i][j], new_b[i][j])
            else:
                new_a[i][j], new_b[i][j] = memo[pixel]
    new_pixels = get_pixels_from_ab_values(new_a, new_b)
    colorized = reconstructing_image(new_pixels, image, lab)
    cv2.imwrite(new_filename, colorized)


def get_pixels_from_ab_values(new_a, new_b):
    """
    Rearrange ab values as a list of float tuples in the form (a,b)
    :param new_a: list of floats
    :param new_b: list of floats
    :return: list of float tuples in the form (a,b)
    """
    pixels = [[[0] for i in range(len(new_a))] for j in range(len(new_a[0]))]
    for i in range(len(new_a)):
        for j in range(len(new_a[0])):
            pixels[i][j] = new_a[i][j], new_b[i][j]
    return pixels


def romero_colorization(images, new_filenames, local_dialect):
    """
    Colorizes images
    :param images: list of string, filename of images to colorize
    :param new_filenames: list of strings, filenames where new images will be saved
    :param local_dialect:
    :return:
    """
    for i in range(len(images)):
        recolorize_one_image(images[i], new_filenames[i], local_dialect)
