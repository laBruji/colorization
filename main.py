import pickle
import numpy as np
from romero_colorization import romero_colorization
from zhang_colorization import zhang_colorization
from aux_funcs_evaluate import evaluate_images_colorization_in_location, get_local_dialect


def get_filenames_current_paths(sample_size, location):
    """
    Get filenames of images available in directory based on its location
    :param sample_size: integer, number of filenames to consider
    :param location: string
    :return: list of filenames
    """
    filenames = []
    for i in range(1, sample_size + 1):
        filenames.append(f'{location}/{location} ({i}).jpg')
    return filenames


def get_filenames_future_paths(sample_size, location):
    """
    Get filenames of recolorized images by zhang algorithm based on its location
    :param sample_size: integer, number of filenames to consider
    :param location: string
    :return: list of filenames
    """
    filenames = []
    for i in range(1, sample_size + 1):
        filenames.append(f'Regular Colorization/{location}/reg ({i}).jpg')

    return filenames


def get_improved_paths(sample_size, location):
    """
    Get filenames of recolorized images by romero algorithm based on its location
    :param sample_size: integer, number of filenames to consider
    :param location: string
    :return: list of filenames
    """
    new_filenames = []
    for i in range(1, sample_size + 1):
        new_filenames.append(f'Improved Colorization/{location}/reg ({i}).jpg')

    return new_filenames


def save_results(location, errors, regular=True):
    """
    Saves results of colorization errors in pickle file
    :param location: string
    :param errors: list of floats
    """
    filename = f'Regular Colorization/{location}/results.pkl'
    if not regular:
        filename = f'Improved Colorization/{location}/results.pkl'

    open_file = open(filename, "wb")
    pickle.dump(errors, open_file)
    open_file.close()


def colorize_and_evaluate():
    """
    Colorizes images available in directory for locations specified in outer scope.
    Evaluates colorization error and saves results
    """
    for location in locations:
        current_paths = get_filenames_current_paths(sample_size, location)
        future_paths = get_filenames_future_paths(sample_size, location)

        # Recolorizing images
        zhang_colorization(current_paths, future_paths)
        print("Images colorized by Zhang")

        zhang_errors = evaluate_images_colorization_in_location(future_paths, location)
        zhang_error = sum(np.asarray(zhang_errors)) / sample_size
        print(f'Zhang error: {zhang_error}')

        save_results(location, zhang_errors)

        improved_paths = get_improved_paths(sample_size, location)
        romero_colorization(future_paths, improved_paths, get_local_dialect(location))
        print("Images recolorized by Romero")

        romerrors = evaluate_images_colorization_in_location(improved_paths, location)

        romerror = sum(np.asarray(romerrors)) / sample_size
        print(f'Rom error: {romerror}')

        save_results(location, romerrors)


if __name__ == '__main__':
    sample_size = 10
    locations = ["Egypt", "Mexico", "Norway", "Paris"]

    colorize_and_evaluate()
