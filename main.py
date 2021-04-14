from evaluate_colorization import *
from aux_funcs_local_dialect import *
from aux_funcs_colorization import *


if __name__ == '__main__':
    sample_size = 10

    # # images to recolorize for evaluation
    # images = []

    # for i in range(1, sample_size + 1):
    #     images.append(f'Mexico/Mexico ({i}).jpg')
    #
    filenames = []
    for i in range(1, sample_size + 1):
        filenames.append(f'Regular Colorization/Mexico/reg ({i}).jpg')
    #
    # # Recolorizing images
    # recolorize_images(images, filenames)
    # print("Images recolorized")

    results = []
    for i in range(sample_size):
        results.append(evaluate_colorization(filenames[i], "Mexico"))

    filename = 'Regular Colorization/Mexico/results.pkl'
    open_file = open(filename, "wb")
    pickle.dump(results, open_file)
    open_file.close()




