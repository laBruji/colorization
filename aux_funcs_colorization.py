import cv2
import numpy as np


def get_image_and_lab(photo):
    """
    Get image in RGB space and in LAB space from filename
    :param photo: string, filename of image
    :return: image in RGB, image in LAB
    """
    image = cv2.imread(photo)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    return image, lab


def get_LAB_channels(lab, target=224):
    """
    resize the Lab image to 224x224 (the dimensions the colorization
    network accepts), split channels, extract the 'L' channel, and
    perform mean centering
    :param target: desired dimension to resize image to a square image
    :param lab: image in LAB space
    :return: channels (L,a,b)
    """
    resized = cv2.resize(lab, (target, target))
    L = cv2.split(resized)[0]
    L -= 50
    a = cv2.split(resized)[1]
    b = cv2.split(resized)[2]

    return L, a, b


def resize_ab_to_224(ab):
    ab = cv2.resize(ab, (224, 224))
    return ab


def reconstructing_image(ab, image, lab):
    """
    Reconstructs image by merging an L channel with ab predicted channels
    :param ab: predicted channels
    :param image: RGB pixel values of original image
    :param lab: LAB pixel values of original image
    :return: colorized image in RGB space
    """
    # resize the predicted 'ab' volume to the same dimensions as our
    # input image
    ab = cv2.resize(np.float32(ab), (image.shape[1], image.shape[0]))
    # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the
    # predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # convert the output image from the Lab color space to RGB, then
    # clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # convert to an unsigned 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    return colorized


