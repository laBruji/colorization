import cv2
import numpy as np


def regular_colorization(photo, filename):
    """
    Saves a recolorized version of photo to filename
    :param photo: string with filename of image
    :param filename: filename to save the recolorized image to.
    """
    # path to Caffe prototxt file
    prototxt = "models/colorization_deploy_v2.prototxt"
    # path to Caffe pre-trained model
    model = "models/colorization_release_v2.caffemodel"
    # path to cluster center points
    points = "models/pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    # add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    image = cv2.imread(photo)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # resize the Lab image to 224x224 (the dimensions the colorization
    # network accepts), split channels, extract the 'L' channel, and then
    # perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # pass the L channel through the network which will *predict* the 'a'
    # and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize the predicted 'ab' volume to the same dimensions as our
    # input image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
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
    cv2.imwrite(filename, colorized)


def recolorize_images(images, filenames):
    """
    Saves a recolorized version of images to filenames
    :param images: images to recolorize
    :param filenames: filenames to save the recolorized images to
    """
    for i in range(len(images)):
        regular_colorization(images[i], filenames[i])
