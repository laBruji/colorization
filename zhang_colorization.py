from aux_funcs_colorization import *


def get_zhang_net():
    """
    Get neural net from Zhang's models
    :return: cv2.dnn object
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
    return net


def get_zhang_ab_prediction(L):
    """
    Get ab channels predicted by zhang algorithm
    :param L: luminosity channel of the image to recolorize
    :return: ab channels
    """
    # pass the L channel through the network which will *predict* the 'a'
    # and 'b' channel values
    net = get_zhang_net()
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    return ab


def zhang_colorization(photos, filenames):
    """
    Saves a recolorized version of photo to filename
    :param photos: string with filename of images
    :param filenames: filenames to save the recolorized images to.
    """
    assert len(photos) == len(filenames)
    for i in range(len(photos)):
        photo = photos[i]
        filename = filenames[i]
        image, lab = get_image_and_lab(photo)
        L, a_original, b_original = get_LAB_channels(lab)
        ab = get_zhang_ab_prediction(L)
        colorized = reconstructing_image(ab, image, lab)
        cv2.imwrite(filename, colorized)