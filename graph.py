import matplotlib.pyplot as plt
import pickle
from p5 import *
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color


def get_clusters(location):
    filename = f'Dialects/{location}.pkl'
    open_file = open(filename, "rb")
    clusters = pickle.load(open_file)
    open_file.close()
    return clusters


def get_zhang_error(location):
    filename = f'Regular Colorization/{location}/results.pkl'
    open_file = open(filename, "rb")
    errors = pickle.load(open_file)
    open_file.close()
    return errors


def get_rom_error(location):
    filename = f'Improved Colorization/{location}/results.pkl'
    open_file = open(filename, "rb")
    errors = pickle.load(open_file)
    open_file.close()
    return errors

def get_rgb_color(a, b):
    lab = LabColor(50, a, b)
    rgb_ = convert_color(lab, sRGBColor)
    return rgb_.clamped_rgb_r, rgb_.clamped_rgb_g, rgb_.clamped_rgb_b


def graph_dialect(location):
    clusters_ = get_clusters(location)
    c = [cl[2] for cl in clusters_]

    for cl in clusters_:
        m_size = remap(cl[2], (min(c), max(c)), (3, 13))
        rgb = get_rgb_color(cl[0], cl[1])
        plt.plot(cl[0], cl[1], color=rgb, marker="*", markersize=m_size)
    plt.show()


def graph_errors(location):
    zhang_errors = get_zhang_error(location)
    av_z = 0
    for e in zhang_errors:
        av_z += e
    av_z /= len(zhang_errors)
    print(av_z)
    rom_errors = get_rom_error(location)
    av_r = 0
    for e in rom_errors:
        av_r += e
    av_r /= len(rom_errors)
    print(av_r)
    plt.hlines(av_z, 0, len(zhang_errors), colors=["red"], linestyles='dashed')
    plt.hlines(av_r, 0, len(rom_errors), colors=["purple"], linestyles='dashed')
    plt.plot(zhang_errors, color="red")
    plt.plot(rom_errors, color="purple")
    plt.show()

