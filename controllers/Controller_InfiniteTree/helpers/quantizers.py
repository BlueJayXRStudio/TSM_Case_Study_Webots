import numpy as np

def normalize_angle(angle_rad):
    return angle_rad % (2 * np.pi)

def quantize_angle(angle_rad, num_bins):
    angle = normalize_angle(angle_rad)
    bin_size = 2 * np.pi / num_bins
    return int(angle // bin_size)

def quantize_position(x, y, cell_size):
    i = int(x // cell_size)
    j = int(y // cell_size)
    return (i, j)

if __name__ == "__main__":
    for i in range(0, 360):
        rad = i * np.pi / 180
        print(quantize_angle(rad, 32), rad)

    # for i in range(-100, 100):
    #     for j in range(-100, 100):
    #         x = i * 10 / 100
    #         y = j * 10 / 100
    #         print(x, y, quantize_position(x, y, 2.54 * 2))