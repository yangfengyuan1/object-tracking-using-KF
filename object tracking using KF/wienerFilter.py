from scipy.signal import wiener
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gasuss_noise(image, mean=0, var=0.01):
    
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

if __name__ == '__main__':
    lena = cv2.imread("/mnt/fengyuan/kalman-filter-in-single-object-tracking-main/data/test1.png")
    if lena.shape[-1] == 3:
        lenaGray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    else:
        lenaGray = lena.copy()

    plt.figure('orial Image')
    plt.imshow(lenaGray, cmap='gray')

    lenaNoise = gasuss_noise(lenaGray)

    plt.figure('Image add Gassion Noise')
    plt.imshow(lenaNoise, cmap='gray')

    
    lenaNoise = lenaNoise.astype('float64')
    lenaWiener = wiener(lenaNoise, [3, 3])
    lenaWiener = np.uint8(lenaWiener / lenaWiener.max() * 255)

    plt.figure('Image after wiener Filter')
    plt.imshow(lenaWiener, cmap='gray')
    plt.show()
