from pathlib import Path
import os
import cv2 as cv
from matplotlib import pyplot as plt

main_path = Path(r"/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/lines")


def denoise(path, grayscale=True):
    plt.figure(dpi=1200)
    if grayscale:
        img = cv.imread(path,0)
        dst = cv.fastNlMeansDenoising(img,None,10,7,21)
        plt.subplot(211),plt.imshow(img) #, cmap='gray')
        plt.subplot(212),plt.imshow(dst) # , cmap='gray')
    else:
        img = cv.imread(path)
        dst = cv.fastNlMeansDenoisingColored(src=img,dst=None,h=10,hColor=10,templateWindowSize=7,searchWindowSize=21)
        plt.subplot(211),plt.imshow(img)
        plt.subplot(212),plt.imshow(dst)

    plt.show()
    #plt.savefig(path, dpi=300)





for p in main_path.rglob("*.png"):
    print(p)
    denoise(str(p))
