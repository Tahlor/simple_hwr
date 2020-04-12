from matplotlib import pyplot as plt
from pathlib import Path
import os
import cv2 as cv
import multiprocessing

def threshold(path,grayscale=True):

    if grayscale:
        img = cv.imread(str(path),0)
        img[img>200]=255
        return img, cv.fastNlMeansDenoising(img,None,20,7,21)

def plot(path, grayscale=True):
    #plt.figure(dpi=1200)
    img_original = cv.imread(str(path), 0)
    img_clean, img_cleaner = threshold(path)
    plt.subplot(311), plt.imshow(img_original)  # , cmap='gray')
    plt.subplot(312), plt.imshow(img_clean)  # , cmap='gray')
    plt.subplot(313), plt.imshow(img_cleaner)  # , cmap='gray')
    plt.show()

def save_function(tup):
    i,p = tup
    output = output_path / p.name
    # print(output)
    # son.relative_to(parent)
    img, *_ = threshold(p, True)
    cv.imwrite(str(output), img)
    if i % 100 == 0:
        print(i)

def test():
    paths = ("/media/data/GitHub/handwriting_data/train_offline/a01-011x-03.png", "/media/data/GitHub/handwriting_data/train_offline/a01-007x-07.png")
    #paths = "/media/data/GitHub/handwriting_data/train_offline_preprocessed/h01-014-07.png",
    for path in paths:
        print(path)
        plot(path)

if __name__=="__main__":
    root = Path("/media/data/GitHub/handwriting_data")

    variant="train_offline"
    main_path = Path(root / variant)
    output_path = Path(root / (variant + "_preprocessed"))
    output_path.mkdir(parents=True, exist_ok=True)

    poolcount = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=poolcount)
    pool.imap_unordered(save_function, enumerate(main_path.rglob("*.png")))  # iterates through everything all at once
    pool.close()

#test()
#main()
# Remove images that are stretched too far