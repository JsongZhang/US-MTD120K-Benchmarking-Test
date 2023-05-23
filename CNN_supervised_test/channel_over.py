import cv2
import os
from PIL import Image
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


file_dir = "./all_data_label/thyroid_7LPRT/"
out_dir = "./all_data_label/thyroid_7LPRT/"
a = os.listdir(file_dir)
Sum = 0
for i in a:
    print(i)
    I = Image.open(file_dir + i)
    L = I.convert('L')
    b = np.array(L)
    image = np.expand_dims(b, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    cv2.imwrite(out_dir + i, image)
    Sum += 1

print(Sum)