import os
import numpy as np
from shutil import copyfile
import albumentations as A
import cv2

# perform augmentations for classes with only one image

def augment_data(data_path):

    classes_aug = []
    for subdir, dirs, files in os.walk(data_path):
        i = 0
        for file in files:
            i += 1
            if i == 1:
                classes_aug.append(os.path.basename(subdir))
            elif i == 2:
                classes_aug.remove(os.path.basename(subdir))
                
    transform = A.Compose(
        [A.CLAHE(),
         A.RandomRotate90(),
         A.Transpose(),
         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50,
                            rotate_limit=45, p=.75),
         A.Blur(blur_limit=3),
         A.OpticalDistortion(),
         A.GridDistortion()
        ])

    for classes in classes_aug:
        path = os.path.join(data_path, classes)
        for subdir, dirs, files in os.walk(path):
            file = files[0]
            file_path = os.path.join(path,file)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_alt = transform(image=image)['image']
            image_alt = cv2.cvtColor(image_alt, cv2.COLOR_RGB2BGR)
            filename = os.path.join(path,'Aug_' + file)
            cv2.imwrite(filename,image_alt)