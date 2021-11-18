import os
import shutil
from shutil import copyfile
from random import randrange

# create train and test directories

def create_train_test(data_path):

    data_path_train_test = data_path + '_train_test'

    train_path = os.path.join(data_path_train_test, 'train')
    test_path = os.path.join(data_path_train_test, 'test')

    if os.path.exists(data_path_train_test):
        shutil.rmtree(data_path_train_test)
     
    os.mkdir(data_path_train_test)
    os.mkdir(train_path)
    os.mkdir(test_path)
    
    for subdir, dirs, files in os.walk(data_path):
        
        if subdir != data_path:
        
            count_img = 0
            list_img = []
            
            for file in files:
                path_src = os.path.join(subdir, file)
                list_img.append(file)
                class_path = os.path.join(train_path, os.path.basename(subdir))
                path_dst = os.path.join(class_path, file)
                if os.path.exists(class_path) == False:
                    os.mkdir(class_path)
                copyfile(path_src,path_dst)
                count_img += 1
            
            number_img = randrange(count_img)
            file_name = list_img[number_img]
            
            class_path_src = os.path.join(train_path, os.path.basename(subdir))
            path_src = os.path.join(class_path_src,file_name)
            
            class_path_dst = os.path.join(test_path, os.path.basename(subdir))
            path_dst = os.path.join(class_path_dst,file_name)
            
            if os.path.exists(class_path_dst) == False:
                os.mkdir(class_path_dst)
            
            copyfile(path_src,path_dst)
            os.remove(path_src)