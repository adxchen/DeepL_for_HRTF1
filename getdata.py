# 获取HRTF文件中的数据
import numpy as np
import h5py
from utils.hrtf import *
import matplotlib.pyplot as plt

hrtf = CipicHRTF('data/sofacoustics.org/data/database/cipic/subject_003.sofa', 44100.0)
print(hrtf.azimuths)
# get pinnas
pinnas = np.array(scipy.io.loadmat('data/sofacoustics.org/data/database/cipic/anthropometry/anthro.mat'))
# 45*17，45位受试者
x = pinnas.item()
print(x.keys())
print(x['id'])

def get_image(folder, num):
    num_str = str(num)
    if num < 100:
        num_str = '0' + num_str
    if num < 10:
        num_str = '0' + num_str
    num = num_str
    try:
        return plt.imread(folder+ 'Subject_'+str(num)+'/'+str(num)+'_left_side.jpg'), 0
    except:
        try:
            return plt.imread(folder+ 'Subject_'+str(num)+'/'+str(num)+'_right_side.jpg'), 1
        except:
            try:
                return plt.imread(folder+ 'Subject_'+str(num)+'/0'+str(num)+'_left.jpg'), 0
            except:
                try:
                    return plt.imread(folder+ 'Subject_'+str(num)+'/0'+str(num)+'_right.jpg'), 1
                except:
                    try:
                        return plt.imread(folder+ 'Subject_'+str(num)+'/0'+str(num)+'_left.JPG'), 0
                    except:
                        try:
                            return plt.imread(folder+ 'Subject_'+str(num)+'/0'+str(num)+'_right.JPG'), 1
                        except:
                            try:
                                return plt.imread(folder+ 'Subject_'+str(num)+'/Subject_'+str(num)+'_left_side.jpg'), 0
                            except:
                                try:
                                    return plt.imread(folder+ 'Subject_'+str(num)+'/0'+str(num)+'_left_2.jpg'), 0
                                except:
                                    return plt.imread(folder+ 'Subject_'+str(num)+'/00'+str(58)+'_left.jpg'), 0


valid_ranges = []
valid_images = []
valid_final = []
valid_image_ranges = []
for i in range(200):
    try:
        get_hrtf_sofa('data/sofacoustics.org/data/database/cipic/', i)
        valid_ranges.append(i)
        try:
            img, ear_num = get_image('data/sofacoustics.org/data/database/ear_photos/', i)
            #
            valid_images.append(img)
            valid_image_ranges.append((i, ear_num))
        except FileNotFoundError as err:
            print(err)
            print('Image not Found', i)
            valid_images.append(None)
            valid_image_ranges.append(None)
    except:
        pass
#         print('Failed to load sofa', i)
valid_image_ranges = np.array(valid_image_ranges,dtype=object)
print(valid_images)
print(valid_image_ranges)