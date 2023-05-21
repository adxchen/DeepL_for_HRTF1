# 获取HRTF文件中的数据
import numpy as np
import h5py
from utils.hrtf import *
import matplotlib.pyplot as plt

hrtf = CipicHRTF('data/sofacoustics.org/data/database/cipic/subject_003.sofa', 44100.0)
print(hrtf.azimuths)
# pinnas存放的是45位测试者的身体参数
pinnas = np.array(scipy.io.loadmat('data/sofacoustics.org/data/database/cipic/anthropometry/anthro.mat')['X'])
# 45*17，45位受试者
print(pinnas.shape)


def get_image(folder, num):
    num_str = str(num)
    if num < 100:
        num_str = '0' + num_str
    if num < 10:
        num_str = '0' + num_str
    num = num_str
    try:
        # 用于读取一张图片，将图像数据变成数组array
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

# 存放cipic数据库中已有的sofa文件
sofa_set = []
# 保存有效的图片
images_set = []
# 最终用到的sofa文件
sofa_finalset = []
# 图片所对应测试者的编号和左右耳编号
imagesinfo_set = []



for i in range(200):
    try:
        index = str(i)
        if i < 100:
            index = '0' + index
        if i < 10:
            index = '0' + index
        f =open('data/sofacoustics.org/data/database/cipic/subject_'+index+'.sofa')
        f.close()
        # 如果存在该SOFA文件
        # 将受试者编号加入sofa_set中
        sofa_set.append(i)
        # 查询受试者是否存在对应的耳朵图片
        try:
            img, ear_num = get_image('data/sofacoustics.org/data/database/ear_photos/', i)
            # 如果存在对应的图片，将图片和图片对应的信息分别添加到list中
            images_set.append(img)
            imagesinfo_set.append((i, ear_num))
        except FileNotFoundError as err:
            print(err)
            print('Image not Found', i)
            images_set.append(None)
            imagesinfo_set.append(None)
    except FileNotFoundError:
        # 如果不存在该文件
        pass
    except PermissionError:
        print("You don't have permission to access this file.")

print("length of valid_ranges:%d" % len(sofa_set))

print("length of images_set:%d" % len(images_set))

print("length of imagesinfo_set:%d" % len(imagesinfo_set))

print("imagesinfo_set:\n", imagesinfo_set)

# pinna_vals存放的是每个测试者的17个身体参数
pinna_vals = [pinnas[i] for i, val in enumerate(sofa_set)]
print("the length of pinna_vals:", len(pinna_vals))

# 45为受试者中，有些人的身体参数不完整，存在NAN
sofa_finalset = np.array(sofa_set)[~np.isnan(pinna_vals).any(axis=1)]
imagesinfo_set = np.array(imagesinfo_set)[~np.isnan(pinna_vals).any(axis=1)]
images_set = np.array(images_set)[~np.isnan(pinna_vals).any(axis=1)]
pinna_vals = np.array(pinna_vals)[~np.isnan(pinna_vals).any(axis=1)]

print(len(sofa_finalset))
print(len(imagesinfo_set))
print(len(images_set))
print(len(pinna_vals))

images_set = [images_set[idx] for idx, val in enumerate(imagesinfo_set) if val is not None]
pinna_vals = [pinna_vals[idx] for idx, val in enumerate(imagesinfo_set) if val is not None]
imagesinfo_set = [imagesinfo_set[idx] for idx, val in enumerate(imagesinfo_set) if val is not None]

print(len(sofa_finalset))
print(len(imagesinfo_set))
print(len(images_set))
print(len(pinna_vals))

impulse_vals = []
for i, ear in imagesinfo_set:
    impulse_vals.append(get_hrtf_sofa('data/sofacoustics.org/data/database/cipic', i).impulses[:,ear,:])
get_hrtf_sofa('data/sofacoustics.org/data/database/cipic', i)
print(impulse_vals)
