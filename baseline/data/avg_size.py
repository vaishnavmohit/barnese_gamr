from PIL import Image
import skimage.io as io
import glob
import numpy as np

im_path='/cifs/data/tserre_lrs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100/train/*/*.jpg'
image_h = []
image_w = []
img_mean_0 = []
img_mean_1 = []
img_mean_2 = []
img_std_0 = []
img_std_1 = []
img_std_2 = []
for filename in glob.glob(im_path): #assuming gif
    im=Image.open(filename).convert('RGB') # [c, h, w]
    im = np.array(im)/255
    w, h, c = im.shape
    image_h.append(h)
    image_w.append(w)
    img_mean_0.append(im[:,:,0].mean())
    img_mean_1.append(im[:,:,1].mean())
    img_mean_2.append(im[:,:,2].mean())
    img_std_0.append(im[:,:,0].std())
    img_std_1.append(im[:,:,1].std())
    img_std_2.append(im[:,:,2].std())
# printing average height and width

print('mean across channel is: ', np.array(img_mean_0).mean(), ' ', np.array(img_mean_1).mean(), ' ', np.array(img_mean_2).mean())
print('std across channel is: ', np.array(img_std_0).mean(), ' ', np.array(img_std_1).mean(), ' ', np.array(img_std_2).mean())
print('mean height is: ', np.array(image_h).mean(), ' and mean width is: ', np.array(image_w).mean())
                                                                                                               