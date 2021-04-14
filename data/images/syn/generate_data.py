
import matplotlib.pyplot as plt
import numpy as np
import cv2

file_names = ['syn01.jpg', 'syn02.jpg', 'syn03.jpg', 
              'syn04.jpg', 'syn05.jpg', 'syn06.jpg',
              'syn07.jpg', 'syn08.jpg', 'syn09.jpg', 
              'syn10.jpg', 'syn11.jpg']
bg_file_names = ['bg01.jpg', 'bg02.jpg']

def random_crop_label(img):
    img_h, img_w, _ = img.shape
    start_x = np.random.randint(25, int(img_w/6))
    end_x = np.random.randint(25, int(img_w/6))
    start_y = np.random.randint(25, int(img_h/3))
    end_y = np.random.randint(25, int(img_h/3))
    croped_img = img[start_y:-end_y,start_x:-end_x,:]
    return croped_img

def generate_random_img(data_class, num=10000):
    
    img_list = []
    for file in file_names:
        img_list.append(plt.imread(file))
    bg_list = []
    for bg in bg_file_names:
        bg_list.append(plt.imread(bg))
    
    
    for i in range(num):
        # Select random image 
        img = img_list[np.random.randint(len(file_names))]
        # select random bg 
        bg = bg_list[np.random.randint(len(bg_file_names))]
        new_img = np.zeros(bg.shape)
        bh, bw, bc = bg.shape
        new_img[:,:,:] = bg[:,:,:]
        if np.random.randint(2)==0:
            # Make False case
            label='f'
            croped_img = random_crop_label(img)
            h, w, c = croped_img.shape
            start_x_bg = int((bw-w)/2)+np.random.randint(int((bw-w)/3))
            start_y_bg = int((bh-h)/2)+np.random.randint(int((bh-h)/3))
            new_img[start_y_bg:h+start_y_bg, start_x_bg:w+start_x_bg, :] = croped_img[:,:,:]
            save_file_name = f"./{data_class}/false/{label}__{i}.jpg"
            
        else:
            # Make True Case 
            label='t'
            h, w, c = img.shape
            start_x_bg = int((bw-w)/2)+np.random.randint(int((bw-w)/3))
            start_y_bg = int((bh-h)/2)+np.random.randint(int((bh-h)/3))
            new_img[start_y_bg:h+start_y_bg, start_x_bg:w+start_x_bg, :] = img[:,:,:]
            save_file_name = f"./{data_class}/true/{label}__{i}.jpg"
            
        
        new_img = cv2.resize(new_img, (400,400), interpolation=cv2.INTER_LINEAR)

        new_img = new_img/255. 
        plt.imsave(save_file_name, new_img)
        
        if i % 500 == 0:
            print (f"index : {i} saved")


generate_random_img('train', 1000)