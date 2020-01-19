import os
import json

import cv2
import numpy as np
import keras
import csv


def treat_mask(mask) :
    ## mask must be a RGB image
    new_mask = np.zeros((mask.shape[0], mask.shape[1],3))
    l0 = (mask[:,:,0] < 50)
    l1 = (mask[:,:,1] < 50)
    l2 = (mask[:,:,2] < 50)
    g0= (mask[:,:,0] > 200)
    g1= (mask[:,:,1] > 200)
    g2= (mask[:,:,2] > 200)

    new_mask[np.logical_and(np.logical_and(g0, g1), g2)] = [1,0,0]
    new_mask[np.logical_and(np.logical_and(g0, l1), l2)] = [0,1,0]
    new_mask[np.logical_and(np.logical_and(l0, l1), l2)] = [0,0,1]
    return new_mask

def calculate_area_MBR(mask) :
    ulcer_blob = treat_mask(mask)[:,:,1].astype(np.uint8)
    real_area = np.sum(ulcer_blob)
    connectivity = 4  
    output = cv2.connectedComponentsWithStats(ulcer_blob, connectivity, cv2.CV_8U)
    im2, contours, hierarchy = cv2.findContours(ulcer_blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    area = 0
    for cnt in contours :
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        tmp = np.zeros(ulcer_blob.shape, dtype = np.uint8)
        cv2.fillPoly(tmp, [box], 1)
        area += np.sum(tmp)

    return area

class Settings() :
    def __init__(self, path = '../') :
        self.path = path
        self.loadSettings()

    def loadSettings(self):
        SETTINGS_JSON = self.path + 'settings.json'
        jsonFile = open(os.path.join(os.path.dirname(__file__), SETTINGS_JSON), 'r')
        self.settings = json.load(jsonFile)
        jsonFile.close()

        LOCAL_SETTINGS = self.path + 'local_settings.json'
        if os.path.isfile(os.path.join(os.path.dirname(__file__), LOCAL_SETTINGS)
    ):
            jsonFile = open(os.path.join(os.path.dirname(__file__), LOCAL_SETTINGS), 'r')
            localSettings = json.load(jsonFile)
            jsonFile.close()
            self.settings = dict(self.mergedicts(self.settings, localSettings))

    def mergedicts(self, dict1, dict2):
        '''
        Copied from here:

        http://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge
        '''
        for k in set(dict1.keys()).union(dict2.keys()):
            if k in dict1 and k in dict2:
                if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                    yield (k, dict(self.mergedicts(dict1[k], dict2[k])))
                else:
                    # If one of the values is not a dict, you can't continue merging it.
                    # Value from second dict **overrides** one in first and we move on.
                    yield (k, dict2[k])
                    # Alternatively, replace this with exception raiser to alert you of value conflicts
            elif k in dict1:
                yield (k, dict1[k])
            else:
                yield (k, dict2[k])


class ImageDataset() :
    def __init__(self, file_path, dataset_name = 'default', k_fold = '1', resize = None, training = False, ignore_augmentation = True, ignore_list = []) :
        self.dataset_name = dataset_name + '_K' + k_fold
        self.dataset_path = file_path
        self.open_folds_CSV(file_path)
        self.open(file_path, k_fold, resize, training, ignore_augmentation, ignore_list)

    def resize_image(self, img, resize = None) :
        if resize is None :
            return img
        elif resize[1] == '?' :
            max_size = resize[0]
            max_index = np.argmax(img.shape)
            max_value = np.max(img.shape)
            if max_value <= max_size :
                return img
            else :
                ratio = float(max_size)/max_value
                new_img = cv2.resize(img, (0,0), fx = ratio, fy = ratio)
                return new_img
            pass
        else :
            return cv2.resize(img, resize)

    def open_folds_CSV(self, filepath) :
        self.folds_list = {'1':[],'2':[],'3':[],'4':[],'5':[]}
        self.file_folds = dict()
        with open(os.path.join(filepath, 'kfolds.csv'),'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader :
                self.folds_list[row[1]].append(row[0])
                self.file_folds[row[0]] = row[1]
        


    def open(self, file_path, k_fold = '1', resize = None, training = False, ignore_augmentation = True, ignore_list = []) :
        self.imgList = []
        self.maskList= []
        self.filenameList = []

        use_fold = []
        if training :
            for i in ['1','2','3','4','5'] : 
                if i != k_fold :
                    use_fold.append(i)
        else :
            use_fold = [k_fold]

        print use_fold

        imgPath = os.path.join(file_path, 'src/img')
        maskPath = os.path.join(file_path, 'src/gt')
        for key in self.folds_list :
            if key not in use_fold : 
                continue
            for filename in self.folds_list[key] :
                print filename
                if filename in ignore_list :
                    print 'skipping', filename
                    continue
                
                if resize != (0,0) :
                    img = cv2.imread(os.path.join(imgPath, filename+'.JPG'), )
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = self.resize_image(img, resize)
                    self.imgList.append(img)

                    img = cv2.imread(os.path.join(maskPath, filename+'.png'),)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = self.resize_image(img, resize)
                    self.maskList.append(img)

                self.filenameList.append(filename)
                

            if ignore_augmentation == False :
                print 'loading augmentation'
                aug_imgPath = os.path.join(file_path, 'aug/%s/img' % (key))
                aug_maskPath = os.path.join(file_path, 'aug/%s/gt' % (key))


                self.filenameList.append(filename)
                print aug_imgPath
                print aug_maskPath

                for filePath in os.listdir(aug_imgPath, ) :
            
                    if not (filePath.endswith('.JPG') or filePath.endswith('.jpg') or filePath.endswith('.tiff')  or filePath.endswith('.png') ):
                        continue
                    

                    tmp = os.path.splitext(filePath)
                    curFilename = tmp[0]
                    print curFilename
                    self.filenameList.append(curFilename)
                    #print curFilename
                    if resize != (0,0) :
                        img = cv2.imread(os.path.join(aug_imgPath, filePath), )
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = self.resize_image(img, resize)
                        self.imgList.append(img)

                        img = cv2.imread(os.path.join(aug_maskPath,curFilename+'.png'),)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = self.resize_image(img, resize)
                        self.maskList.append(img)

                    
        self.imgList = np.array(self.imgList)
        self.maskList = np.array(self.maskList)

    def get_size(self) :
        return len(self.imgList)

    def get_images(self) :
        return np.array(self.imgList), np.array(self.maskList)

    def get_images_resized(self, new_size = (512, 512)) :
        img_resized = []
        mask_resized = []

        for i in xrange(len(self.imgList)) :
            print i, self.imgList[i].shape, new_size
            img_resized.append(cv2.resize(self.imgList[i], new_size))
        
        print 'done'

        for i in xrange(len(self.maskList)) :
            if self.maskList[i] is not None :
                mask_resized.append(cv2.resize(self.maskList[i], new_size))
        
        return np.array(img_resized), np.array(mask_resized)

    def data_augmentation(self, n_variations = 5, seed = 1) :
        data_gen_args = dict(rotation_range=15.,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             fill_mode='constant')

        img_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        mask_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

        np.random.seed(seed)
        nImg = len(self.filenameList)

        for i in xrange(nImg) :
            curFilename = self.filenameList[i]
            print self.file_folds[curFilename]
            key = self.file_folds[curFilename]

            print curFilename, key

            dstImg = os.path.join(self.dataset_path, 'aug/%s/img' %(key)) 
            dstGt = os.path.join(self.dataset_path, 'aug/%s/gt' %(key)) 

            if not os.path.isdir(dstImg) :
                os.makedirs(dstImg)
            if not os.path.isdir(dstGt) :
                os.makedirs(dstGt)

            cur_seed = np.random.randint(10000)

            j=0
            cur_shape = self.imgList[i].shape
            for batch in img_datagen.flow(self.imgList[i].reshape(1, cur_shape[0], cur_shape[1], cur_shape[2]), batch_size=1, 
                                          save_to_dir=dstImg, save_prefix='_aug_'+curFilename, 
                                          save_format='JPG', seed = cur_seed):
                j+=1
                if j>=n_variations:
                    break
            
            j=0
            cur_shape = self.maskList[i].shape
            for batch in mask_datagen.flow(self.maskList[i].reshape(1, cur_shape[0], cur_shape[1], cur_shape[2]), batch_size=1, 
                                          save_to_dir=dstGt, save_prefix='_aug_'+curFilename, 
                                          save_format='png', seed = cur_seed):
                j+=1
                if j>=n_variations:
                    break
        

    # Only merges the lists, does not updates the paths. Do not use for processing.
    def merge(self, dataset) :
        self.imgList = np.concatenate((self.imgList,dataset.imgList))
        self.maskList = np.concatenate((self.maskList,dataset.maskList))
        self.filenameList = np.concatenate((self.filenameList,dataset.filenameList))



