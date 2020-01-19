import sys
import os
import argparse

import cv2
import numpy as np

import tensorflow as tf
import keras

import src.util
import src.metrics
import src.unet

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


def getMethod(model_args = None, dataset = None):
    model = None
    if model_args is None :
        print 'Invalid method.'
        exit()
    elif model_args['model_type'] == 'unet' :
        print 'Creating U-Net'
        model = src.unet.UNet()
        loadFrom = model.create_model_filename(dataset, model_args)
        model.load(loadFrom)
    return model

def training(dataset_path, dataset_name, k_fold = '1', model_args = None) :
    if not os.path.isdir('./models') :
        os.makedirs('./models')

    if model_args is None :
        print 'Invalid model.'
        exit()
    elif model_args['model_type'] == 'unet' :
        print 'Creating U-Net'
        model = src.unet.UNet()

        trainset = src.util.ImageDataset(dataset_path, dataset_name = dataset_name, k_fold = k_fold, resize = (512, 512), training = True, ignore_augmentation = False)
        
    
    saveTo = model.create_model_filename(trainset, model_args)
    print saveTo
    model.train(trainset = trainset, saveTo = saveTo, model_args = model_args)
    

def runSegmentation(dataset_path, dataset_name, k_fold = '1', model_args = None) :
    dataset = src.util.ImageDataset(dataset_path, dataset_name = dataset_name, k_fold = k_fold)

    method = getMethod(model_args, dataset)

    output_path = method.create_output_path(dataset, model_args)
    print output_path
    if not os.path.isdir(output_path) :
        os.makedirs(output_path)

    for i in xrange(dataset.get_size()) :
        print i, dataset.get_size(), dataset.filenameList[i]
        img = dataset.imgList[i]
        mask = dataset.maskList[i]
        filename = dataset.filenameList[i]+'_out.png'
        ret = method.run(img, filename = dataset.filenameList[i])

        cv2.imwrite(os.path.join(output_path, filename), ret)

def runSingleSegmentation(in_image, in_model, out_image = None) :
    output_path = 'output.png'
    if out_image is not None :
        output_path = out_image

    print in_image, in_model, out_image
    
    method = src.unet.UNet()
    method.load(in_model)

    img = cv2.imread(in_image, )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ret = method.run(img)
    cv2.imwrite(output_path, ret)
    

def evaluateOutput(datasets):
    ds = dict()
    for names in datasets:
        print names
        datasetPath = datasets[names]
        print datasetPath
        dataset = src.util.ImageDataset(datasetPath, dataset_name = names, k_fold='0', training = True)
        ds[names] = dataset


    root_folder = './output/'
    for ds_folder in os.listdir(root_folder) :
        if ds_folder not in ds : 
            continue
        print ds_folder
        full_ds_folder = os.path.join(root_folder, ds_folder)
        print full_ds_folder
        if not os.path.isdir(full_ds_folder) :
            continue
        for method_folder in os.listdir(full_ds_folder) :
            print method_folder
            full_method_folder = os.path.join(full_ds_folder, method_folder)
            if not os.path.isdir(full_method_folder) :
                continue
            filename = ds_folder + '_' + method_folder + '.csv'
            print filename
            if os.path.isfile(os.path.join(root_folder, filename)) :
                print 'Already calculated, skipping.'
                continue
            res_list = []
            with open(os.path.join(root_folder, filename), 'w') as fFolder:
                print >> fFolder, '#Filename,Jaccard,Dice_Coefficient,Accuracy,Precision,Recall,F_Measure'
                curTotal = ds[ds_folder].get_size()
                for i in xrange(ds[ds_folder].get_size()) :
                    #print ds[ds_folder].filenameList[i]
                    cur_file = ds[ds_folder].filenameList[i] + '_out.png'
                    if not os.path.isfile(os.path.join(full_method_folder,cur_file)) :
                        print 'Failed to load %s.' % (cur_file)
                        continue
                    ret = cv2.imread(os.path.join(full_method_folder,cur_file), )
                    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)

                    results = list(src.metrics.calculate_metrics(ret, ds[ds_folder].maskList[i]))
                    print '\r%d/%d' % (i, curTotal)
                    res_list.append(results)
                    print >> fFolder, '%s,%s' % (ds[ds_folder].filenameList[i], ','.join(str(e) for e in results))
            res_list = np.array(res_list)
            
            mean_res = np.mean(res_list, axis = 0)
            std_res = np.std(res_list, axis = 0)
            inter = np.ravel(np.column_stack((mean_res,std_res)))
            with open('./output/results.csv','a') as fResults :
                print >> fResults, '%s,%s' % (filename, ','.join(str(e) for e in inter))
                

def data_augmentation(dataset_path, n_variations = 5) :
    trainset = src.util.ImageDataset(trainPath, k_fold = '0', training = True)
    trainset.data_augmentation(n_variations = n_variations)               




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Gota')
    parser.add_argument('-a', action='store', dest = 'action', help = 'Action (train, test, evaluate, augmentation, area, ruler, single)', required = True)
    parser.add_argument('-m', action='store', dest = 'model', help = 'Model (see settings.json for reference)', required = False)
    parser.add_argument('-d', action='store', dest = 'dataset', help = 'Dataset (see settings.json for reference)', required = False)
    parser.add_argument('-k', action='store', dest = 'kfold', help = 'K fold (1 through 5)', required = False)
    parser.add_argument('-n', action='store', dest = 'augmentation', type = int, help = 'Number of variations to augment the dataset.', required = False)
    parser.add_argument('-t', action='store', dest = 'threshold', type = float, help = 'Error threshold.', required = False)
    parser.add_argument('-g', action='store', dest = 'gpu', type = float, help = 'GPU memory limit (0-1).', required = False)
    parser.add_argument('-i', action='store', dest = 'in_image', help = 'Input image (only on single).', required = False)
    parser.add_argument('-o', action='store', dest = 'out_image', help = 'Output image (only on single).', required = False)
    parser.add_argument('-l', action='store', dest = 'in_model', help = 'Load U-Net model (only on single).', required = False)

    args = parser.parse_args()
    
    settings = src.util.Settings()

    fraction = 0.95
    if args.gpu is not None :
        fraction = args.gpu

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=fraction
    keras.backend.set_session(tf.Session(config=config))

    
    if args.action == 'train' :
        trainPath = settings.settings['datasets'][args.dataset]
        training(trainPath, args.dataset, args.kfold, settings.settings['model'][args.model])
    elif args.action == 'test' :
        datasetPath = settings.settings['datasets'][args.dataset]
        runSegmentation(datasetPath, args.dataset, args.kfold, settings.settings['model'][args.model])
    elif args.action == 'evaluate' :
        evaluateOutput(settings.settings['datasets'])
    elif args.action == 'augmentation' :
        trainPath = settings.settings['datasets'][args.dataset]
        data_augmentation(trainPath, n_variations = args.augmentation)
    elif args.action == 'single' :
        runSingleSegmentation(args.in_image, args.in_model, args.out_image)


    
    
