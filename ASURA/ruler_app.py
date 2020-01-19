import sys
import os
import argparse

from Tkinter import *
import Tkinter, Tkconstants

import numpy as np
import cv2

from PIL import ImageTk, Image

import src.util
import src.ruler_detector

ignore_list = []

def create_blank_image() :
    img = np.full((512, 512, 3), 0).astype(np.uint8)
    return Image.fromarray(img)

def imagetk_fromarray(img, ratio = None) :
    PILimg = Image.fromarray(img)
    newW = int(float(PILimg.size[0])*ratio)
    newH = int(float(PILimg.size[1])*ratio)
    PILimg = PILimg.resize((newW, newH), Image.ANTIALIAS)
    return ImageTk.PhotoImage(PILimg)


def load_dataset_segmentation(dataset_path, dataset_name, method) :
    global image_dataset, seg_list, max_image_index, ruler_imgs, dist_imgs, marked_imgs, real_dists, est_dists, max_ruler_index
    print dataset_path, dataset_name
    image_dataset = src.util.ImageDataset(dataset_path, dataset_name = dataset_name, k_fold = '0', training = True, ignore_list = ignore_list)
    max_image_index = image_dataset.get_size()
    
    if not os.path.isdir(ruler_path) :
        os.makedirs(ruler_path)

    if os.path.isfile(real_dists_path) :
        s = open(real_dists_path, 'r').read()
        real_dists = eval(s)

    if os.path.isfile(est_dists_path) :
        s = open(est_dists_path, 'r').read()
        est_dists = eval(s)
    print est_dists
    print real_dists

    seg_path = './output/%s/%s' % (dataset_name, method)
    print seg_path
    seg_list = []
    for i in xrange(image_dataset.get_size()) :
        print image_dataset.filenameList[i]
        if not use_gt :
            seg_filename = image_dataset.filenameList[i] + '_out.png'
            full_seg_path = os.path.join(seg_path,seg_filename)
            if not os.path.isfile(full_seg_path) :
                print 'You must run method <%s> on <%s> dataset before using this script.' % (method, dataset_name)
                exit()
            img = cv2.imread(full_seg_path, )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            seg_list.append(img)

        ruler_imgs[image_dataset.filenameList[i]] = []
        dist_imgs[image_dataset.filenameList[i]] = []
        marked_imgs[image_dataset.filenameList[i]] = []
        for j in xrange(20) :
            full_ruler_img_path = os.path.join(ruler_path, image_dataset.filenameList[i] + '_ruler_%d.jpg' %(j))
            print full_ruler_img_path
            if os.path.isfile(full_ruler_img_path) :
                img = cv2.imread(full_ruler_img_path, )
                r_img = np.copy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ruler_imgs[image_dataset.filenameList[i]].append(r_img)
            else :
                break
        
            full_dist_img_path = os.path.join(ruler_path, image_dataset.filenameList[i] + '_dist_%d.jpg' % (j))
            if os.path.isfile(full_dist_img_path) :
                img = cv2.imread(full_dist_img_path, )
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dist_imgs[image_dataset.filenameList[i]].append(img)

            full_marked_img_path = os.path.join(ruler_path, image_dataset.filenameList[i] + '_marked_%d.jpg' % (j))
            if os.path.isfile(full_marked_img_path) :
                img = cv2.imread(full_marked_img_path, )
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                marked_imgs[image_dataset.filenameList[i]].append(img)
                print 'Loaded marked'
            else :
                marked_imgs[image_dataset.filenameList[i]].append(r_img)
        max_ruler_index[image_dataset.filenameList[i]] = len(ruler_imgs[image_dataset.filenameList[i]])

    print len(ruler_imgs), len(dist_imgs), len(marked_imgs)

            

def process_all_rulers() :
    for i in xrange(image_dataset.get_size()) :
        process_image_rulers(i)
        
        
    with open(est_dists_path, 'w') as f:
        print >> f, str(est_dists)


def process_image_rulers(index = 0) :
    global ruler_imgs, dist_imgs, est_dists, max_ruler_index, marked_imgs
    curFilename = image_dataset.filenameList[index]
    
    curImg = image_dataset.imgList[index]
    if use_gt :
        curSeg = image_dataset.maskList[index]
    else :
        curSeg = seg_list[index]
    if len(ruler_imgs[curFilename]) == 0 :
        print 'Processing', curFilename
        ruler_imgs[curFilename] = []
        dist_imgs[curFilename] = []
        est_dists[curFilename] = []
        marked_imgs[curFilename] = []
        j = 0
        i = -1
        while True :
            i+= 1
            try :
                cur_ruler, cur_dist_img, cur_dists = src.ruler_detector.get_ruler(curImg, curSeg, index = i)
            except Exception as error:
                if repr(error).find('My exception') >= 0 :
                    break
                print('Caught this error: ' + repr(error))
                continue 

            ruler_imgs[curFilename].append(cur_ruler)
            dist_imgs[curFilename].append(cur_dist_img)
            est_dists[curFilename].append(cur_dists)
            

            full_ruler_img_path = os.path.join(ruler_path, curFilename + '_ruler_%d.jpg' % (j))
            img = cv2.cvtColor(cur_ruler, cv2.COLOR_RGB2BGR)
            cv2.imwrite(full_ruler_img_path, img)
            
            
            full_dist_img_path = os.path.join(ruler_path, curFilename + '_dist_%d.jpg' % (j))
            img = cv2.cvtColor(cur_dist_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(full_dist_img_path, img)
            j+=1
        max_ruler_index[curFilename] = j
        
        
def previous_image(event = None) :
    global  cur_image_index, cur_ruler_index
    cur_image_index -= 1 + max_image_index
    cur_image_index %= max_image_index
    cur_ruler_index = 0
    update_images()

def next_image(event = None) :
    global  cur_image_index, cur_ruler_index
    cur_image_index += 1
    cur_image_index %= max_image_index
    cur_ruler_index = 0
    update_images()

def previous_ruler(event = None) :
    curFilename = image_dataset.filenameList[cur_image_index]
    
    global  cur_ruler_index
    cur_ruler_index -= 1 + max_ruler_index[curFilename]
    cur_ruler_index %= max_ruler_index[curFilename]
    update_images()

def next_ruler(event = None) :
    curFilename = image_dataset.filenameList[cur_image_index]
    
    global  cur_ruler_index
    cur_ruler_index += 1
    cur_ruler_index %= max_ruler_index[curFilename]
    update_images()

def save_ruler(event = None) :
    curFilename = image_dataset.filenameList[cur_image_index]
    full_marked_img_path = os.path.join(ruler_path, curFilename + '_marked_%d.jpg' % (cur_ruler_index))
    img = cv2.cvtColor(marked_imgs[curFilename][cur_ruler_index], cv2.COLOR_BGR2RGB)
    cv2.imwrite(full_marked_img_path, img)

    with open(real_dists_path, 'w') as f:
        print >> f, str(real_dists)

def save_real_ruler_and_next_image(event = None) :
    save_ruler()

    print 'save_ruler'
    next_image()

def save_real_ruler_and_next_ruler(event = None) :
    save_ruler()

    print 'save_ruler'
    print cur_ruler_index
    next_ruler()
    print cur_ruler_index
    if cur_ruler_index == 0 :
        next_image()


def createToolbar() :
    toolbar = Frame(root)
    b = Button(toolbar, text="Previous (a)", width=8, command=previous_image)
    b.pack(side=LEFT, padx=2, pady=2)
    b2 = Button(toolbar, text="Next (d)", width=6, command=next_image)
    b2.pack(side=LEFT, padx=2, pady=2)
    b3 = Button(toolbar, text="Save and Next (s)", width=12, command=save_real_ruler_and_next_image)
    b3.pack(side=LEFT, padx=2, pady=2)
    
    b4 = Button(toolbar, text="Previous Ruler (q)", width=12, command=previous_ruler)
    b4.pack(side=LEFT, padx=2, pady=2)
    b5 = Button(toolbar, text="Next Ruler (e)", width=10, command=next_ruler)
    b5.pack(side=LEFT, padx=2, pady=2)
    b6 = Button(toolbar, text="Save and Next Ruler (w)", width=16, command=save_real_ruler_and_next_ruler)
    b6.pack(side=LEFT, padx=2, pady=2)
    

    toolbar.pack(side=TOP, fill=X)

    
def update_images(event = None) :
    curFilename = image_dataset.filenameList[cur_image_index]

    labelFilename.config(text='Filename %d/%d: %s'% (cur_image_index, max_image_index, curFilename))
   
    print est_dists
    if curFilename in real_dists :
        labelRuler.config(text = 'Ruler %d: %s' % (cur_ruler_index, str(real_dists[curFilename][cur_ruler_index])))
        
    else :
        labelRuler.config(text = 'Ruler %d:' % (cur_ruler_index))
    if curFilename in est_dists :
        labelDist.config(text = 'Distances %d: %s' % (cur_ruler_index, str(est_dists[curFilename][cur_ruler_index])))
        

    
    global showImg, showSeg, showRuler, showDist, img_ratio, ruler_ratio
    
    mainHeight, mainWidth = root.winfo_height()-20, root.winfo_width()
    imgsWidth = 2*image_dataset.imgList[cur_image_index].shape[1] + 10
    tmp_ratio_img_ruler = float(imgsWidth)/dist_imgs[curFilename][cur_ruler_index].shape[1]
    ruler_h = tmp_ratio_img_ruler*dist_imgs[curFilename][cur_ruler_index].shape[0]
    
    imgsHeight = image_dataset.imgList[cur_image_index].shape[0] + 2*ruler_h + 200
    

    ratioW = mainWidth/float(imgsWidth)
    ratioH = mainHeight/float(imgsHeight)
    
    if ratioW*imgsHeight < mainHeight :
        ratio = ratioW
    else :
        ratio = ratioH
    
    
    img_ratio = ratio
    ruler_ratio = ratio*tmp_ratio_img_ruler

    showImg = imagetk_fromarray(image_dataset.imgList[cur_image_index], img_ratio)
    panelImg.configure(image=showImg)
    if use_gt :
        showSeg = imagetk_fromarray(image_dataset.maskList[cur_image_index], img_ratio)
    else :
        showSeg = imagetk_fromarray(seg_list[cur_image_index], img_ratio)
    panelSeg.configure(image=showSeg)

    if curFilename in marked_imgs :
        showRuler = imagetk_fromarray(marked_imgs[curFilename][cur_ruler_index], ruler_ratio)
    else :
        showRuler = imagetk_fromarray(ruler_imgs[curFilename][cur_ruler_index], ruler_ratio)
    panelRuler.configure(image = showRuler)

    showDist = imagetk_fromarray(dist_imgs[curFilename][cur_ruler_index], ruler_ratio)
    panelDist.configure(image = showDist)

def changeGraphic(event) :
    pass

def mouseB1Press(event) :
    global app_x0, app_y0
    app_x0, app_y0 = event.x, event.y
    
def mouseB1Release(event) :
    global app_x1, app_y1, cur_real_dist, marked_imgs, real_dists
    app_x1, app_y1 = event.x, event.y
    curFilename = image_dataset.filenameList[cur_image_index]
    
    if curFilename not in real_dists :
        real_dists[curFilename] = [-1] * max_ruler_index[curFilename]

    cur_ruler = np.copy(ruler_imgs[curFilename][cur_ruler_index])
    x0 = int(app_x0/ruler_ratio)
    y0 = int(app_y0/ruler_ratio)
    
    x1 = int(app_x1/ruler_ratio)
    y1 = int(app_y1/ruler_ratio)
    pen_width = 1 + cur_ruler.shape[0] // 100
    
    cv2.line(cur_ruler, (x0, y0), (x1, y1), [255,0,0], pen_width)

    cur_real_dist = np.sqrt(np.square(x0-x1)+np.square(y0-y1))
    
    

    real_dists[curFilename][cur_ruler_index] = cur_real_dist
    marked_imgs[curFilename][cur_ruler_index] = cur_ruler
    update_images()


def close(event = None) :
    sys.exit()


image_dataset = None
seg_list = None
ruler_imgs = dict()
dist_imgs = dict()
marked_imgs = dict()
real_dists = dict()
est_dists = dict()
use_gt = False
cur_image_index = 0
max_image_index = 1
cur_ruler_index = 0
max_ruler_index = dict()
ruler_path = './output/ruler'
real_dists_path = './output/ruler/real_dists.txt'
est_dists_path = './output/ruler/est_dists.txt'
app_x0 = 0
app_y0 = 0
app_x1 = 0
app_y1 = 0


root = Tk()
root.geometry("810x740")
root.title('ASURA')
root.bind('s', save_real_ruler_and_next_image)
root.bind('a', previous_image)
root.bind('d', next_image)

root.bind('w', save_real_ruler_and_next_ruler)
root.bind('q', previous_ruler)
root.bind('e', next_ruler)
root.bind('<Escape>', close)

mainFrame = Frame(root)
mainFrame.bind('<Configure>', update_images)
imagesFrame = Frame(mainFrame)

blank = ImageTk.PhotoImage(create_blank_image())

labelFilename = Label(mainFrame, text = 'Filename:')

showImg = None
labelImg = Label(imagesFrame, text = 'Source:')
panelImg = Label(imagesFrame, image = blank)

showSeg = None
labelSeg = Label(imagesFrame, text = 'Segmentation:')
panelSeg = Label(imagesFrame, image = blank)

showRuler = None
labelRuler = Label(mainFrame, text = 'Ruler:')
panelRuler = Label(mainFrame, image = blank)

showDist = None
labelDist = Label(mainFrame, text = 'Distances:')
panelDist = Label(mainFrame, image = blank)

img_ratio = 1.0
ruler_ratio = 1.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'ASURA GUI')
    parser.add_argument('-d', action='store', dest = 'dataset', help = 'Dataset (see settings.json for reference)', required = True)
    parser.add_argument('-m', action='store', dest = 'model', help = 'Model (see settings.json for reference)', default = 'unet_E300', required = False)
    parser.add_argument('-gt', action='store_true', dest = 'use_gt', help = 'Use ground truth image to estimate area.', required = False)

    args = parser.parse_args()
    
    use_gt = args.use_gt
    settings = src.util.Settings()

    if args.use_gt :
        ruler_path = './output/ruler_gt'
        real_dists_path = './output/ruler_gt/real_dists.txt'
        est_dists_path = './output/ruler_gt/est_dists.txt'
    else :
        ruler_path = './output/ruler'
        real_dists_path = './output/ruler/real_dists.txt'
        est_dists_path = './output/ruler/est_dists.txt'

    load_dataset_segmentation(settings.settings['datasets'][args.dataset], args.dataset, args.model)
    process_all_rulers()

    createToolbar()
    mainFrame.pack()
    
    labelFilename.pack()

    imagesFrame.pack()
    
    labelImg.grid(row=0, column=0)
    panelImg.grid(row=0, column=0, sticky = "nesw")
    
    
    labelSeg.grid(row=0, column=1)
    panelSeg.grid(row=0, column=1, sticky = "nesw")

    labelRuler.pack()
    panelRuler.pack()

    labelDist.pack()
    panelDist.pack()

    panelRuler.bind("<ButtonPress-1>", mouseB1Press)
    panelRuler.bind("<ButtonRelease-1>", mouseB1Release)
 

    root.mainloop()
    update_images()

