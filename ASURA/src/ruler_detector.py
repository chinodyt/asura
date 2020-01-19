import numpy as np
import cv2
import scipy
import scipy.signal
import math

import util

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def detect_regions(mask) :

    gt = util.treat_mask(mask)
    ruler = np.copy(gt[:,:,0])
    ruler = ruler.astype(np.uint8)
    if np.sum(ruler) == 0 :
        return None, None
    
    mid = (np.max(ruler) - np.min(ruler))/2
    blobs = ruler > mid
    background = ruler <= mid
    ruler[blobs] = 1
    ruler[background] = 0
    
    # You need to choose 4 or 8 for connectivity type
    connectivity = 4  
    # Perform the operation
    output = cv2.connectedComponentsWithStats(ruler, connectivity, cv2.CV_8U)
    #print output[0]
    #print np.unique(output[1], return_counts=True)
    #print output[2]
    regions = []
    areas = []
    for k in xrange(output[0]) :
        tmp = np.zeros(ruler.shape)
        cur_blob = output[1] == k
        if np.sum(np.logical_and(cur_blob,blobs)) == 0 :
            continue
        tmp[output[1] == k] = 1
        #print np.sum(output[1] == k)
        #print np.sum(ruler[output[1] == k])
        regions.append(tmp)
        areas.append(output[2][k][3])
        total_area = tmp.shape[0]*tmp.shape[1]
        #print total_area, output[2][k]
    
        #print '\t', 100.0*float(output[2][k][3])/total_area
        
        #plt.imshow(tmp)
        #plt.show()
    regions = np.array(regions)
#    plt.imshow(output[1])
#    plt.show()
    # Sorting by area
    areas = np.array(areas)
    
    sorted_index = np.argsort(areas)[::-1]
    #print sorted_index
    #print areas[sorted_index]
    return regions[sorted_index], areas[sorted_index]

def crop_background(img, gt) :
    mid = (np.max(gt) - np.min(gt))/2.0
    #print mid
    
    p = np.where(gt > mid)
    x_min = np.min(p[0])
    x_max = np.max(p[0])
    y_min = np.min(p[1])
    y_max = np.max(p[1])
    
    if False :
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(gt)
        plt.show()
    
    #print x_min, x_max
    #print y_min, y_max
    gt_cropped = np.copy(gt[x_min:x_max, y_min:y_max])
    gt_cropped[gt_cropped > mid] = 1
    gt_cropped[gt_cropped <= mid] = 0
    gt_cropped = gt_cropped.astype(np.uint8)
    
    img_cropped = np.copy(img[x_min:x_max, y_min:y_max,:])
    img_cropped[gt_cropped < mid] = [255,255,255]
    
    return img_cropped, gt_cropped

def find_lines(edge, b_min = 0, b_max = 1000, min_lines = 1, tries = 1) :
    mid = (b_min+b_max)/2
    
    lines = cv2.HoughLines(edge,1,np.pi/180,mid)
    if lines is None :
        pp = None
    else :
        pp = lines.shape
    #print b_min, b_max, mid, pp
    if b_min == mid or b_max == mid :
        return lines, tries
    if lines is None :
        return find_lines(edge, b_min, mid, tries = tries+1)
    elif lines.shape[0] < min_lines :
        return find_lines(edge, b_min, mid, tries = tries+1)
    elif lines.shape[0] > min_lines :
        return find_lines(edge, mid, b_max, tries = tries+1)
    else :
        return lines, tries

def treat_squared_ruler(img, gt) :
    half = gt.shape[1]/2
    row_sum = np.sum(gt, axis = 1)
    #print len(row_sum)
    index = np.where(row_sum < half)
    gt[index,:] = 0
    
    
    
    return crop_background(img, gt)


def crop_ruler(img, mask, max_ratio = 0.5) :
    img_ruler, gt_ruler = crop_background(img, mask)

    edge = cv2.Laplacian(gt_ruler, cv2.CV_8U, 0, 1)
    
    img_out = np.copy(img_ruler)
     
    lines, tries = find_lines(edge)
    
    theta = lines[0][0][1]
    theta_degree = np.rad2deg(theta)
    #print theta, theta_degree
    
    #return
    '''
    for item in lines:
        rho = item[0][0]
        theta = item[0][1]
        print rho, theta
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000*(-b))
        y1 = int(y0 + 10000*(a))
        x2 = int(x0 - 10000*(-b))
        y2 = int(y0 - 10000*(a))

        cv2.line(img_out,(x1,y1),(x2,y2),(255,0,0),2)
    '''
    
    #rotating image
    gt_ruler = rotate_image(gt_ruler, theta_degree-90)
    img_ruler = rotate_image(img_ruler, theta_degree-90)
    
    img_ruler, gt_ruler = crop_background(img_ruler, gt_ruler)
    
    ratio = float(img_ruler.shape[0])/img_ruler.shape[1]
    #print ratio
    if np.abs(ratio-1) <= max_ratio :
        img_ruler, gt_ruler = treat_squared_ruler(img_ruler,gt_ruler)
    
    return img_ruler, gt_ruler
    #cv2.HoughLines(gt_ruler, )
    #im2, contours, hierarchy = cv2.findContours(gt_ruler, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def get_ticks(edge) :
    lsd = cv2.createLineSegmentDetector(0)

    #Detect lines in the image
    lines = lsd.detect(edge) #Position 0 of the returned tuple are the detected lines
    
    
    
    #Getting angles in degree
    dx = lines[0][:,0,0] - lines[0][:,0,2]
    dy = lines[0][:,0,1] - lines[0][:,0,3]
    angles = np.rad2deg(np.arctan(np.divide(dy,dx))) + 180
    angles %= 180
    
    #Getting length
    width = np.sqrt(np.square(dx) + np.square(dy))
    
    lines = lines[0]
    
    mean, std = np.mean(angles), np.std(angles)
    #sns.distplot(angles)
    #plt.show()
    ret = np.ones(edge.shape)
    
    for i in xrange(len(lines)) :
        if np.abs(angles[i] - mean) > 3*std :
            continue
        #print angles[i]
        p = lines[i]
#        print p
        cv2.line(ret, (int(p[0][0]),int(p[0][1])), (int(p[0][2]),int(p[0][3])), (0,0,0), 1)
#        break
    #plt.figure(figsize= (20,5))
    #plt.imshow(ret)
    #plt.show()
    
    #print lines
    return lines[:,0], width, angles
    #Draw detected lines in the image

def get_parallels(lines, widths, angles, min_angle = 5) :
    kernel = scipy.stats.gaussian_kde(angles)
    
    mode = angles[np.argmax(kernel(angles))]
    expected = np.sum(angles*kernel.evaluate(angles))
    mean, std = np.mean(angles), np.std(angles)
    #print angles
    #print kernel.evaluate(angles)
    #print mean, expected
    
    if min_angle is None :
        parallel = np.where(np.abs(angles - mode) <= std)
    else : 
        parallel = np.where(np.abs(angles - mode) <= min_angle)
    return lines[parallel], widths[parallel], angles[parallel]

def is_collinear(p0, p1, p2) :
    area = p0[0]*(p1[1]-p2[1]) + p1[0]*(p2[1]-p0[1]) + p2[0]*(p0[1]-p1[1])
    return np.abs(area/2)

def merge_lines(lines, width, min_h = 2) :
    union_find = np.arange(len(lines))
    print len(lines)
    for i in xrange(len(lines)) :
        p = lines[i]
        for j in xrange(i+1, len(lines)) :
            if i == j :
                continue
            p1 = lines[j]
            a1 = is_collinear(p[:2], p1[:2], p1[2:])
            a2 = is_collinear(p[2:], p1[:2], p1[2:])
            if a1/width[j] < min_h or a2/width[j] < min_h:
                union_find[union_find == i] = j
                
            #break
    #    break
    sets = np.unique(union_find)
    new_lines = []
    for s in sets :
        #print s
        cur_lines = lines[union_find == s]
        cur_lines = np.reshape(cur_lines, (cur_lines.shape[0]*2, 2))
        i_max = np.argmax(cur_lines[:,1])
        i_min = np.argmin(cur_lines[:,1])
        
        new_lines.append(np.concatenate((cur_lines[i_max],cur_lines[i_min])))
    new_lines = np.array(new_lines)
    dx = new_lines[:,0] - new_lines[:,2]
    dy = new_lines[:,1] - new_lines[:,3]
    angles = np.rad2deg(np.arctan(np.divide(dy,dx))) + 180
    angles %= 180
    return new_lines, np.sqrt(np.square(dx) + np.square(dy)), angles

def get_tick_distance(img, lines, width) : 
    kernal_width = scipy.stats.gaussian_kde(width)
    #plt.title("Length")
    #sns.distplot(width)
    #plt.show()
    
    x = np.arange(img.shape[0])
    maximums = scipy.signal.argrelextrema(kernal_width(x), np.greater)
    minimums = scipy.signal.argrelextrema(kernal_width(x), np.less)
    #print maximums
    #print minimums
    t = len(minimums[0])
    lower = 0
    upper = img.shape[0]
    cur_color = 0
    dists = []
    
    ret = np.copy(img)
    for i in xrange(t) :
        indexes = np.logical_and(width >= lower, width < minimums[0][i])
        cur_lines = lines[indexes]
        if len(cur_lines) < 2 :
            continue
        
        
        ret, cur_dist = paint_ticks_and_calculate_dist(ret, cur_lines, cur_color)
        cur_color += 1
        dists.append(cur_dist)
        
        lower = minimums[0][i]
        
    indexes = width >= lower
    cur_lines = lines[indexes]
    if len(cur_lines) >= 2 :
        #sns.distplot(dists)
        ret, cur_dist = paint_ticks_and_calculate_dist(ret, cur_lines, cur_color)
        dists.append(cur_dist)

    return ret, dists

def paint_ticks_and_calculate_dist(img, lines, cur_color) :

    dists = get_lines_distance(lines)
    print len(dists)
    if len(dists) == 1 :
        cur_dist = dists[0]
    else :
        kernel_dists = scipy.stats.gaussian_kde(dists)

        mode = dists[np.argmax(kernel_dists(dists))]
        cur_dist = mode
    
    cur_dist = int(cur_dist)
    
    pen_width = 1+ img.shape[0] // 100
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    use_color = colors[cur_color%6]
    
    ruler_height = np.max([img.shape[0] // 10, 10])
    #print ruler_height
    ruler = np.full((ruler_height, img.shape[1],3), 255).astype(np.uint8)

    cv2.line(ruler, (10, 0), (10, ruler_height), use_color, pen_width)
    cv2.line(ruler, (10+cur_dist, 0), (10+cur_dist, ruler_height), use_color, pen_width)
    cv2.line(ruler, (10, ruler_height/2), (10+cur_dist, ruler_height/2), use_color, pen_width)

    
    ret = np.copy(img)
    ret = np.vstack([ret, ruler])
    ret = ret.astype(np.uint8)
    
    for p in lines :
        #print p[:2].astype(int), p[2:].astype(int), colors[cur_color]
        cv2.line(ret, tuple(p[:2].astype(int)), tuple(p[2:].astype(int)), use_color, pen_width)
    return ret, cur_dist



def get_lines_distance(lines) : 
    t = len(lines)
    dist = np.full(t, np.inf)
    
    for i in xrange(t) :
        p = lines[i]
        for j in xrange(i+1,t) :
            p1 = lines[j]
            d = distance_point_line(p[:2], p1)
            if d < 0 :
                dist[j] = np.min((dist[j], np.abs(d)))
            else :
                dist[i] = np.min((dist[i], np.abs(d)))
    dist = dist[dist != np.inf]
    return dist



def distance_point_line(point, line) :
    dx = float(line[0] - line[2])
    dy = float(line[1] - line[3])

    if dx == 0 :
        return line[0] - point[0]

    m = dy/dx
    b = line[1] - m*line[0]

    y_dash = m*point[0] + b
    
    is_left = (y_dash < point[1])^ (m < 0) 

    d = np.abs(m*point[0]-point[1]+b)/np.sqrt(np.square(m)+1)
    if not is_left :
        d *= -1
    return d



def threshold_adjuster(image, minCut = 255) :
    T = 1
    i = 0
    image = image[image<minCut]
    for i in xrange(1,254) :
        m_l = np.mean(image[image<i])
        m_h = np.mean(image[image>i])
        #print m_l, m_h
        T = (m_l + m_h)/2
        #print i, T
        if i > T :
            break
    return T

def binarize_img(img, gt) :
    
    
    ret = np.copy(img)
    
    
    ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
    #_, ret = cv2.threshold(ret, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    T = threshold_adjuster(ret)
    #T = find_threshold(ret)
    ret[ret >= T] = 255
    ret[ret < T] = 0
    

    
    #print T
    mean = np.mean(ret[ret!=255])
    std = np.std(ret[ret!=255])
    cutAt = np.max([0, mean-std])
    
    h = np.unique(ret[ret!=255], return_counts=True)
    #print h
    
    #ret = cv2.adaptiveThreshold(ret,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    
    edge = cv2.Sobel(ret, cv2.CV_8U,1,0,ksize=3)
    #edge = np.copy(ret)
    #edge = pixelize(edge)
    
    #plt.imshow(drawn_img)
    #plt.show()
    
    return edge

def treat_colors(img, gt, remove_colors) :
    ret = np.copy(img)
    if not remove_colors :
        return ret

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    channel = hls[:,:,2]
    
    kernel = scipy.stats.gaussian_kde(channel.flatten())
    
    x = np.arange(255)
    minimums = scipy.signal.argrelextrema(kernel(x), np.less)
    
    if len(minimums[0]) > 1 :
        ret[channel > minimums[0][0]] = [255,255,255]
    return ret
    

def resize_ruler_img(img, max_width = 1000) :
    w = img.shape[1]
    if w <= max_width :
        return img, 1
    ratio = float(max_width)/w
    new_img = cv2.resize(img, (0,0), fx = ratio, fy = ratio)
    return new_img, ratio

def treat_ruler(img, gt, remove_colors = False) :
    #resize here
    new_img, ratio = resize_ruler_img(img)
    new_gt, ratio = resize_ruler_img(gt)

    #ret = treat_colors(new_img, new_gt, remove_colors)
    ret = treat_colors(new_img, new_gt, False)
    
    ret = binarize_img(ret, new_gt)
    lines, width, angles = get_ticks(ret)
    lines, width, angles = get_parallels(lines, width, angles, min_angle=10)
    lines, width, angles = merge_lines(lines, width, min_h=2)
    
    lines_original = lines/ratio
    width_original = width/ratio
    ticks_img, dists = get_tick_distance(img, lines_original, width_original)
    return ticks_img, dists

def get_ruler(img, mask, index = 0, remove_colors = False) :
    regions, areas = detect_regions(mask)
    if regions is None :
        blank = np.zeros((10, img.shape[1], 3))
        return blank, blank, []        
    
    if index >= len(regions) :
        raise Exception('My exception: Index %d out of bounds [%d].' % (index, len(regions)))

    img_ruler, gt_ruler = crop_ruler(img, regions[index])


    ticks_img, dists = treat_ruler(img_ruler, gt_ruler, remove_colors)

    return img_ruler, ticks_img, dists

