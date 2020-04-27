#TODO
# split code to more functions
# change output images name
# change bbox saving to cover a little more space than limited by object border points
import cv2
import numpy as np
import json
import os
import time
from shapely.geometry import Point, Polygon
#Directory with coco images
inputDir=''
#Directory where output images will be stored
outputDir='output/'
#Json file with annotations
jsonFile='test-9.json'


#Create output directory if not exist
if not os.path.isdir(outputDir) and outputDir:
    os.mkdir(outputDir)
    print("[Log] Successfully created the directory")
#Check if json file exist
if not os.path.exists(jsonFile):
    print("[Warning] File "+jsonFile+" not found, exiting")
    exit()


with open(jsonFile) as f:
  data = json.load(f)

#convert array of coordinates to array of points
#[x1, y1, x2, y2...] to [(x1,y1), (x2,y2)...]
def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

#return bbox coordinates based on annotation points
def bb_box(points):
    points = np.array(points, dtype=np.int32)
    x_values=points[0::2]
    y_values=points[1::2]
    x_1=min(x_values)
    x_2=max(x_values)
    y_1=min(y_values)
    y_2=max(y_values)
    return int(x_1), int(x_2), int(y_1), int(y_2)


#iterate through each annotation
for y in range(0,len(data['annotations'])):
    photo_id = data['annotations'][y]['image_id']
    bbox=np.array([])
    #search for image with include current annotation
    for z in range(0, len(data['images'])):
        if data['images'][z]['id']== photo_id:
            image = cv2.imread(inputDir+data['images'][z]['file_name'], -1)
            #create white mask with shape same as input image
            mask = np.ones(image.shape, dtype=np.uint8)
            mask.fill(255)
            break
    for x in data['annotations'][y]['segmentation']:
        #check if photo was found, If not skip
        if image is None:
            print("[Warning] File " + data['images'][z]['file_name'] + " not found, skipping")
            break
        else:
            #checking if annotation polygon is full or empty inside using negation
            roi = None
            for a in data['annotations'][y]['segmentation']:
                #check if we check all higher polygons in hierarchy
                if np.array_equal(x, a):
                    if roi == None:
                        roi= True
                    break
                points = chunks(a, 2)
                points = np.array(points, dtype=np.int32)
                poly = Polygon(points)
                pp=Point(x[0], x[1])
                #check if current polygon is inside higher rank polygon
                # polygon inside full polygon is empty inside
                # polygin inside empty polygon is full inside
                if (pp.within(poly)):
                    if roi == None:
                        roi= False
                    else:
                        roi = not roi
                if roi == None:
                    roi=True

            points = chunks(x, 2)
            points = np.array(points, dtype=np.int32)

            #if polygon is empty fill it with white color, otherwise black
            if roi == True:
                cv2.fillPoly(mask, np.array([points]), 0)
            else:
                cv2.fillPoly(mask, np.array([points]), (255,255,255))
            #sum all points of specific coco object
            bbox=np.concatenate((bbox, x))
    #calculate bbox coordinates
    x_1, x_2, y_1, y_2=bb_box(bbox)

    #apply mask to an image
    image=cv2.bitwise_or(image, mask)

    #save image
    cv2.imwrite(outputDir + str(time.time()) + "_" + data['images'][z]['file_name'], image[y_1:y_2, x_1:x_2])
    print("[Log] Saving: " + data['images'][z]['file_name'])




