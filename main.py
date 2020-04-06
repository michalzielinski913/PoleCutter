import cv2
import numpy as np
import json
import os
import time
# Create blank white image
#open the file
inputDir=''
outputDir=''
jsonFile='Test.json'


if not os.path.isdir(outputDir) and outputDir:
    os.mkdir(outputDir)
    print("[Log] Successfully created the directory")

if not os.path.exists(jsonFile):
    print("[Warning] File "+jsonFile+" not found, exiting")
    exit()


with open(jsonFile) as f:
  data = json.load(f)

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]




for y in range(0,len(data['annotations'])):
    photo_id = data['annotations'][y]['image_id']
    for z in range(0, len(data['images'])):
        if data['images'][z]['id']== photo_id:
            image = cv2.imread(inputDir+data['images'][z]['file_name'], -1)
            break
    for x in data['annotations'][y]['segmentation']:
        if image is not None:
            points=chunks(x, 2)
            milliseconds = int(round(time.time() * 1000))
            points = np.array(points, dtype=np.int32)
            mask = np.ones(image.shape, dtype=np.uint8)
            mask.fill(255)
        # points to be cropped
        # fill the ROI into the mask
            cv2.fillPoly(mask, np.array([points]), 0)

        # The mask image
        # applying th mask to original image
            masked_image = cv2.bitwise_or(image, mask)

        # The resultant image
            cv2.imwrite(outputDir+data['images'][z]['file_name'], masked_image)
            print("[Log] Saving: "+data['images'][z]['file_name'])
        else:
            print("[Warning] File "+data['images'][z]['file_name']+" not found, skipping")
            break



