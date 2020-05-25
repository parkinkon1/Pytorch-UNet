#######################################
### COCO mask image generator : pky ###
#######################################
import sys
COCOAPI_PATH = '/Users/parkkyuyeol/Desktop/DGIST_intern/cocoapi/PythonAPI'
sys.path.append(COCOAPI_PATH)
from pycocotools.coco import COCO

import numpy as np
import cv2

dataDir='/Users/parkkyuyeol/Desktop/DGIST_intern/cocodataset'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
img_infos = coco.loadImgs(imgIds)

for idx in range(20):
    img_info = img_infos[idx]
    height, width = img_info['height'], img_info['width']
    
    annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
    ann_info = coco.loadAnns(annIds)
    
    mask = np.zeros((height, width, 3), np.uint8)
    for i in range(len(ann_info)):
        if (ann_info[i]['area'] < height*width*0.01): continue
        ann = ann_info[i]
        cid = ann['category_id']
        x, y, w, h = ann['bbox']
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        mask = cv2.rectangle(mask, p1, p2, (cid, cid, cid), -1)
        
    img_dir = dataDir + '/images/' + dataType + '/' + img_info['file_name']
    img = cv2.imread(img_dir)
    
    cv2.imwrite('data/imgs/'+str(img_info['file_name']), img)
    cv2.imwrite('data/masks/'+str(img_info['file_name']), mask)
