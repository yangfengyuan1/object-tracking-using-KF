import os
import sys
# import random
import math
import numpy as np
# import skimage.io
# import matplotlib
# import matplotlib.pyplot as plt
import argparse,configparser
import cv2
import json
import csv

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

OUTPUT_TXT_DIR = os.path.join(ROOT_DIR, "outputTXT")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

def Cal3dBBox( boxes, masks, class_ids, scores, vp):
    N=boxes.shape[0]
    ret=[]
    if not N:
        return ret
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i in range(N):
        class_id=class_ids[i]
        #if class_id not in [2,3,4,6,7,8]:
        #    continue
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        now=dict()
        now['box']=boxes[i]
        now['class_id']=class_id
        now['class_name']=class_names[class_id]
        now['score']=scores[i]
        y1, x1, y2, x2 = boxes[i]
        maskvec=[[[y-v[1],x-v[0]] for x in range(x1,x2) for y in range(y1,y2) if masks[y][x][i]] for v in vp]
        
        def CMPF(x,y):
            return math.atan2(x[1],x[0])-math.atan2(y[1],y[0])
        def CMPF1(x,y):
            return math.atan2(x[1],-x[0])-math.atan2(y[1],-y[0])
        
        def lineIntersection(a,b,c,d):
            a,b,c,d=np.array(a),np.array(b),np.array(c),np.array(d)
            denominator=np.cross(b-a,d-c)
            if abs(denominator)<1e-6:
                return False
            x=a+(b-a)*(np.cross(c-a,d-c)/denominator)
            return x

        from functools import cmp_to_key

        for j in range(2):
            maskvec[j].sort(key=cmp_to_key(CMPF))
        maskvec[2].sort(key=cmp_to_key(CMPF1))

        maskvec=np.array(maskvec)
        vp=np.array(vp)
        edg=[[maskvec[i][0][::-1],maskvec[i][-1][::-1]] if abs(math.atan2(maskvec[i][0][1],maskvec[i][0][0]))<abs(math.atan2(maskvec[i][-1][1],maskvec[i][-1][0])) else [maskvec[i][-1][::-1],maskvec[i][0][::-1]] for i in range(2)]
        tmp=[maskvec[2][0][::-1],maskvec[2][-1][::-1]] if abs(math.atan2(maskvec[2][0][1],-maskvec[2][0][0]))<abs(math.atan2(maskvec[2][-1][1],-maskvec[2][-1][0])) else [maskvec[2][-1][::-1],maskvec[2][0][::-1]] 
        edg.append(tmp)

        if edg[0][0][0]*edg[0][-1][0]<0:
            cross1=lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][0])
            cross2=lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][1])
            if cross1[0]>cross2[0]:
                cross1,cross2=cross2,cross1
            cross5=lineIntersection(vp[0], vp[0]+edg[0][0], vp[1], vp[1]+edg[1][1])
            cross6=lineIntersection(vp[0], vp[0]+edg[0][1], vp[1], vp[1]+edg[1][1])
            if cross5[0]>cross6[0]:
                cross5,cross6=cross6,cross5
            cross3=lineIntersection(vp[0], cross1, vp[2], cross5)
            cross4=lineIntersection(vp[0], cross2, vp[2], cross6)
            
            cross7=lineIntersection(vp[0], cross5, vp[2], cross1)
            cross8=lineIntersection(vp[0], cross6, vp[2], cross2)
            
            cross3,cross4=cross4,cross3
            cross5,cross6,cross7,cross8=cross7,cross8,cross6,cross5
            
            # To make it beautiful
            # tmp=cross8-cross4
            # cross5=cross1+tmp
            # cross6=cross2+tmp
            # cross7=cross3+tmp
            # cross8=cross4+tmp
            
        elif edg[1][0][0]*edg[1][-1][0]<0:
            cross1=lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][0])
            cross2=lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][1])
            if cross1[0]>cross2[0]:
                cross1,cross2=cross2,cross1
            cross5=lineIntersection(vp[1], vp[1]+edg[1][0], vp[0], vp[0]+edg[0][1])
            cross6=lineIntersection(vp[1], vp[1]+edg[1][1], vp[0], vp[0]+edg[0][1])
            if cross5[0]>cross6[0]:
                cross5,cross6=cross6,cross5
            cross3=lineIntersection(vp[1], cross1, vp[2], cross5)
            cross4=lineIntersection(vp[1], cross2, vp[2], cross6)
            
            cross7=lineIntersection(vp[1], cross5, vp[2], cross1)
            cross8=lineIntersection(vp[1], cross6, vp[2], cross2)
            
            cross3,cross4=cross4,cross3
            cross5,cross6,cross7,cross8=cross7,cross8,cross6,cross5
            
            # To make it beautiful
            # tmp=cross8-cross4
            # cross5=cross1+tmp
            # cross6=cross2+tmp
            # cross7=cross3+tmp
            # cross8=cross4+tmp
            
        else:
            cross1=lineIntersection(vp[0], vp[0]+edg[0][0], vp[1], vp[1]+edg[1][0])
            tmp1=lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][0])
            tmp2=lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][0])
            cross2=tmp1 if tmp1[1]<tmp2[1] else tmp2
            tmp1=lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][1])
            tmp2=lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][1])
            cross3=tmp1 if tmp1[1]<tmp2[1] else tmp2

            if type(lineIntersection(vp[0], cross1, vp[0], cross2))==bool:
                cross4=lineIntersection(vp[0], cross3, vp[1], cross2)
                cross7=lineIntersection(vp[0], vp[0]+edg[0][1], vp[2], cross3)
                cross6=lineIntersection(vp[1], vp[1]+edg[1][1], vp[2], cross2)
                cross8=lineIntersection(vp[0], cross7, vp[1], cross6)
                cross5=lineIntersection(vp[0], cross6, vp[1], cross7)
                # cross8=lineIntersection(vp[0], vp[0]+edg[0][1], vp[1], vp[1]+edg[1][1])
                # cross6=lineIntersection(vp[1], cross8, vp[2], cross2)
                # cross7=lineIntersection(vp[0], cross8, vp[2], cross3)
                # cross5=lineIntersection(vp[0], cross6, vp[1], cross7)
            else:
                cross4=lineIntersection(vp[0], cross2, vp[1], cross3)
                cross7=lineIntersection(vp[1], vp[1]+edg[1][1], vp[2], cross3)
                cross6=lineIntersection(vp[0], vp[0]+edg[0][1], vp[2], cross2)
                cross5=lineIntersection(vp[0], cross7, vp[1], cross6)
                cross8=lineIntersection(vp[0], cross6, vp[1], cross7)
                # cross8=lineIntersection(vp[0], vp[0]+edg[0][1], vp[1], vp[1]+edg[1][1])
                # cross6=lineIntersection(vp[0], cross8, vp[2], cross2)
                # cross7=lineIntersection(vp[1], cross8, vp[2], cross3)
                # cross5=lineIntersection(vp[0], cross7, vp[1], cross6)
            
            
            cross3,cross4=cross4,cross3
            cross7,cross8=cross8,cross7
            
            # To make it beautiful
            # tmp=cross8-cross4
            # cross5=cross1+tmp
            # cross6=cross2+tmp
            # cross7=cross3+tmp
            # cross8=cross4+tmp

        assert type(cross1)==np.ndarray and type(cross2)==np.ndarray and type(cross3)==np.ndarray and type(cross4)==np.ndarray
        assert type(cross5)==np.ndarray and type(cross6)==np.ndarray and type(cross7)==np.ndarray and type(cross8)==np.ndarray
        now['bottom']=np.array([cross1,cross2,cross3,cross4]).reshape([-1,2])
        now['roof']=np.array([cross5,cross6,cross7,cross8]).reshape([-1,2])
        ret.append(now)

    return ret



def save_3d_bbox_as_txt(results, output_path):
    with open(output_path, 'w') as f:
        for frame_idx, frame_results in enumerate(results):
            f.write(f"Frame {frame_idx}:\n")
            for result in frame_results:
                if np.any(result['bottom'] < 0) or np.any(result['roof'] < 0):
                    continue
                f.write(f"Class: {result['class_name']}, Class ID: {result['class_id']}, Score: {result['score']:.2f}, ")
                for i in range(4):
                    f.write(f"Bottom-{i+1}: {result['bottom'][i][0]:.2f}, {result['bottom'][i][1]:.2f}, ")
                for i in range(4):
                    f.write(f"Roof-{i+1}: {result['roof'][i][0]:.2f}, {result['roof'][i][1]:.2f}, ")
                f.write("\n")

def save_2dbbox_as_txt(results, outputTXT_path):
    with open(outputTXT_path, 'w') as f:
        for frame_idx, frame_results in enumerate(results):
            # f.write(f"Frame {frame_idx}:\n")
            for result in frame_results:
                if np.any(result['bottom'] < 0) or np.any(result['roof'] < 0):
                    continue
                # f.write(f"0 ")
                f.write(f"0 {result['box'][1]} {result['box'][0]} {result['box'][3]} {result['box'][2]}")
                f.write("\n")
                # f.write(f"Class: {result['class_name']}, Class ID: {result['class_id']}, Score: {result['score']:.2f}, ")



"""

def save_mrcnn_results_as_json(results, output_path):
    data = []
    for frame_idx, frame_results in enumerate(results):
        frame_data = []
        for result in frame_results:
            class_id = int(result['class_id'])  # Convert class_id to integer
            class_name = class_names[class_id]
            score = result['score']
            mask = result['masks'].tolist()
            bbox = result['rois'].tolist()
            item_data = {
                'class_name': class_name,
                'class_id': class_id,
                'score': score,
                'mask': mask,
                'bbox': bbox
            }
            frame_data.append(item_data)
        data.append(frame_data)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def save_mrcnn_results_as_txt(results, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Class', 'Class_ID', 'Score', 'Mask', 'Bbox'])
        for frame_idx, frame_results in enumerate(results):
            for result in frame_results:
                #if 'class_name' not in result or 'class_id' not in result or 'score' not in result or 'masks' not in result or 'rois' not in result:
                #    continue  
                class_name = result['class_name']
                class_id = result['class_id']
                score = result['score']
                mask = result['masks'].tolist()
                bbox = result['rois'].tolist()
                writer.writerow([frame_idx, class_name, class_id, score, mask, bbox])
"""


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='source(can be image, video, folder)')
    parser.add_argument('--config',type=str,help='Config path for roud calibration',default='config')

    args = parser.parse_args()

    cap=None
    image=None
    source=args.source
    if os.path.exists(source):
        if source.endswith(('mp4','avi')):
            cap=cv2.VideoCapture(source)
        elif source.endswith(('jpg','png','bmp')):
            image=cv2.imread(source)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        else:
            print('Bad file format.')
            exit(0)
    else:
        print('Source path not exists.')
        exit(0)

    configPath=args.config
    if not os.path.exists(configPath):
        print('Road config file not exists.')
        exit(0)
    confRoad=configparser.ConfigParser()
    confRoad.read(configPath,encoding='utf-8')
    vp=[eval(confRoad.get('vps','vp{}'.format(i))) for i in range(1,4)]
    #pp=eval(confRoad.get('info','pp'))
    #focal=eval(confRoad.get('info','focal'))
    #P=eval(confRoad.get('info','P'))

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    if cap is not None:
        ext=os.path.splitext(os.path.basename(source))
        savePath=os.path.join(OUTPUT_DIR,'{}-3D.mp4'.format(ext[0]))
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)

        out=None
        id=0

        rets=[]

        while cap.isOpened():
            ret,frame=cap.read()
            if not ret:
                break
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results = model.detect([frame], verbose=0)
            r = results[0]
            frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            ret=Cal3dBBox(r['rois'],r['masks'],r['class_ids'],r['scores'],vp)
            rets.append(ret)
            testRet = []
            testRet.append(ret)
            for item in ret:
                if np.any(item['bottom']<0) or np.any(item['roof']<0):
                    continue
                for i in range(4):
                    cv2.line(frame, tuple(item['bottom'][i].astype(int)), tuple(item['bottom'][(i+1)%4].astype(int)), (0,255,255), thickness=2)
                        
                for i in range(4):
                    cv2.line(frame, tuple(item['roof'][i].astype(int)), tuple(item['roof'][(i+1)%4].astype(int)), (0,255,255), thickness=2)
                    
                for i in range(4):
                    cv2.line(frame, tuple(item['bottom'][i].astype(int)), tuple(item['roof'][i].astype(int)), (0,255,255), thickness=2)
                        
            if not out:
                out = cv2.VideoWriter(savePath, fourcc, fps, (width,height))

            out.write(frame)
            print('frame {} finished.'.format(id))
            id+=1
            save_2dbbox_path = os.path.join(OUTPUT_TXT_DIR, '{}_'.format(ext[0])+str(id)+'.txt')
            save_2dbbox_as_txt(np.array(testRet,dtype=object).tolist(), save_2dbbox_path)
        print('savePath=',savePath)
        rets=np.array(rets,dtype=object)
        # np.save(os.path.join(OUTPUT_DIR,'{}-3D.npy'.format(ext[0])),rets)
        save_path = os.path.join(OUTPUT_DIR, '{}-3D.txt'.format(ext[0]))
        save_3d_bbox_as_txt(rets.tolist(), save_path)
        
        
        
        print('Saved to:', save_path)
        out.release()




    if image is not None:
        results = model.detect([image], verbose=0)
        r = results[0]
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        ret=Cal3dBBox(r['rois'],r['masks'],r['class_ids'],r['scores'],vp)
        for item in ret:
            if np.any(item['bottom']<0) or np.any(item['roof']<0):
                continue
            for i in range(4):
                cv2.line(image, tuple(item['bottom'][i].astype(int)), tuple(item['bottom'][(i+1)%4].astype(int)), (0,255,255), thickness=2)
                        
            for i in range(4):
                cv2.line(image, tuple(item['roof'][i].astype(int)), tuple(item['roof'][(i+1)%4].astype(int)), (0,255,255), thickness=2)
                    
            for i in range(4):
                cv2.line(image, tuple(item['bottom'][i].astype(int)), tuple(item['roof'][i].astype(int)), (0,255,255), thickness=2)
        savePath=os.path.join(OUTPUT_DIR,os.path.basename(source))
        ext=os.path.splitext(os.path.basename(source))
        # print('savePath=',savePath)
        # np.save(os.path.join(OUTPUT_DIR,'{}-3D.npy'.format(ext[0])),ret)
        save_path = os.path.join(OUTPUT_DIR, '{}-3D.txt'.format(ext[0]))
        # save_mr_path = os.path.join(OUTPUT_DIR, '{}-MRCNN_results.json'.format(ext[0]))
        # save_mr_path = os.path.join(OUTPUT_DIR, '{}-MRCNN_results.txt'.format(ext[0]))
        save_3d_bbox_as_txt([ret], save_path)
        # save_mrcnn_results_as_txt(r, save_mr_path)
        # save_mrcnn_results_as_json([r], save_mr_path)
        print('Saved cal3cbbox result to:', save_path)
        # print('Saved mrcnn result to:', save_mr_path)
        cv2.imwrite(savePath,image)
        
        
        
        
        
