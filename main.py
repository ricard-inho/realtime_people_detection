import torch

import numpy as np
import cv2

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

import tensorflow as tf
from funct import draw_bbox, image_preprocess

# Definition of the parameters
max_cosine_distance = 0.5
nn_budget = None
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/crowdhuman_yolov5m.pt')
store = 'rtsp://admin:Homerun1!@174.105.187.236:8000/streaming/channels/701.sdp'
input_size = 640


key_list = ["Head","Person"]
val_list = [1, 0]

CLASSES = ["Head", "Person"]

def main():
    # define a video capture object
    vid = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    #initialize deep sort object
    model_filename = 'weights/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        if not ret: 
            break

        try:
            original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = tf.expand_dims(image_data, 0)

        #Detect people
        detections = model(frame)

        img_detection = np.copy(frame)

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for x1,y1,x2,y2,conf,obj_type in detections.xyxy[0]:
            boxes.append([int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())])
            scores.append(conf.item())
            names.append(obj_type.item())
            '''
            if obj_type.item() == 0:
                img_detection = cv2.rectangle(frame, (int(x1.item()),int(y1.item())), (int(x2.item()),int(y2.item())), (255,0,0), 2)
            else:
                img_detection = cv2.rectangle(frame, (int(x1.item()),int(y1.item())), (int(x2.item()),int(y2.item())), (0,0,255), 2)
            '''

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(image_data, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[val_list.index(int(class_name))] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function


        image = draw_bbox(original_image, tracked_bboxes, CLASSES=CLASSES, tracking=True)

        # Display the resulting frame
        cv2.imshow('frame', image)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()