import torchvision.transforms as transforms
import torchvision
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image) # get the predictions on the image
    # print the results individually
    # print(f"BOXES: {outputs[0]['boxes']}")
    # print(f"LABELS: {outputs[0]['labels']}")
    # print(f"SCORES: {outputs[0]['scores']}")
    # get all the predicited class names
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes, pred_classes, outputs[0]['labels']

def draw_boxes(boxes, classes, labels, image, zone):
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        if(classes[i] != "person"):
            continue
        color = (0, 255, 255) if calculate_intersection(zone, [int(box[0]), int(box[1]), int(box[2]), int(box[3])]) < 0.5 else (0, 0, 255)
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, 
                    lineType=cv2.LINE_AA)
    return image

# Calculates intersection of boxes over surface of zone
def calculate_intersection(zone, box2):
    x_overlap = max(0, min(zone[3], box2[3]) - max(zone[1], box2[1]))
    y_overlap = max(0, min(zone[2], box2[2]) - max(zone[0], box2[0]))
    overlap_area = x_overlap * y_overlap
    pred_box_area = (box2[3]-box2[1]) * (box2[2]-box2[0])
    return overlap_area/pred_box_area

# Returns coords of zone bounding boxe
def getZoneBbox(pts):
    y = pts[:, 0]
    x = pts[:, 1]

    return [min(y), min(x), max(y), max(x)]

if __name__ == "__main__":
    
    # download or load the model from disk
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=500)

    # Setting up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    # Start reading the video
    cap = cv2.VideoCapture("./videos/people_walking.mp4")
    ret, frame = cap.read()

    # Initialize video writer
    out = cv2.VideoWriter('./results/output_rcnn.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

    # zone coords
    zone = np.array([(742, 304), (1163, 298), (1250, 484), (742, 484)], np.int32)

    zone_bbox = getZoneBbox(zone)

    f = 1
    while(True):
        ret, frame = cap.read()

        # Process 1 frame out of 5
        if(f == 10):
            f = 1
            continue
        else:
            f += 1

        # Draw zone
        cv2.polylines(frame, [zone],True,(0,255,255), 2)

        # Inference
        image = Image.fromarray(frame)
        with torch.no_grad():
            boxes, classes, labels = predict(image, model, device, 0.8)
        frame = draw_boxes(boxes, classes, labels, image, zone_bbox)

        cv2.imshow("frame", frame)
        out.write(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()