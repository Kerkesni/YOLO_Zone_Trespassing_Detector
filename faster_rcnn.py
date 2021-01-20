import torchvision.transforms as transforms
import torchvision
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import *
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

# Returns true if point is inside zone
def is_point_inside_area(zone, point):
    if (point[0] > zone[0] and point[0] < zone[2] and point[1] > zone[1] and point[1] < zone[3]) :
        return True; 
    return False; 

# draws bounding boxes on image
def draw(original_image, det_boxes, det_labels,is_zoned, zone_bbox, intersection_threshold):
    inside_zone_color = (0, 0, 255)
    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    suppress = ["person"]
    for i in range(len(det_boxes)):
        if suppress is not None:
            if not det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i]

        # Check if person is inside zone
        if is_zoned:
            inside = is_point_inside_area(zone_bbox, [box_location[2], box_location[3]]) or calculate_intersection(zone_bbox, [int(box_location[0]), int(box_location[1]), int(box_location[2]), int(box_location[3])]) > intersection_threshold

            if not inside:
                draw.rectangle(xy=[int(box_location[0]), int(box_location[1]), int(box_location[2]), int(box_location[3])], outline=label_color_map[det_labels[i]])
            else:
                draw.rectangle(xy=[int(box_location[0]), int(box_location[1]), int(box_location[2]), int(box_location[3])], outline=inside_zone_color)
        else:
                draw.rectangle(xy=[int(box_location[0]), int(box_location[1]), int(box_location[2]), int(box_location[3])], outline=label_color_map[det_labels[i]])


        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]

        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
    del draw

    return annotated_image

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
        pil_frame = draw(Image.fromarray(frame), boxes, classes, True, zone_bbox, 0.6)

        cv2.imshow("frame", np.array(pil_frame))
        out.write(np.array(pil_frame))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()