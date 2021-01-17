from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = './weights/checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Model Inference
def detect(original_image, min_score, max_overlap, top_k):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    return det_boxes, det_labels

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
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if not det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()

        # Check if person is inside zone
        if is_zoned:
            inside = is_point_inside_area(zone_bbox, [box_location[2], box_location[3]]) and calculate_intersection(zone_bbox, box_location) > intersection_threshold

            if not inside:
                draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
            else:
                draw.rectangle(xy=box_location, outline=inside_zone_color)
        else:
                draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])


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

if __name__ == '__main__':
    
    zone = np.array([[423, 313],[501, 217],[642, 181],[800, 253], [790, 355], [580, 403]], np.int32)

    zone_bbox = getZoneBbox(zone)

    cap = cv2.VideoCapture("./videos/people_walking.mp4")
    ret, frame = cap.read()

    # out = cv2.VideoWriter('./results/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

    f = 1

    while(True):
        # Capture
        ret, frame = cap.read()
        
        # Process 1 frame out of 5
        if(f == 3):
            f = 1
            continue
        else:
            f += 1

        # Inference
        pil_frame = Image.fromarray(frame)
        boxes, lables = detect(pil_frame, min_score=0.2, max_overlap=0.5, top_k=200)


        # Drawing zone
        pil_frame = np.array(pil_frame)
        cv2.polylines(pil_frame, [zone],True,(0,255,255), 2)
        pil_frame = draw(Image.fromarray(pil_frame), boxes, lables, True, zone_bbox, 0.6)
        
        cv2.imshow("window", np.array(pil_frame))
        # out.write(np.array(pil_frame))
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()
