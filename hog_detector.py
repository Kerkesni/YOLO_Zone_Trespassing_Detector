import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

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

    # Start reading the video
    cap = cv2.VideoCapture("./videos/people_walking.mp4")
    ret, frame = cap.read()

    # Initialize video writer
    out = cv2.VideoWriter('./results/output_hog.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

    # zone coords
    zone = np.array([[455, 181], [843, 172], [869, 253], [435, 261]], np.int32)

    zone_bbox = getZoneBbox(zone)

    f = 1
    while(True):
        ret, frame = cap.read()

        # Process 1 frame out of 5
        if(f == 5):
            f = 1
            continue
        else:
            f += 1

        # using a greyscale picture, also for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), scale=1)

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        pick = non_max_suppression(boxes, probs=None, overlapThresh=0.7)

        # Draw zone
        cv2.polylines(frame, [zone],True,(0,255,255), 2)

        for (xA, yA, xB, yB) in pick:
            # display the detected boxes in the colour picture
            inside = calculate_intersection(zone_bbox, [xA, yA, xB, yB]) > 0.3
            if inside:
                cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (xA, yA), (xB, yB),(255, 0, 0), 2)

        cv2.imshow("frame", frame)
        out.write(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()