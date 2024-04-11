import cv2
import imutils

from tracker import Tracker
from ultralytics import YOLO

def getTracker(frame, object_tracker, object_detections, class_id):
    object_tracker.update(frame, object_detections)
    for track_object in object_tracker.tracks:
        object_bbox = track_object.bbox
        object_x1, object_y1, object_x2, object_y2 = object_bbox
        object_x1, object_y1, object_x2, object_y2 = int(object_x1), int(object_y1), int(object_x2), int(object_y2)

        if class_id == 0:
            cv2.rectangle(frame, (object_x1, object_y1), (object_x2, object_y2), (0, 255, 0), 2)

        return object_x1, object_y1, object_x2, object_y2

def getIOU(bbox1, bbox2):
    if bbox1 is None or bbox2 is None:
        return 0

    # Calculate intersection area
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Check if either bounding box has zero area
    if bbox1[0] == bbox1[2] or bbox1[1] == bbox1[3] or bbox2[0] == bbox2[2] or bbox2[1] == bbox2[3]:
        return 0.0

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate Union Area
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IOU
    iou = intersection_area / union_area

    return iou

team_model = YOLO('custom_models/inter_vs_milan_yolov8x.pt')
ball_model = YOLO('custom_models/soccer_ball_yolov8x.pt')

cap = cv2.VideoCapture('Inter-Milan 5-1.mp4')
fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
output_vid = 'test_0.8.mp4'

fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_size = int(640), int(360)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_writter = cv2.VideoWriter(output_vid, fourcc, fps, frame_size)

ball_tracker = Tracker()
inter_tracker = Tracker()
milan_tracker = Tracker()

ball_in_inter = 0
ball_in_milan = 0
ball_bbox = 0
inter_bbox = 0, 0, 0, 0
milan_bbox = 0, 0, 0, 0
possession = ''
count_possession = ''

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # frame = imutils.resize(frame, width=640)

    team_results = team_model.predict(frame)
    ball_results = ball_model.predict(frame)

    highest_ball_score = 0
    best_ball_score = None
    for ball_result in ball_results:
        ball_detections = []
        for r in ball_result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

            x = x2 - x1
            y = y2 - y1

            if (class_id == 0) and ((x or y) > 15) and ((x or y) < 30):
                ball_detections.append([x1, y1, x2, y2, score])

        if len(ball_detections) == 1:
            x1, y1, x2, y2, score = ball_detections[0]
        else:
            for detection in ball_detections:
                x1, y1, x2, y2, score = detection
                if score > highest_ball_score:
                    highest_ball_score = score
                    best_ball_score = detection
            if best_ball_score is not None:
                x1, y1, x2, y2, score = best_ball_score
                best_ball_score = ball_detections

        ball_bbox = getTracker(frame, ball_tracker, ball_detections, 0)

    for team_result in team_results:
        inter_detections = []
        milan_detections = []
        for r in team_result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

            if score > 0.8:
                if class_id == 1:
                    inter_bbox = x1, y1, x2, y2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # inter_detections.append([x1, y1, x2, y2, score])
                if class_id == 2:
                    milan_bbox = x1, y1, x2, y2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # milan_detections.append([x1, y1, x2, y2, score])

        # inter_bbox = getTracker(frame, inter_tracker, inter_detections, 1)
        # milan_bbox = getTracker(frame, milan_tracker, milan_detections, 2)

        inter_iou = getIOU(ball_bbox, inter_bbox)
        milan_iou = getIOU(ball_bbox, milan_bbox)

        if inter_iou:
            ball_in_inter += 1
        if milan_iou:
            ball_in_milan += 1

        total_possession = ball_in_inter + ball_in_milan
        try:
            inter_possession = int((ball_in_inter / total_possession) * 100)
            milan_possession = int((ball_in_milan / total_possession) * 100)
            count_possession = f'Inter Touches:  {ball_in_inter}, Milan Touches: {ball_in_milan}'
        except ZeroDivisionError:
            possession = f'Inter Possession 0% - 0% Milan Possession'
        else:
            possession = f'Inter Possession {inter_possession}% - {milan_possession}% Milan Possession'

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(possession, font, 1, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 40
    cv2.putText(frame, count_possession, (text_x, text_y - 40), font, 1, (255, 255, 255), 2)
    cv2.putText(frame, possession, (text_x, text_y), font, 1, (255, 255, 255), 2)

    video_writter.write(frame)
    cv2.imshow('Inter vs Milan', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writter.release()
cv2.destroyAllWindows()