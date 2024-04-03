import cv2
import imutils

from tracker import Tracker
from ultralytics import YOLO

def getTracker(frame, object_tracker, object_detections, class_id):
    object_x1, object_y1, object_x2, object_y2 = None, None, None, None
    object_tracker.update(frame, object_detections)
    for track_object in object_tracker.tracks:
        object_bbox = track_object.bbox
        object_x1, object_y1, object_x2, object_y2 = object_bbox
        object_x1, object_y1, object_x2, object_y2 = int(object_x1), int(object_y1), int(object_x2), int(object_y2)

        if class_id == 0:
            cv2.rectangle(frame, (object_x1, object_y1), (object_x2, object_y2), (0, 255, 0), 2)
        if class_id == 1:
            cv2.rectangle(frame, (object_x1, object_y1), (object_x2, object_y2), (255, 0, 0), 2)
        if class_id == 2:
            cv2.rectangle(frame, (object_x1, object_y1), (object_x2, object_y2), (0, 0, 255), 2)

    return object_x1, object_y1, object_x2, object_y2

team_model = YOLO('custom_models/inter_vs_milan_yolov8x.pt')
ball_model = YOLO('custom_models/soccer_ball_yolov8x.pt')

cap = cv2.VideoCapture('Inter-Milan 5-1.mp4')
fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
output_vid = 'test.mp4'

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = int(640), int(360)
video_writter = cv2.VideoWriter(output_vid, fourcc, fps, frame_size)

ball_tracker = Tracker()
inter_tracker = Tracker()
milan_tracker = Tracker()

ball_in_inter = 0
ball_in_milan = 0
possession = ''

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=640)

    team_results = team_model.predict(frame)
    ball_results = ball_model.predict(frame)

    ball_x1, ball_y1, ball_x2, ball_y2 = None, None, None, None
    for ball_result in ball_results:
        ball_detections = []
        for r in ball_result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

            x = x2 - x1
            y = y2 - y1

            if (class_id == 0) and (x or y) < 10:
                ball_detections.append([x1, y1, x2, y2, score])

        ball_x1, ball_y1, ball_x2, ball_y2 = getTracker(frame, ball_tracker, ball_detections, 0)

    for team_result in team_results:
        inter_detections = []
        milan_detections = []
        for r in team_result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

            if score > 0.5:
                if (class_id == 1):
                    inter_detections.append([x1, y1, x2, y2, score])
                if (class_id == 2):
                    milan_detections.append([x1, y1, x2, y2, score])

        inter_x1, inter_y1, inter_x2, inter_y2 = getTracker(frame, inter_tracker, inter_detections, 1)
        milan_x1, milan_y1, milan_x2, milan_y2 = getTracker(frame, milan_tracker, milan_detections, 2)

        if (ball_x1 and ball_x2 and ball_y1 and ball_y2) and (inter_x1 and inter_x2 and inter_y1 and inter_y2) and (milan_x1 and milan_x2 and milan_y1 and milan_y2) is not None:
            if (ball_x1 >= inter_x1) and (ball_x2 <= inter_x2) and (ball_y1 >= inter_y1) and (ball_y2 <= inter_y2):
                ball_in_inter += 1
            if (ball_x1 >= milan_x1) and (ball_x2 <= milan_x2) and (ball_y1 >= milan_y1) and (ball_y2 <= milan_y2):
                ball_in_milan += 1

            total_possession = ball_in_inter + ball_in_milan
            try:
                inter_posession = int((ball_in_inter / total_possession) * 100)
                milan_posession = int((ball_in_milan / total_possession) * 100)
            except ZeroDivisionError:
                possession = f'Inter Possession 0% - 0% Milan Possession'
            else:
                possession = f'Inter Possession {inter_posession}% - {milan_posession}% Milan Possession'
        else:
            pass

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(possession, font, 0.5, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 20
    cv2.putText(frame, possession, (text_x, text_y), font, 0.5, (255, 255, 255), 2)

    video_writter.write(frame)
    cv2.imshow('Inter vs Milan', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writter.release()
cv2.destroyAllWindows()