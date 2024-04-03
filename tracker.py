import numpy as np

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker

class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        nn_budget = None
        max_cosine_distance = 0.4
        encoder_model_file = 'tracker_model/mars-small128.pb'
        metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)

        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_file, batch_size=1)

    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()

            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]

        scores = [d[-1] for d in detections]
        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            id = track.track_id
            bbox = track.to_tlbr()

            tracks.append(Tracks(id, bbox))

        self.tracks = tracks

class Tracks:
    bbox = None
    track_id = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox