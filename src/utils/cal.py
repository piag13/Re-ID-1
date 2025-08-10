import logging
import torch
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine

class BoxUtils:
    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denominator = boxAArea + boxBArea - interArea
        return interArea / denominator if denominator > 0 else 0

    @staticmethod
    def get_centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

class PersonGallery:
    def __init__(self, threshold=0.812, iou_threshold=0.46, temp_frames=10, update_alpha=0.3):
        self.gallery = {}  # {id: embedding}
        self.temp_gallery = {}  # {temp_id: {'frames': [], 'bbox': (), 'centroid': (), 'last_seen': int}}
        self.next_id = 0
        self.next_temp_id = 10000
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.temp_frames = temp_frames
        self.update_alpha = update_alpha

    def add_person(self, avg_embedding):
        person_id = self.next_id
        self.gallery[person_id] = avg_embedding
        self.next_id += 1
        return person_id

    def get_gallery(self):
        return self.gallery

    def match_person(self, embedding):
        for person_id, stored_embedding in self.gallery.items():
            similarity = 1 - cosine(embedding, stored_embedding)
            if similarity > self.threshold:
                return person_id
        return None

    def find_matching_temp_id(self, bbox):
        for temp_id, data in self.temp_gallery.items():
            if BoxUtils.iou(bbox, data['bbox']) > self.iou_threshold:
                return temp_id
        return None

    def get_next_temp_id(self):
        temp_id = self.next_temp_id
        self.next_temp_id += 1
        return temp_id

    def store_temp_embedding(self, temp_id, embedding, bbox, frame_idx):
        if temp_id not in self.temp_gallery:
            self.temp_gallery[temp_id] = {
                "frames": [],
                "bbox": bbox,
                "centroid": BoxUtils.get_centroid(bbox),
                "last_seen": frame_idx
            }

        self.temp_gallery[temp_id]["frames"].append(embedding)
        self.temp_gallery[temp_id]["bbox"] = bbox
        self.temp_gallery[temp_id]["centroid"] = BoxUtils.get_centroid(bbox)
        self.temp_gallery[temp_id]["last_seen"] = frame_idx

        if len(self.temp_gallery[temp_id]["frames"]) >= self.temp_frames:
            avg_embedding = np.mean(self.temp_gallery[temp_id]["frames"], axis=0)
            match_id = self.match_person(avg_embedding)
            if match_id is not None:
                self.update_embedding(match_id, avg_embedding)
                del self.temp_gallery[temp_id]
                return match_id

            person_id = self.add_person(avg_embedding)
            del self.temp_gallery[temp_id]
            return person_id
        return None

    def update_embedding(self, person_id, new_embedding):
        if person_id not in self.gallery:
            logging.warning(f"Person ID {person_id} not found in gallery.")
            return
        old_embedding = self.gallery[person_id]
        self.gallery[person_id] = (1 - self.update_alpha) * old_embedding + self.update_alpha * new_embedding

    def identify_or_register(self, bbox, embedding, frame_idx, active_ids):
        similarities = [
            (person_id, 1 - cosine(embedding, stored_embedding))
            for person_id, stored_embedding in self.gallery.items()
            if 1 - cosine(embedding, stored_embedding) > self.threshold
        ]

        if similarities:
            similarities.sort(key=lambda x: x[1], reverse=True)
            for person_id, _ in similarities:
                if person_id not in active_ids:
                    self.update_embedding(person_id, embedding)
                    return person_id
            best_id = similarities[0][0]
            self.update_embedding(best_id, embedding)
            return best_id

        temp_id = self.find_matching_temp_id(bbox) or self.get_next_temp_id()
        result = self.store_temp_embedding(temp_id, embedding, bbox, frame_idx)
        return result if result is not None else temp_id

class Tracklet:
    def __init__(self, track_id, bbox, embedding, standing_threshold=5):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding
        self.frames_since_seen = 0
        self.matched_in_frame = True
        self.prev_centroid = BoxUtils.get_centroid(bbox)
        self.standing_threshold = standing_threshold

    def is_standing(self, new_bbox):
        new_centroid = BoxUtils.get_centroid(new_bbox)
        dx = abs(new_centroid[0] - self.prev_centroid[0])
        dy = abs(new_centroid[1] - self.prev_centroid[1])
        return dx < self.standing_threshold and dy < self.standing_threshold

    def update(self, new_bbox, new_embedding=None):
        if new_embedding is not None:
            self.embedding = new_embedding
        self.bbox = new_bbox
        self.prev_centroid = BoxUtils.get_centroid(new_bbox)
        self.frames_since_seen = 0
        self.matched_in_frame = True

class TrackletManager:
    def __init__(self, max_age=30, sim_threshold=0.7, iou_threshold=0.4, high_sim_iou=0.2, sim_weight=0.8):
        self.tracklets = []
        self.next_id = 0
        self.max_age = max_age
        self.sim_threshold = sim_threshold
        self.iou_threshold = iou_threshold
        self.high_sim_iou = high_sim_iou
        self.sim_weight = sim_weight

    def match(self, detections):
        for tracklet in self.tracklets:
            tracklet.matched_in_frame = False

        for bbox, embedding in detections:
            best_match = None
            best_score = -1

            for tracklet in self.tracklets:
                similarity = 1 - cosine(embedding, tracklet.embedding)
                spatial_iou = BoxUtils.iou(bbox, tracklet.bbox)

                if similarity > self.sim_threshold:
                    iou_required = self.high_sim_iou if similarity > 0.85 else self.iou_threshold
                    if spatial_iou > iou_required:
                        match_score = self.sim_weight * similarity + (1 - self.sim_weight) * spatial_iou
                        if match_score > best_score:
                            best_score = match_score
                            best_match = tracklet

            if best_match:
                new_embedding = embedding if not best_match.is_standing(bbox) else None
                best_match.update(bbox, new_embedding)
            else:
                new_tracklet = Tracklet(self.next_id, bbox, embedding)
                self.tracklets.append(new_tracklet)
                self.next_id += 1

        self.tracklets = [t for t in self.tracklets if t.matched_in_frame or t.frames_since_seen < self.max_age]
        for t in self.tracklets:
            if not t.matched_in_frame:
                t.frames_since_seen += 1

    def get_active_tracks(self):
        return [(t.track_id, t.bbox) for t in self.tracklets if t.matched_in_frame]

class EmbeddingExtractor:
    def __init__(self, transform, device=None, resize_size=(224, 224)):
        self.transform = transform
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.resize_size = resize_size

    def _preprocess_image(self, person_crop):
        if not isinstance(person_crop, np.ndarray):
            person_crop = np.array(person_crop)
        if person_crop is None or person_crop.size == 0 or person_crop.shape[2] != 3:
            raise ValueError("Invalid person_crop: must be a non-empty 3-channel BGR image.")
        img_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize(self.resize_size)
        return self.transform(img_pil).unsqueeze(0).to(self.device)

    def _extract(self, model, img_tensor):
        model = model.to(self.device).eval()
        with torch.no_grad():
            feat = model(img_tensor).squeeze().cpu().numpy()
        return feat / (np.linalg.norm(feat) + 1e-8)

    def extract_embedding(self, model, person_crop):
        img_tensor = self._preprocess_image(person_crop)
        return self._extract(model, img_tensor)

    def extract_combined_embedding(self, body_model, face_model, person_crop):
        img_tensor = self._preprocess_image(person_crop)
        body_feat = self._extract(body_model, img_tensor)
        face_feat = self._extract(face_model, img_tensor)
        return np.concatenate((body_feat, face_feat))