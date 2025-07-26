from scipy.spatial.distance import cosine
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

class PersonGallery:
    def __init__(self, threshold=0.812, iou_threshold=0.46):
        self.gallery = {}  # {id: embedding}
        self.temp_gallery = {}  # {temp_id: {'frames': [], 'bbox': (), 'centroid': (), 'last_seen': int}}
        self.next_id = 0
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.next_temp_id = 10000  

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

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def find_matching_temp_id(self, bbox):
        for temp_id, data in self.temp_gallery.items():
            if self.iou(bbox, data['bbox']) > self.iou_threshold:
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
                "centroid": self.get_centroid(bbox),
                "last_seen": frame_idx
            }

        self.temp_gallery[temp_id]["frames"].append(embedding)
        self.temp_gallery[temp_id]["bbox"] = bbox
        self.temp_gallery[temp_id]["centroid"] = self.get_centroid(bbox)
        self.temp_gallery[temp_id]["last_seen"] = frame_idx

        if len(self.temp_gallery[temp_id]["frames"]) >= 10:
            avg_embedding = sum(self.temp_gallery[temp_id]["frames"]) / 10

            #Try matching again before registering
            match_id = self.match_person(avg_embedding)
            if match_id is not None:
                self.update_embedding(match_id, avg_embedding)
                del self.temp_gallery[temp_id]
                return match_id

            person_id = self.add_person(avg_embedding)
            del self.temp_gallery[temp_id]
            return person_id

        return None

    def update_embedding(self, person_id, new_embedding, alpha=0.3):
        if person_id not in self.gallery:
            logging.warning(f"Person ID {person_id} not found in gallery.")
            return

        old_embedding = self.gallery[person_id]
        smoothed = (1 - alpha) * old_embedding + alpha * new_embedding
        self.gallery[person_id] = smoothed


    def identify_or_register(self, bbox, embedding, frame_idx, active_ids):
        similarities = []
        
        for person_id, stored_embedding in self.gallery.items():
            similarity = 1 - cosine(embedding, stored_embedding)
            if similarity > self.threshold:
                similarities.append((person_id, similarity))

        if similarities:
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            for person_id, sim in similarities:
                if person_id not in active_ids:
                    self.update_embedding(person_id, embedding)
                    return person_id

            # Fallback: return the most similar (even if it’s active)
            best_id = similarities[0][0]
            self.update_embedding(best_id, embedding)
            return best_id

        # No match found — handle temp gallery
        temp_id = self.find_matching_temp_id(bbox)
        if temp_id is None:
            temp_id = self.get_next_temp_id()
            
        result = self.store_temp_embedding(temp_id, embedding, bbox, frame_idx)

        if result is not None:
            return result  # Return real person_id once added to gallery
        else:
            return temp_id  # Use temp ID for now
        

class Tracklet:
    def __init__(self, track_id, bbox, embedding):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding
        self.frames_since_seen = 0
        self.matched_in_frame = True
        self.prev_centroid = self.get_centroid(bbox)
        
    def get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def is_standing(self, new_bbox, threshold=5):
        new_centroid = self.get_centroid(new_bbox)
        dx = abs(new_centroid[0] - self.prev_centroid[0])
        dy = abs(new_centroid[1] - self.prev_centroid[1])
        return dx < threshold and dy < threshold
    
class TrackletManager:
    def __init__(self, max_age=30, threshold=0.7, iou_threshold=0.4):
        self.tracklets = []
        self.next_id = 0
        self.max_age = max_age
        self.threshold = threshold
        self.iou_threshold = iou_threshold

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def match(self, detections):
    
        for tracklet in self.tracklets:
            tracklet.matched_in_frame = False

        for bbox, embedding in detections:
            best_match = None
            best_score = -1

            for tracklet in self.tracklets:
                # Calculate similarity and IoU
                similarity = 1 - cosine(embedding, tracklet.embedding)
                spatial_iou = self.iou(bbox, tracklet.bbox)

                # Adaptive threshold logic
                if similarity > self.threshold:
                    iou_required = 0.2 if similarity > 0.85 else self.iou_threshold
                    if spatial_iou > iou_required:
                        # Combine score (can tune weights if needed)
                        match_score = 0.8 * similarity + 0.2 * spatial_iou
                        if match_score > best_score:
                            best_score = match_score
                            best_match = tracklet

            if best_match:
                if not best_match.is_standing(bbox):
                    best_match.embedding = embedding  # only update if person moved
                best_match.bbox = bbox
                best_match.frames_since_seen = 0
                best_match.matched_in_frame = True
                best_match.prev_centroid = best_match.get_centroid(bbox)

            else:
                # Only create a new track if no match found
                new_tracklet = Tracklet(self.next_id, bbox, embedding)
                self.tracklets.append(new_tracklet)
                self.next_id += 1

        # Clean up unmatched tracklets
        self.tracklets = [
            t for t in self.tracklets if t.matched_in_frame or t.frames_since_seen < self.max_age
        ]
        for t in self.tracklets:
            if not t.matched_in_frame:
                t.frames_since_seen += 1

    def get_active_tracks(self):
        return [(t.track_id, t.bbox) for t in self.tracklets if t.matched_in_frame]

class Cal:
    def __init__(self, transform, device=None):
        self.transform = transform
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def extract_embedding(self, model, person_crop):
        """
        Extract embedding from a single model (body or face).
        Args:
            model: torch model (e.g., ResNet, FaceNet).
            person_crop: image crop (numpy.ndarray, BGR).
        Returns:
            Normalized embedding vector (numpy.ndarray).
        """
        if person_crop is None:
            raise ValueError("Input person_crop is None.")

        # Ensure person_crop is a numpy array
        if not isinstance(person_crop, np.ndarray):
            person_crop = np.array(person_crop)

        # Convert BGR -> RGB, resize and transform
        img_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((224, 224))
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        model = model.to(self.device).eval()
        with torch.no_grad():
            embedding = model(img_tensor)  # Forward pass

        embedding = embedding.squeeze().cpu().numpy()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize
        return embedding

    def extract_combined_embedding(self, body_model, face_model, person_crop):
        """
        Extract combined embedding from body and face models.
        Args:
            body_model: torch model for body embedding.
            face_model: torch model for face embedding.
            person_crop: image crop (numpy.ndarray, BGR).
        Returns:
            Concatenated normalized embedding (numpy.ndarray).
        """
        if person_crop is None:
            raise ValueError("Input person_crop is None.")

        # Ensure person_crop is a numpy array
        if not isinstance(person_crop, np.ndarray):
            person_crop = np.array(person_crop)

        # Convert BGR -> RGB, resize and transform
        img_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((224, 224))
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        body_model = body_model.to(self.device).eval()
        face_model = face_model.to(self.device).eval()

        with torch.no_grad():
            # Extract embeddings
            body_feat = body_model(img_tensor).squeeze()
            face_feat = face_model(img_tensor).squeeze()

        # Convert to numpy and normalize
        body_feat_np = body_feat.cpu().numpy()
        face_feat_np = face_feat.cpu().numpy()

        body_feat_np /= (np.linalg.norm(body_feat_np) + 1e-8)
        face_feat_np /= (np.linalg.norm(face_feat_np) + 1e-8)

        # Concatenate body and face embeddings
        combined_embedding = np.concatenate((body_feat_np, face_feat_np))
        return combined_embedding
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate Euclidean distance between embeddings
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Contrastive loss
        loss = label * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        return loss.mean()