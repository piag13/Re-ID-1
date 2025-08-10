import cv2
from ultralytics import YOLO
import torch
from utils.cal import PersonGallery, TrackletManager, EmbeddingExtractor
from facenet_pytorch import InceptionResnetV1
from torchvision import models, transforms
import yaml
import os
import logging

class ReIDProcessor:
    def __init__(self, config_path, video_url, output_url, yolo_weights, threshold=0.771, iou_threshold=0.512, max_age=30, device=None):
        """Initialize ReID pipeline with configuration and parameters."""
        self.config = self._load_config(config_path)
        self.device = torch.device(device if device else self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.video_url = video_url if video_url else self.config["url"]
        self.output_url = output_url if output_url else self.config["output_url"]
        self.yolo_weights = yolo_weights
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.transform = self._init_transform()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self._init_components()

    def _load_config(self, config_path):
        """Load and validate configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        required_keys = ["input_size", "device", "url", "output_url"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required config key: {key}")
        if not isinstance(config["input_size"], (list, tuple)) or len(config["input_size"]) != 2:
            raise ValueError("input_size in config must be a tuple or list of length 2")
        return config

    def _init_transform(self):
        """Initialize image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(self.config["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _init_components(self):
        """Initialize models and ReID components."""
        try:
            self.yolo_model = YOLO(self.yolo_weights)
            self.resnet_model = models.resnet50(weights='IMAGENET1K_V1').to(self.device)
            self.resnet_model.fc = torch.nn.Identity()
            self.resnet_model.eval()
            self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.embedding_extractor = EmbeddingExtractor(self.transform, self.device, resize_size=tuple(self.config["input_size"]))
            self.gallery = PersonGallery(threshold=self.threshold, iou_threshold=self.iou_threshold, temp_frames=10, update_alpha=0.3)
            self.tracklet_manager = TrackletManager(max_age=self.max_age, sim_threshold=self.threshold, iou_threshold=self.iou_threshold, high_sim_iou=0.2, sim_weight=0.8)
            self.logger.info("Models and components initialized: YOLO, ResNet50, FaceNet, PersonGallery, TrackletManager")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def run(self):
        """Run the ReID pipeline on the input video."""
        try:
            video_url = int(self.video_url) if self.video_url.isdigit() else self.video_url
            cap = cv2.VideoCapture(video_url)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video source: {video_url}")
        except Exception as e:
            self.logger.error(f"Error opening video source: {e}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(self.output_url), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_url, fourcc, fps, (1280, 720))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                self.logger.info("End of video reached")
                break

            results = self.yolo_model(frame)
            detections = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) != 0:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (y2 - y1) < 0.4 * height:
                        continue
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                        continue
                    try:
                        embedding = self.embedding_extractor.extract_combined_embedding(self.resnet_model, self.facenet_model, person_crop)
                        detections.append(((x1, y1, x2, y2), embedding))
                    except Exception as e:
                        self.logger.error(f"Embedding extraction failed in frame {frame_idx}: {e}")
                        continue

            self.tracklet_manager.match(detections)
            active_tracks = self.tracklet_manager.get_active_tracks()
            active_ids = []
            for track_id, bbox in active_tracks:
                x1, y1, x2, y2 = bbox
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                    continue
                try:
                    embedding = self.embedding_extractor.extract_combined_embedding(self.resnet_model, self.facenet_model, person_crop)
                    person_id = self.gallery.identify_or_register(bbox, embedding, frame_idx, active_ids)
                    active_ids.append(person_id)
                    display_id = f"T{person_id % 10000}" if person_id >= 10000 else f"ID: {person_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, display_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                except Exception as e:
                    self.logger.error(f"ID assignment failed in frame {frame_idx}: {e}")
                    continue

            output_frame = cv2.resize(frame, (1280, 720))
            out.write(output_frame)
            cv2.imshow("Re-ID Tracking", frame)
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.logger.info(f"Processing completed: {frame_idx} frames processed")