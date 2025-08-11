import os
import torch
import cv2
import logging
import yaml
from torchvision import models, transforms
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from utils.cal import EmbeddingExtractor, TrackletManager, PersonGallery


class ReIDProcessor:
    def __init__(self, config_path, video_url=None, output_url=None,
                 yolo_weights=None, threshold=0.771, iou_threshold=0.512,
                 max_age=30, device=None):
        # Logging setup
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        # Load config
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_url = video_url or self.config["url"]
        self.output_url = output_url or self.config["output_url"]
        self.yolo_weights = yolo_weights
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.transform = self._init_transform()

        # Init models & components
        self._init_models_and_components()

    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        for key in ["input_size", "device", "url", "output_url"]:
            if key not in cfg:
                raise KeyError(f"Missing key in config: {key}")
        return cfg

    def _init_transform(self):
        return transforms.Compose([
            transforms.Resize(self.config["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _init_models_and_components(self):
        self.logger.info("Initializing YOLO, ResNet50, FaceNet...")
        self.yolo_model = YOLO(self.yolo_weights)
        self.resnet_model = models.resnet50(weights='IMAGENET1K_V1').to(self.device)
        self.resnet_model.fc = torch.nn.Identity()
        self.resnet_model.eval()
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.embedding_extractor = EmbeddingExtractor(
            self.transform, self.device,
            resize_size=tuple(self.config["input_size"])
        )
        self.gallery = PersonGallery(
            threshold=self.threshold, iou_threshold=self.iou_threshold,
            temp_frames=10, update_alpha=0.3
        )
        self.tracklet_manager = TrackletManager(
            max_age=self.max_age, sim_threshold=self.threshold,
            iou_threshold=self.iou_threshold,
            high_sim_iou=0.2, sim_weight=0.8
        )

    def _process_frame(self, frame, height, frame_idx):
        results = self.yolo_model(frame)
        detections = []

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:  # Only person
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (y2 - y1) < 0.4 * height:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                emb = self.embedding_extractor.extract_combined_embedding(
                    self.resnet_model, self.facenet_model, crop
                )
                detections.append(((x1, y1, x2, y2), emb))

        # Match & update tracks
        self.tracklet_manager.match(detections)
        active_tracks = self.tracklet_manager.get_active_tracks()
        active_ids = []

        for track_id, bbox in active_tracks:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            emb = self.embedding_extractor.extract_combined_embedding(
                self.resnet_model, self.facenet_model, crop
            )
            pid = self.gallery.identify_or_register(bbox, emb, frame_idx, active_ids)
            active_ids.append(pid)
            label = f"T{pid % 10000}" if pid >= 10000 else f"ID: {pid}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    def run(self):
        video_src = int(self.video_url) if str(self.video_url).isdigit() else self.video_url
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {self.video_url}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(self.output_url), exist_ok=True)
        out = cv2.VideoWriter(self.output_url,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (1280, 720))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self._process_frame(frame, height, frame_idx)
            out.write(cv2.resize(frame, (1280, 720)))
            cv2.imshow("Re-ID Tracking", frame)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(self.device)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.logger.info(f"Done: {frame_idx} frames processed")
