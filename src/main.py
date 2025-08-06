import cv2
from ultralytics import YOLO
import torch
from utils.cal import PersonGallery, TrackletManager, Embedding
from facenet_pytorch import InceptionResnetV1
from torchvision import models, transforms
import yaml

yolo_model = YOLO("YOLO11\weights\yolo11m.pt")

with open("Re-ID-1\src\config\config.yaml", "r") as f:
    config = yaml.safe_load(f)

transform = transforms.Compose([
    transforms.Resize(config["input_size"]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    ),
])

resnet_model = models.resnet50(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model.to(device)
resnet_model.fc = torch.nn.Identity()
cal = Embedding(transform, device)
resnet_model.eval()
print("ResNet50 model loaded and ready for inference.")
# Load FaceNet for face embeddings
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Models loaded: ResNet for body, FaceNet for face.")

# Set up video capture
output_url = "\output_video"
# url = "rtsp://admin:Tinvl12345@@192.168.1.13:554/media/live/1/main"
url = 0
# url = "rtsp://admin:Tinvl12345@@192.168.0.149:554/media/live/1/main"
# url = "rtsp://admin:Tinvl1903@@192.168.0.103:554/media/live/1/main"
# url = "rtsp://admin:Tinvl1903@@169.254.17.253:554/media/live/1/main"
# url = r"D:\Dissertation\Re-id\sample video\footage_2.mp4"  # Use video file path or 0 for webcam
cap = cv2.VideoCapture(url)  # Use video file path or 0 for webcam
gallery = PersonGallery(threshold=0.771, iou_threshold=0.512)
tracklet_manager = TrackletManager(max_age=30, threshold=0.751, iou_threshold=0.512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up video writer for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_url + "\output_reid_4.mp4", fourcc, fps, (1280, 720))

print(device)

frame_idx = 0  # Initialize frame index

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)  # YOLO inference
    detections = []

    # Collect all detections
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            conf = box.conf[0].item()

            if class_id == 0:  # Person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)
                ## If detect small or partial detections, skip them
                #calculate bounding box dimensions
                bbox_height = y2 - y1
                bbox_width = x2 - x1
                bbox_area = bbox_height * bbox_width

                # Filter out partial or small detections
                if bbox_height < 0.4 * height:
                    continue  # skip this detection

                person_crop = frame[y1:y2, x1:x2]

                if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                    embedding = cal.extract_combined_embedding(resnet_model, facenet_model, person_crop)
                    detections.append((bbox, embedding))

    # Match tracklets
    tracklet_manager.match(detections)

    # Get fresh active tracks
    active_tracks = tracklet_manager.get_active_tracks()
    active_ids = []

    # ID assignment and display
    for track_id, bbox in active_tracks:
        x1, y1, x2, y2 = bbox
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
            embedding = cal.extract_combined_embedding(resnet_model, facenet_model, person_crop)

            person_id = gallery.identify_or_register(bbox, embedding, frame_idx, active_ids)

            if person_id is None:
                person_id = gallery.get_next_temp_id()  # prevent fallback to 0

            # Keep track of used IDs in this frame
            active_ids.append(person_id)

            # Draw results
            if person_id >= 10000:
                display_id = f"T{person_id % 10000}"  # For temp: T0, T1, ...
            else:
                display_id = f"ID: {person_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, display_id, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    output_frame = cv2.resize(frame, (1280, 720))
    out.write(output_frame)
    cv2.imshow("Re-ID Tracking", frame)
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
