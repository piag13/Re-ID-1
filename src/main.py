import argparse
from test import ReIDProcessor 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Re-Identification tracking on video")
    parser.add_argument('--config', type=str, default="D:\Re_ID\src\config\config.yaml", help='Path to configuration file')
    parser.add_argument('--video', type=str, help='Input video file path or webcam index (e.g., 0)')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--yolo-weights', type=str, default="YOLO11/weights/yolo11m.pt", help='Path to YOLO model weights')
    parser.add_argument('--threshold', type=float, help='Similarity threshold for PersonGallery')
    parser.add_argument('--iou-threshold', type=float, help='IoU threshold for PersonGallery and TrackletManager')
    parser.add_argument('--max-age', type=int, help='Max age for tracklets in TrackletManager')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device for inference (cuda or cpu)')

    args = parser.parse_args()
    processor = ReIDProcessor(
        config_path=args.config,
        video_url=args.video,
        output_url=args.output,
        yolo_weights=args.yolo_weights,
        threshold=args.threshold if args.threshold is not None else 0.771,
        iou_threshold=args.iou_threshold if args.iou_threshold is not None else 0.512,
        max_age=args.max_age if args.max_age is not None else 30,
        device=args.device
    )
    processor.run()

    