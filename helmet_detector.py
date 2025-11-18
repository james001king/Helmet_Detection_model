import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import uuid
import time
from typing import Dict, Any


class HelmetDetector:
    def __init__(self):
        self.model = None
        self.model_path = "models/helmet_yolov11.pt"
        self.confidence_threshold = 0.5
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)

    def load_model(self):
        """Load YOLOv11 model safely"""
        try:
            if not os.path.exists(self.model_path):
                print("⚠️ Custom helmet model not found, loading default YOLOv11n...")
                self.model = YOLO("yolo11n.pt")
            else:
                self.model = YOLO(self.model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = YOLO("yolo11n.pt")

    def detect_image(self, image_path: str) -> Dict[str, Any]:
        """Detect helmets in an image"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image file.")

            results = self.model(image, conf=self.confidence_threshold)

            detections = []
            annotated = image.copy()

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]

                    helmet_status = self._determine_helmet_status(class_name, confidence)
                    color = (0, 255, 0) if helmet_status["has_helmet"] else (0, 0, 255)

                    # Draw bounding box
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{'Helmet' if helmet_status['has_helmet'] else 'No Helmet'}: {confidence:.2f}"
                    cv2.putText(annotated, label, (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": confidence,
                        "class": class_name,
                        "helmet_detected": helmet_status["has_helmet"]
                    })

            # Save output image
            output_name = f"detected_{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(self.output_dir, output_name)
            cv2.imwrite(output_path, annotated)

            persons_with_helmet = sum(1 for d in detections if d["helmet_detected"])
            persons_without_helmet = len(detections) - persons_with_helmet

            return {
                "success": True,
                "summary": {
                    "total_persons": len(detections),
                    "persons_with_helmet": persons_with_helmet,
                    "persons_without_helmet": persons_without_helmet,
                    "compliance_rate": (persons_with_helmet / len(detections) * 100) if detections else 0
                },
                "detections": detections,
                "output_image": f"/outputs/{output_name}"
            }


        except Exception as e:
            return {"success": False, "error": str(e)}

    def detect_video(self, video_path: str) -> Dict[str, Any]:
        """Detect helmets in a video"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file.")

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_name = f"detected_{uuid.uuid4().hex}.mp4"
            output_path = os.path.join(self.output_dir, output_name)
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            frame_count = 0
            helmet_detections = 0
            no_helmet_detections = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame, conf=self.confidence_threshold)
                annotated = frame.copy()

                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]

                        helmet_status = self._determine_helmet_status(class_name, confidence)
                        has_helmet = helmet_status["has_helmet"]
                        if has_helmet is None:
                            has_helmet = False
                        
                        if has_helmet:
                            helmet_detections += 1
                        else:
                            no_helmet_detections += 1
                        
                        color = (0, 255, 0) if has_helmet else (0, 0, 255)
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{'Helmet' if has_helmet else 'No Helmet'}: {confidence:.2f}"
                        cv2.putText(annotated, label, (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                out.write(annotated)
                frame_count += 1

            cap.release()
            out.release()

            total_detections = helmet_detections + no_helmet_detections
            compliance_rate = (helmet_detections / total_detections * 100) if total_detections > 0 else 0

            return {
    "success": True,
    "total_frames": total_frames,
    "processed_frames": total_frames,
    "output_path": f"/outputs/{output_name}",
    "summary": {
        "helmet_detections": helmet_detections,
        "no_helmet_detections": no_helmet_detections,
        "overall_compliance_rate": compliance_rate
    }
}


        except Exception as e:
            return {"success": False, "error": str(e)}

    def _determine_helmet_status(self, class_name: str, confidence: float) -> Dict[str, Any]:
        """Decide helmet status based on class name"""
        helmet_classes = ["helmet", "WITH_HELMET"]
        no_helmet_classes = ["no_helmet", "WITHOUT_HELMET"]

        if class_name in helmet_classes:
            return {"has_helmet": True, "confidence": confidence}
        elif class_name in no_helmet_classes:
            return {"has_helmet": False, "confidence": confidence}
        else:
            # Default: assume not helmet
            return {"has_helmet": False, "confidence": confidence}
