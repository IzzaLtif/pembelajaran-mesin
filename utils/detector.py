"""
detector.py
Wrapper untuk model YOLO11 (Ultralytics) yang digunakan untuk mendeteksi
kepatuhan Alat Pelindung Diri (APD): helm (helmet) dan rompi (vest).

Dataset acuan (Roboflow "Construction PPE") umumnya punya kelas semacam:
    Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person, Mask, NO-Mask, dst.

Kelas yang namanya diawali "NO-" / "No-" / "no_" akan otomatis dianggap
sebagai PELANGGARAN oleh sistem ini.
"""

from ultralytics import YOLO
import numpy as np
import time


class Detection:
    """Representasi satu hasil deteksi (satu bounding box)."""

    def __init__(self, cls_id, cls_name, conf, xyxy):
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.conf = conf
        self.xyxy = xyxy  # (x1, y1, x2, y2)

    @property
    def is_violation(self) -> bool:
        name = self.cls_name.lower()
        return name.startswith("no-") or name.startswith("no_") or name.startswith("no ")


class PPEDetector:
    """Bungkus model YOLO11 + utilitas untuk aplikasi Streamlit."""

    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # dict {id: name}

    def set_confidence(self, conf: float):
        self.conf_threshold = conf

    def predict(self, frame: np.ndarray):
        """Jalankan inferensi pada satu frame (numpy array BGR)."""
        t0 = time.time()
        results = self.model.predict(
            frame, conf=self.conf_threshold, verbose=False
        )[0]
        infer_ms = (time.time() - t0) * 1000

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            detections.append(Detection(cls_id, cls_name, conf, xyxy))

        annotated_frame = results.plot()  # BGR numpy array dgn bbox tergambar
        return detections, annotated_frame, infer_ms

    def summarize(self, detections):
        """Hitung ringkasan: jumlah orang, jumlah patuh, jumlah melanggar."""
        violations = [d for d in detections if d.is_violation]
        compliant = [d for d in detections if not d.is_violation and d.cls_name.lower() != "person"]
        persons = [d for d in detections if d.cls_name.lower() == "person"]
        return {
            "persons": len(persons),
            "compliant_items": len(compliant),
            "violation_items": len(violations),
            "violation_labels": [v.cls_name for v in violations],
        }
