"""
utils/detector.py
------------------
YOLOv8 modelini saran, tespit islemlerini standart bir arayuz uzerinden
sunan yardimci sinif. Uygulamanin geri kalani dogrudan ultralytics'e
bagimli olmak yerine bu sinifi kullanir; boylece model degistirmek ya da
farkli bir backend'e gecmek istersek sadece burasi degisir.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Tek bir tespit sonucunu temsil eden veri sinifi."""
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple  # (x1, y1, x2, y2)
    track_id: Optional[int] = None


@dataclass
class FrameResult:
    """Bir kare/goruntu uzerindeki tum tespit sonuclari."""
    annotated_image: np.ndarray
    detections: List[Detection] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.detections)

    def counts_by_class(self) -> dict:
        counts = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts


class ObjectDetector:
    """YOLOv8 tabanli nesne tespit / takip sarmalayicisi."""

    AVAILABLE_MODELS = {
        "Hizli (nano)": "yolov8n.pt",
        "Dengeli (small)": "yolov8s.pt",
        "Hassas (medium)": "yolov8m.pt",
    }

    def __init__(self, model_key: str = "Hizli (nano)"):
        self.model_key = model_key
        self.model_path = self.AVAILABLE_MODELS.get(model_key, "yolov8n.pt")
        self._model: Optional[YOLO] = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def set_model(self, model_key: str):
        """Farkli bir model boyutuna gecis yapar (lazy-load)."""
        if model_key != self.model_key:
            self.model_key = model_key
            self.model_path = self.AVAILABLE_MODELS.get(model_key, "yolov8n.pt")
            self._model = None  # yeniden yuklenecek

    def detect(self, image: np.ndarray, conf: float = 0.4,
               classes: Optional[List[int]] = None,
               imgsz: int = 640, augment: bool = False) -> FrameResult:
        """Tek bir goruntu/kare uzerinde tespit yapar (takip ID'siz).

        imgsz: modele verilen goruntunun ic cozunurlugu. Varsayilan 640;
               960-1280 gibi daha yuksek degerler kucuk/uzak nesneleri
               yakalamada belirgin sekilde daha iyi sonuc verir (yavaslama
               pahasina).
        augment: True ise test-time augmentation (TTA) uygular - goruntuyu
               birden fazla varyantta (flip, olcek) calistirip sonuclari
               birlestirir; hassasiyeti artirir, suredeni ~2-3x uzatir.
        """
        model = self._ensure_loaded()
        results = model.predict(
            source=image, conf=conf, classes=classes,
            imgsz=imgsz, augment=augment, verbose=False,
        )
        return self._to_frame_result(results[0], model)

    def track(self, image: np.ndarray, conf: float = 0.4,
              classes: Optional[List[int]] = None,
              persist: bool = True, imgsz: int = 640) -> FrameResult:
        """Video/webcam akisinda kare uzerinde takip ID'si ile birlikte tespit yapar."""
        model = self._ensure_loaded()
        results = model.track(
            source=image, conf=conf, classes=classes,
            persist=persist, tracker="bytetrack.yaml", imgsz=imgsz, verbose=False,
        )
        return self._to_frame_result(results[0], model)

    @staticmethod
    def _to_frame_result(result, model) -> FrameResult:
        annotated = result.plot()
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else None
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=model.names[cls_id],
                        confidence=float(box.conf[0]),
                        box_xyxy=tuple(box.xyxy[0].tolist()),
                        track_id=track_id,
                    )
                )
        return FrameResult(annotated_image=annotated, detections=detections)

    @property
    def class_names(self) -> dict:
        return self._ensure_loaded().names
