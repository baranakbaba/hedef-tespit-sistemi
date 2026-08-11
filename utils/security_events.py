"""
utils/security_events.py
-------------------------
Davranis tabanli guvenlik olayi tespiti.

Tek bir karedeki nesne tespiti yeterli degildir; gercek guvenlik sistemleri
zaman icindeki DAVRANISA bakar. Bu modul, ObjectDetector.track() ciktisini
zaman icinde izleyerek 3 tur olay uretir:

  1. ZONE_IHLALI       -> Bir kisi tanimli bir "yasak bolge"ye girdiginde
  2. TERK_EDILMIS_NESNE -> Bir canta/valiz, yaninda kimse olmadan uzun sure
                            hareketsiz kaldiginda
  3. UZUN_SURE_BEKLEME  -> Bir kisi alanda beklenenden uzun sure kaldiginda
                            (loitering)

Bu, gercek CCTV/perimeter guvenlik urunlerinde kullanilan yaklasimin
basitlestirilmis fakat calisan bir versiyonudur.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2

ZONE_RELEVANT_PERSON_CLASS = "person"
PORTABLE_OBJECT_CLASSES = {"backpack", "suitcase", "handbag"}

SEVERITY_RENK = {
    "BILGI": "#4C9AFF",
    "UYARI": "#FFA500",
    "KRITIK": "#FF3B30",
}


@dataclass
class SecurityEvent:
    timestamp: float
    event_type: str  # "ZONE_IHLALI" | "TERK_EDILMIS_NESNE" | "UZUN_SURE_BEKLEME"
    severity: str  # "BILGI" | "UYARI" | "KRITIK"
    class_name: str
    track_id: Optional[int]
    description: str


@dataclass
class _TrackState:
    class_name: str
    first_seen: float
    last_seen: float
    positions: deque = field(default_factory=lambda: deque(maxlen=50))
    zone_event_fired: bool = False
    abandoned_event_fired: bool = False
    loiter_event_fired: bool = False


class SecurityMonitor:
    """Takip edilen nesnelerin davranisindan guvenlik olayi ureten durum makinesi."""

    def __init__(
        self,
        zone: Optional[Tuple[float, float, float, float]] = None,
        loiter_seconds: float = 8.0,
        abandoned_seconds: float = 8.0,
        abandoned_move_threshold_px: float = 25.0,
        abandoned_person_distance_px: float = 150.0,
    ):
        """
        zone: (x1, y1, x2, y2) - kare genisligine/yuksekligine gore 0-1
              arasinda normalize edilmis dikdortgen yasak bolge koordinati.
        """
        self.zone = zone
        self.loiter_seconds = loiter_seconds
        self.abandoned_seconds = abandoned_seconds
        self.abandoned_move_threshold_px = abandoned_move_threshold_px
        self.abandoned_person_distance_px = abandoned_person_distance_px
        self._tracks: Dict[int, _TrackState] = {}

    def set_zone(self, zone: Optional[Tuple[float, float, float, float]]):
        self.zone = zone

    def reset(self):
        self._tracks.clear()

    def _point_in_zone(self, cx_norm: float, cy_norm: float) -> bool:
        if self.zone is None:
            return False
        x1, y1, x2, y2 = self.zone
        return x1 <= cx_norm <= x2 and y1 <= cy_norm <= y2

    def process_frame(
        self, detections, frame_width: int, frame_height: int, timestamp: Optional[float] = None
    ) -> List[SecurityEvent]:
        """Bir karedeki tespitleri isler, varsa yeni guvenlik olaylarini dondurur."""
        timestamp = timestamp if timestamp is not None else time.time()
        events: List[SecurityEvent] = []
        seen_ids = set()

        person_centers = [
            ((d.box_xyxy[0] + d.box_xyxy[2]) / 2, (d.box_xyxy[1] + d.box_xyxy[3]) / 2)
            for d in detections
            if d.class_name == ZONE_RELEVANT_PERSON_CLASS
        ]

        for det in detections:
            if det.track_id is None:
                continue  # takip ID'si olmayan (sadece tek-kare) tespitler atlanir
            tid = det.track_id
            seen_ids.add(tid)
            x1, y1, x2, y2 = det.box_xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            state = self._tracks.setdefault(
                tid, _TrackState(class_name=det.class_name, first_seen=timestamp, last_seen=timestamp)
            )
            state.last_seen = timestamp
            state.positions.append((timestamp, cx, cy))
            duration = timestamp - state.first_seen

            # --- 1) Yasak bolge ihlali ---
            if self.zone and det.class_name == ZONE_RELEVANT_PERSON_CLASS:
                inside = self._point_in_zone(cx / frame_width, cy / frame_height)
                if inside and not state.zone_event_fired:
                    state.zone_event_fired = True
                    events.append(SecurityEvent(
                        timestamp, "ZONE_IHLALI", "KRITIK", det.class_name, tid,
                        f"Kisi #{tid} yasak bolgeye girdi",
                    ))
                elif not inside:
                    state.zone_event_fired = False  # tekrar girerse yeniden tetiklensin

            # --- 2) Uzun sure bekleme (loitering) ---
            if (det.class_name == ZONE_RELEVANT_PERSON_CLASS
                    and duration >= self.loiter_seconds
                    and not state.loiter_event_fired):
                state.loiter_event_fired = True
                events.append(SecurityEvent(
                    timestamp, "UZUN_SURE_BEKLEME", "UYARI", det.class_name, tid,
                    f"Kisi #{tid} {duration:.0f} saniyedir alanda",
                ))

            # --- 3) Terk edilmis nesne ---
            if (det.class_name in PORTABLE_OBJECT_CLASSES
                    and not state.abandoned_event_fired
                    and duration >= self.abandoned_seconds
                    and len(state.positions) >= 2):
                _, first_x, first_y = state.positions[0]
                moved = ((cx - first_x) ** 2 + (cy - first_y) ** 2) ** 0.5
                if moved <= self.abandoned_move_threshold_px:
                    nearest = min(
                        (((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 for px, py in person_centers),
                        default=float("inf"),
                    )
                    if nearest > self.abandoned_person_distance_px:
                        state.abandoned_event_fired = True
                        events.append(SecurityEvent(
                            timestamp, "TERK_EDILMIS_NESNE", "KRITIK", det.class_name, tid,
                            f"{det.class_name} (#{tid}) sahipsiz birakilmis olabilir",
                        ))

        # Artik ekranda gorunmeyen takipleri bellekten temizle
        stale = [tid for tid, st in self._tracks.items()
                 if timestamp - st.last_seen > 5.0 and tid not in seen_ids]
        for tid in stale:
            del self._tracks[tid]

        return events

    @staticmethod
    def draw_zone(frame, zone, color=(0, 0, 255), alpha=0.25):
        """Yasak bolgeyi yari saydam bir dikdortgen olarak karenin uzerine cizer."""
        if zone is None:
            return frame
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = zone
        pt1, pt2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        overlay = frame.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.rectangle(frame, pt1, pt2, color, 2)
        cv2.putText(frame, "YASAK BOLGE", (pt1[0], max(pt1[1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
