"""
tests/test_security_events.py
------------------------------
SecurityMonitor'un yasak bolge, terk edilmis nesne ve loitering mantiginin
dogru calistigini, sahte (mock) tespit nesneleriyle -gercek YOLO'ya ihtiyac
duymadan- dogrular.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.security_events import SecurityMonitor

FRAME_W, FRAME_H = 640, 480


@dataclass
class FakeDetection:
    """utils.detector.Detection ile ayni arayuzu taklit eden test yardimcisi."""
    class_name: str
    track_id: Optional[int]
    box_xyxy: tuple


def person_at(track_id, cx, cy, size=40):
    return FakeDetection("person", track_id, (cx - size, cy - size, cx + size, cy + size))


def bag_at(track_id, cx, cy, size=20):
    return FakeDetection("backpack", track_id, (cx - size, cy - size, cx + size, cy + size))


class TestZoneIhlali:
    def test_bolge_disindaki_kisi_olay_uretmiyor(self):
        monitor = SecurityMonitor(zone=(0.6, 0.6, 1.0, 1.0))
        events = monitor.process_frame([person_at(1, 50, 50)], FRAME_W, FRAME_H, timestamp=0)
        assert events == []

    def test_bolgeye_giren_kisi_olay_uretiyor(self):
        monitor = SecurityMonitor(zone=(0.6, 0.6, 1.0, 1.0))
        # Bolge (0.6-1.0, 0.6-1.0) normalize -> piksel karsiligi yaklasik (384-640, 288-480)
        events = monitor.process_frame([person_at(1, 500, 400)], FRAME_W, FRAME_H, timestamp=0)
        assert len(events) == 1
        assert events[0].event_type == "ZONE_IHLALI"
        assert events[0].severity == "KRITIK"

    def test_ayni_kisi_bolgede_kalirsa_tekrar_tetiklenmiyor(self):
        monitor = SecurityMonitor(zone=(0.6, 0.6, 1.0, 1.0))
        monitor.process_frame([person_at(1, 500, 400)], FRAME_W, FRAME_H, timestamp=0)
        events2 = monitor.process_frame([person_at(1, 505, 405)], FRAME_W, FRAME_H, timestamp=1)
        assert events2 == []

    def test_bolgeden_cikip_tekrar_girince_yeniden_tetikleniyor(self):
        monitor = SecurityMonitor(zone=(0.6, 0.6, 1.0, 1.0))
        monitor.process_frame([person_at(1, 500, 400)], FRAME_W, FRAME_H, timestamp=0)
        monitor.process_frame([person_at(1, 50, 50)], FRAME_W, FRAME_H, timestamp=1)  # cikti
        events3 = monitor.process_frame([person_at(1, 500, 400)], FRAME_W, FRAME_H, timestamp=2)
        assert len(events3) == 1
        assert events3[0].event_type == "ZONE_IHLALI"


class TestLoitering:
    def test_kisa_sureli_varlik_uyari_uretmiyor(self):
        monitor = SecurityMonitor(loiter_seconds=8.0)
        monitor.process_frame([person_at(1, 100, 100)], FRAME_W, FRAME_H, timestamp=0)
        events = monitor.process_frame([person_at(1, 100, 100)], FRAME_W, FRAME_H, timestamp=3)
        assert events == []

    def test_esik_asilinca_loitering_uyarisi_geliyor(self):
        monitor = SecurityMonitor(loiter_seconds=8.0)
        monitor.process_frame([person_at(1, 100, 100)], FRAME_W, FRAME_H, timestamp=0)
        events = monitor.process_frame([person_at(1, 100, 100)], FRAME_W, FRAME_H, timestamp=9)
        assert len(events) == 1
        assert events[0].event_type == "UZUN_SURE_BEKLEME"
        assert events[0].severity == "UYARI"

    def test_ayni_kisi_icin_ikinci_kez_tetiklenmiyor(self):
        monitor = SecurityMonitor(loiter_seconds=8.0)
        monitor.process_frame([person_at(1, 100, 100)], FRAME_W, FRAME_H, timestamp=0)
        monitor.process_frame([person_at(1, 100, 100)], FRAME_W, FRAME_H, timestamp=9)
        events = monitor.process_frame([person_at(1, 100, 100)], FRAME_W, FRAME_H, timestamp=15)
        assert events == []


class TestTerkEdilmisNesne:
    def test_yaninda_insan_varken_uyari_uretmiyor(self):
        monitor = SecurityMonitor(abandoned_seconds=5.0, abandoned_person_distance_px=150)
        monitor.process_frame([bag_at(2, 200, 200), person_at(1, 210, 210)], FRAME_W, FRAME_H, timestamp=0)
        events = monitor.process_frame([bag_at(2, 200, 200), person_at(1, 210, 210)], FRAME_W, FRAME_H, timestamp=6)
        assert events == []

    def test_sahipsiz_ve_hareketsiz_canta_uyari_uretiyor(self):
        monitor = SecurityMonitor(abandoned_seconds=5.0, abandoned_person_distance_px=150)
        monitor.process_frame([bag_at(2, 200, 200)], FRAME_W, FRAME_H, timestamp=0)
        events = monitor.process_frame([bag_at(2, 200, 200)], FRAME_W, FRAME_H, timestamp=6)
        assert len(events) == 1
        assert events[0].event_type == "TERK_EDILMIS_NESNE"
        assert events[0].severity == "KRITIK"

    def test_hareket_eden_canta_uyari_uretmiyor(self):
        """Canta tasiniyorsa (pozisyonu degisiyorsa) terk edilmis sayilmamali."""
        monitor = SecurityMonitor(abandoned_seconds=5.0, abandoned_move_threshold_px=25)
        monitor.process_frame([bag_at(2, 200, 200)], FRAME_W, FRAME_H, timestamp=0)
        events = monitor.process_frame([bag_at(2, 300, 300)], FRAME_W, FRAME_H, timestamp=6)
        assert events == []

    def test_yeterli_sure_gecmeden_uyari_uretmiyor(self):
        monitor = SecurityMonitor(abandoned_seconds=10.0)
        monitor.process_frame([bag_at(2, 200, 200)], FRAME_W, FRAME_H, timestamp=0)
        events = monitor.process_frame([bag_at(2, 200, 200)], FRAME_W, FRAME_H, timestamp=3)
        assert events == []


class TestGenelDavranis:
    def test_takip_idsi_olmayan_tespit_atlaniyor(self):
        monitor = SecurityMonitor(zone=(0.0, 0.0, 1.0, 1.0))
        events = monitor.process_frame([person_at(None, 100, 100)], FRAME_W, FRAME_H, timestamp=0)
        assert events == []

    def test_eskiyen_takip_bellekten_siliniyor(self):
        monitor = SecurityMonitor()
        monitor.process_frame([person_at(1, 100, 100)], FRAME_W, FRAME_H, timestamp=0)
        assert 1 in monitor._tracks
        monitor.process_frame([], FRAME_W, FRAME_H, timestamp=10)  # 5sn+ gorunmedi
        assert 1 not in monitor._tracks
