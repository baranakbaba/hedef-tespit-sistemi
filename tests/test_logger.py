"""
tests/test_logger.py
---------------------
Loglama fonksiyonlarinin dogru calistigini dogrulayan testler.
Gercek data/history.csv dosyasina dokunmamak icin gecici bir dosya kullanir.
"""

import os
import sys

import pytest
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.logger as logger_module
from utils.logger import (LogEntry, clear_events, clear_history, load_events,
                           load_history, log_entries, log_events)


@dataclass
class FakeEvent:
    """utils.security_events.SecurityEvent ile ayni arayuzu taklit eden test yardimcisi."""
    timestamp: float
    event_type: str
    severity: str
    class_name: str
    track_id: int
    description: str


@pytest.fixture(autouse=True)
def gecici_log_dosyasi(tmp_path, monkeypatch):
    """Her testte gercek CSV dosyalari yerine gecici dosyalar kullanilmasini saglar."""
    temp_history = tmp_path / "test_history.csv"
    temp_events = tmp_path / "test_events.csv"
    monkeypatch.setattr(logger_module, "HISTORY_PATH", str(temp_history))
    monkeypatch.setattr(logger_module, "EVENTS_PATH", str(temp_events))
    yield


class TestLogger:
    def test_bos_gecmis_bos_dataframe_donduruyor(self):
        df = load_history()
        assert len(df) == 0

    def test_kayit_eklenebiliyor(self):
        entries = [LogEntry(source_type="goruntu", class_name="person", confidence=0.87)]
        log_entries(entries)
        df = load_history()
        assert len(df) == 1
        assert df.iloc[0]["class_name"] == "person"

    def test_birden_fazla_kayit_birikiyor(self):
        log_entries([LogEntry(source_type="goruntu", class_name="bus", confidence=0.9)])
        log_entries([LogEntry(source_type="video", class_name="person", confidence=0.8)])
        df = load_history()
        assert len(df) == 2

    def test_bos_liste_hicbir_sey_eklemiyor(self):
        log_entries([])
        df = load_history()
        assert len(df) == 0

    def test_gecmis_temizleniyor(self):
        log_entries([LogEntry(source_type="goruntu", class_name="bus", confidence=0.9)])
        assert len(load_history()) == 1
        clear_history()
        assert len(load_history()) == 0

    def test_track_id_kaydediliyor(self):
        log_entries([LogEntry(source_type="video", class_name="person", confidence=0.8, track_id=7)])
        df = load_history()
        assert df.iloc[0]["track_id"] == 7


class TestEventLogger:
    def test_bos_olay_gecmisi_bos_dataframe_donduruyor(self):
        df = load_events()
        assert len(df) == 0

    def test_olay_eklenebiliyor(self):
        events = [FakeEvent(5.0, "ZONE_IHLALI", "KRITIK", "person", 1, "Kisi #1 yasak bolgeye girdi")]
        log_events(events)
        df = load_events()
        assert len(df) == 1
        assert df.iloc[0]["event_type"] == "ZONE_IHLALI"
        assert df.iloc[0]["severity"] == "KRITIK"

    def test_bos_liste_hicbir_sey_eklemiyor(self):
        log_events([])
        df = load_events()
        assert len(df) == 0

    def test_birden_fazla_olay_birikiyor(self):
        log_events([FakeEvent(1.0, "ZONE_IHLALI", "KRITIK", "person", 1, "a")])
        log_events([FakeEvent(2.0, "UZUN_SURE_BEKLEME", "UYARI", "person", 2, "b")])
        df = load_events()
        assert len(df) == 2

    def test_olay_gecmisi_temizleniyor(self):
        log_events([FakeEvent(1.0, "TERK_EDILMIS_NESNE", "KRITIK", "backpack", 3, "c")])
        assert len(load_events()) == 1
        clear_events()
        assert len(load_events()) == 0

    def test_video_saniyesi_dogru_kaydediliyor(self):
        log_events([FakeEvent(12.3, "UZUN_SURE_BEKLEME", "UYARI", "person", 4, "d")])
        df = load_events()
        assert abs(df.iloc[0]["video_saniye"] - 12.3) < 0.01
