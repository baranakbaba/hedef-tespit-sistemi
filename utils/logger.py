"""
utils/logger.py
----------------
Tespit oturumlarini ve guvenlik olaylarini kalici olarak data/ klasorune
CSV olarak kaydeden ve gecmis kayitlari okuyup analiz icin geri donduren
yardimci modul.
"""

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
HISTORY_PATH = os.path.join(_DATA_DIR, "history.csv")
EVENTS_PATH = os.path.join(_DATA_DIR, "events.csv")

_COLUMNS = ["timestamp", "source_type", "class_name", "confidence", "track_id"]
_EVENT_COLUMNS = ["timestamp", "video_saniye", "event_type", "severity", "class_name", "track_id", "description"]


@dataclass
class LogEntry:
    source_type: str  # "goruntu" | "video" | "webcam"
    class_name: str
    confidence: float
    track_id: int = -1


def _ensure_file(path, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(columns)


def log_entries(entries: List[LogEntry]):
    """Bir grup tespit sonucunu gecmis dosyasina ekler."""
    if not entries:
        return
    _ensure_file(HISTORY_PATH, _COLUMNS)
    now = datetime.now().isoformat(timespec="seconds")
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for e in entries:
            writer.writerow([now, e.source_type, e.class_name, f"{e.confidence:.4f}", e.track_id])


def load_history() -> pd.DataFrame:
    """Tum tespit gecmisini bir pandas DataFrame olarak dondurur (bossa bos DataFrame)."""
    _ensure_file(HISTORY_PATH, _COLUMNS)
    try:
        df = pd.read_csv(HISTORY_PATH)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=_COLUMNS)


def clear_history():
    """Tespit gecmisini sifirlar (kullanicinin acikca istemesi uzerine)."""
    _ensure_file(HISTORY_PATH, _COLUMNS)
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_COLUMNS)


def log_events(events: List) -> None:
    """SecurityEvent listesini kalici olarak events.csv'ye ekler.

    events: utils.security_events.SecurityEvent nesnelerinin listesi
    (timestamp, event_type, severity, class_name, track_id, description alanlarina sahip).
    """
    if not events:
        return
    _ensure_file(EVENTS_PATH, _EVENT_COLUMNS)
    now = datetime.now().isoformat(timespec="seconds")
    with open(EVENTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for e in events:
            writer.writerow([
                now, f"{e.timestamp:.1f}", e.event_type, e.severity,
                e.class_name, e.track_id if e.track_id is not None else -1, e.description,
            ])


def load_events() -> pd.DataFrame:
    """Tum guvenlik olayi gecmisini bir pandas DataFrame olarak dondurur."""
    _ensure_file(EVENTS_PATH, _EVENT_COLUMNS)
    try:
        df = pd.read_csv(EVENTS_PATH)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=_EVENT_COLUMNS)


def clear_events():
    """Guvenlik olayi gecmisini sifirlar."""
    _ensure_file(EVENTS_PATH, _EVENT_COLUMNS)
    with open(EVENTS_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_EVENT_COLUMNS)
