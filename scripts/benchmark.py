"""
scripts/benchmark.py
---------------------
Farkli YOLOv8 model boyutlarinin (nano / small / medium) hiz ve tespit
performansini ayni goruntu uzerinde karsilastirir. Sonucu hem konsola
yazdirir hem de examples/benchmark.png olarak grafikler.

Calistirmak icin (proje kok dizininden):
    python scripts/benchmark.py
"""

import os
import sys
import time

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.detector import ObjectDetector

SAMPLE_IMAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "ornek.jpg"
)
OUTPUT_CHART = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "benchmark.png"
)
N_RUNS = 5  # her model icin kac kere calistirip ortalama alinacak


def benchmark_model(model_key: str, image, n_runs: int = N_RUNS):
    detector = ObjectDetector(model_key)
    detector.detect(image, conf=0.4)  # ilk cagri: model yukleme suresini olcumden ayir

    durations = []
    detection_count = 0
    for _ in range(n_runs):
        start = time.perf_counter()
        result = detector.detect(image, conf=0.4)
        durations.append((time.perf_counter() - start) * 1000)  # ms
        detection_count = result.count

    avg_ms = sum(durations) / len(durations)
    return {
        "model": model_key,
        "avg_ms": avg_ms,
        "fps": 1000 / avg_ms if avg_ms > 0 else 0,
        "detections": detection_count,
    }


def main():
    image = cv2.imread(SAMPLE_IMAGE)
    if image is None:
        raise FileNotFoundError(f"Ornek goruntu bulunamadi: {SAMPLE_IMAGE}")

    print(f"[INFO] Benchmark basliyor, her model {N_RUNS} kere calistirilacak...\n")

    results = []
    for model_key in ObjectDetector.AVAILABLE_MODELS.keys():
        print(f"[INFO] Test ediliyor: {model_key} ...")
        res = benchmark_model(model_key, image)
        results.append(res)
        print(f"   -> Ortalama: {res['avg_ms']:.1f} ms  |  {res['fps']:.1f} FPS  |  "
              f"{res['detections']} tespit\n")

    # Konsol tablosu
    print(f"{'Model':<20}{'Ort. Sure (ms)':<18}{'FPS':<10}{'Tespit Sayisi':<15}")
    print("-" * 63)
    for r in results:
        print(f"{r['model']:<20}{r['avg_ms']:<18.1f}{r['fps']:<10.1f}{r['detections']:<15}")

    # Grafik
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    models = [r["model"] for r in results]
    fps_values = [r["fps"] for r in results]
    ms_values = [r["avg_ms"] for r in results]

    colors = ["#4C9AFF", "#7A5FFF", "#FF6B6B"]

    ax1.bar(models, fps_values, color=colors)
    ax1.set_title("Hiz (FPS - yuksek daha iyi)")
    ax1.set_ylabel("FPS")
    ax1.tick_params(axis="x", rotation=15)

    ax2.bar(models, ms_values, color=colors)
    ax2.set_title("Gecikme (ms - dusuk daha iyi)")
    ax2.set_ylabel("Milisaniye")
    ax2.tick_params(axis="x", rotation=15)

    fig.suptitle("YOLOv8 Model Boyutu Karsilastirmasi (CPU)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_CHART), exist_ok=True)
    fig.savefig(OUTPUT_CHART, dpi=150)
    print(f"\n[INFO] Grafik kaydedildi -> {OUTPUT_CHART}")


if __name__ == "__main__":
    main()
