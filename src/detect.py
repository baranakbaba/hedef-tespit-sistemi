"""
Hedef Tespit Sistemi
---------------------
YOLOv8 tabanli, goruntu / video / webcam uzerinde gercek zamanli nesne
(hedef) tespiti yapan basit ve genisletilebilir bir sistem.

Kullanim:
    python src/detect.py --source assets/ornek.jpg --output examples/sonuc.jpg
    python src/detect.py --source 0                       # webcam
    python src/detect.py --source video.mp4 --output out.mp4

Yazar: (Adini buraya ekle)
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 ile hedef tespit sistemi")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Goruntu/video dosya yolu ya da webcam icin '0'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Kullanilacak YOLO model agirligi (varsayilan: yolov8n.pt - hafif ve hizli)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Tespit icin minimum guven skoru (0-1 arasi)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Sonucun kaydedilecegi dosya yolu (verilmezse ekranda gosterilir/kaydedilmez)",
    )
    return parser.parse_args()


def run_on_image(model, source_path, conf, output_path):
    results = model.predict(source=source_path, conf=conf, verbose=False)
    result = results[0]
    annotated = result.plot()  # tespitleri kutu+etiket olarak ciziyor

    n_objects = len(result.boxes)
    print(f"[INFO] {n_objects} nesne tespit edildi.")
    for box in result.boxes:
        cls_name = model.names[int(box.cls[0])]
        conf_score = float(box.conf[0])
        print(f"   - {cls_name}: %{conf_score * 100:.1f} guven")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Sonuc kaydedildi -> {output_path}")

    return annotated, n_objects


def run_on_video_or_webcam(model, source, conf, output_path):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Kaynak acilamadi: {source}")

    writer = None
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        annotated = results[0].plot()

        if writer:
            writer.write(annotated)
        else:
            cv2.imshow("Hedef Tespit Sistemi", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1

    elapsed = time.time() - start
    fps_avg = frame_count / elapsed if elapsed > 0 else 0
    print(f"[INFO] {frame_count} kare islendi, ortalama {fps_avg:.1f} FPS")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    print(f"[INFO] Model yukleniyor: {args.model}")
    model = YOLO(args.model)

    source = args.source
    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_image:
        run_on_image(model, source, args.conf, args.output)
    else:
        run_on_video_or_webcam(model, source, args.conf, args.output)


if __name__ == "__main__":
    main()
