"""
app.py
------
Hedef Tespit ve Takip Sistemi - Streamlit Dashboard

Bu uygulama; goruntu, video ya da webcam uzerinde YOLOv8 tabanli nesne
tespiti/takibi yapar, davranis tabanli guvenlik olaylarini (yasak bolge
ihlali, terk edilmis nesne, loitering) tespit eder, sonuclari
gorsellestirir ve tum tespit/olay gecmisini kaydedip analiz eder.

Calistirmak icin:
    streamlit run app.py
"""

import os
import tempfile
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.detector import ObjectDetector
from utils.logger import (LogEntry, clear_events, clear_history, load_events,
                           load_history, log_entries, log_events)
from utils.security_events import SEVERITY_RENK, SecurityMonitor

# --------------------------------------------------------------------------
# Sayfa ayarlari
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Hedef Tespit ve Takip Sistemi",
    page_icon="🎯",
    layout="wide",
)

if "detector" not in st.session_state:
    st.session_state.detector = ObjectDetector()
if "session_detections" not in st.session_state:
    st.session_state.session_detections = []


# --------------------------------------------------------------------------
# Kenar cubugu - genel ayarlar
# --------------------------------------------------------------------------
st.sidebar.title("🎯 Hedef Tespit Sistemi")
st.sidebar.caption("YOLOv8 tabanli gercek zamanli tespit, takip ve guvenlik olayi analizi")

page = st.sidebar.radio(
    "Sayfa",
    ["🖼️ Tespit Calistir", "📊 Analiz & Gecmis", "🚨 Guvenlik Olaylari", "ℹ️ Proje Hakkinda"],
)

st.sidebar.divider()
st.sidebar.subheader("Model Ayarlari")

model_key = st.sidebar.selectbox(
    "Model boyutu",
    options=list(ObjectDetector.AVAILABLE_MODELS.keys()),
    index=0,
    help="Nano en hizli fakat en az hassas; Medium en yavas fakat en hassas modeldir.",
)
st.session_state.detector.set_model(model_key)

conf_threshold = st.sidebar.slider(
    "Guven esigi", min_value=0.1, max_value=0.9, value=0.4, step=0.05,
    help="Bu degerin altindaki tespitler gosterilmez.",
)

hassas_mod = st.sidebar.toggle(
    "⚡ Gelismis Tespit Modu",
    value=False,
    help="Yuksek cozunurluk (imgsz=1280) + test-time augmentation ile calisir. "
         "Kucuk/uzak nesnelerde daha guvenilir olabilir, ancak 2-4x daha yavastir.",
)
DETECT_IMGSZ = 1280 if hassas_mod else 640
DETECT_AUGMENT = hassas_mod

all_classes = st.session_state.detector.class_names
class_filter = st.sidebar.multiselect(
    "Sadece su siniflari goster (bos = hepsi)",
    options=sorted(all_classes.values()),
)
selected_class_ids = (
    [cid for cid, name in all_classes.items() if name in class_filter]
    if class_filter else None
)


# --------------------------------------------------------------------------
# Yardimci fonksiyonlar
# --------------------------------------------------------------------------
def render_stats_row(result, elapsed_ms=None):
    cols = st.columns(4 if elapsed_ms is not None else 3)
    cols[0].metric("Tespit edilen nesne", result.count)
    cols[1].metric("Farkli sinif sayisi", len(result.counts_by_class()))
    top_class = max(result.counts_by_class().items(), key=lambda x: x[1])[0] if result.count else "-"
    cols[2].metric("En sik gorulen", top_class)
    if elapsed_ms is not None:
        cols[3].metric("Islem suresi", f"{elapsed_ms:.0f} ms")


def render_class_bar(result):
    counts = result.counts_by_class()
    if not counts:
        return
    df = pd.DataFrame({"Sinif": list(counts.keys()), "Adet": list(counts.values())})
    fig = px.bar(df, x="Sinif", y="Adet", color="Sinif", text="Adet")
    fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def add_to_session(result, source_type):
    entries = [
        LogEntry(source_type=source_type, class_name=d.class_name,
                  confidence=d.confidence, track_id=d.track_id or -1)
        for d in result.detections
    ]
    st.session_state.session_detections.extend(entries)
    log_entries(entries)


def render_event_badge(severity: str) -> str:
    renk = SEVERITY_RENK.get(severity, "#999999")
    return f'<span style="background-color:{renk};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;">{severity}</span>'


# --------------------------------------------------------------------------
# SAYFA 1: Tespit Calistir
# --------------------------------------------------------------------------
if page == "🖼️ Tespit Calistir":
    st.title("Hedef Tespit ve Takip Sistemi")
    st.caption("Bir goruntu/video yukle ya da webcam'ini kullanarak gercek zamanli tespit yap.")
    if hassas_mod:
        st.info("⚡ Gelismis Tespit Modu aktif — islemler normalden yavas olacak.")

    tab_image, tab_video, tab_webcam = st.tabs(["📷 Goruntu", "🎬 Video + Guvenlik Izleme", "🔴 Canli Webcam"])

    # ---- Goruntu sekmesi ----
    with tab_image:
        uploaded_img = st.file_uploader(
            "Bir goruntu yukle (jpg/png)", type=["jpg", "jpeg", "png"], key="img_upload"
        )
        use_sample = st.checkbox("Ornek goruntuyu kullan (assets/ornek.jpg)", value=uploaded_img is None)

        image_bytes = None
        if uploaded_img is not None:
            image_bytes = uploaded_img.read()
        elif use_sample:
            with open("assets/ornek.jpg", "rb") as f:
                image_bytes = f.read()

        if image_bytes is not None:
            file_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)

            start = time.time()
            result = st.session_state.detector.detect(
                img, conf=conf_threshold, classes=selected_class_ids,
                imgsz=DETECT_IMGSZ, augment=DETECT_AUGMENT,
            )
            elapsed_ms = (time.time() - start) * 1000

            col_img, col_stats = st.columns([2, 1])
            with col_img:
                st.image(
                    cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB),
                    caption="Tespit sonucu", use_container_width=True,
                )
            with col_stats:
                render_stats_row(result, elapsed_ms)
                render_class_bar(result)

            if st.button("💾 Bu sonucu gecmise kaydet", key="save_img"):
                add_to_session(result, "goruntu")
                st.success(f"{result.count} tespit gecmise kaydedildi.")

    # ---- Video sekmesi ----
    with tab_video:
        uploaded_video = st.file_uploader("Bir video yukle (mp4)", type=["mp4", "mov", "avi"])
        max_frames = st.slider("Islenecek maksimum kare sayisi", 10, 300, 60,
                                help="Uzun videolarda tum kareleri islemek yavas olabilir; demo icin sinirlandirildi.")

        st.markdown("##### 🚨 Guvenlik Izleme (opsiyonel)")
        guvenlik_aktif = st.checkbox(
            "Yasak bolge + terk edilmis nesne + uzun sure bekleme tespitini etkinlestir",
        )

        zone = None
        loiter_seconds, abandoned_seconds = 8.0, 8.0
        if guvenlik_aktif:
            st.caption("Yasak bolgeyi kare yuzdesi olarak tanimla (0 = sol/ust, 100 = sag/alt):")
            zc1, zc2, zc3, zc4 = st.columns(4)
            zx1 = zc1.number_input("X baslangic %", 0, 100, 60)
            zy1 = zc2.number_input("Y baslangic %", 0, 100, 60)
            zx2 = zc3.number_input("X bitis %", 0, 100, 100)
            zy2 = zc4.number_input("Y bitis %", 0, 100, 100)
            if zx1 < zx2 and zy1 < zy2:
                zone = (zx1 / 100, zy1 / 100, zx2 / 100, zy2 / 100)
            else:
                st.warning("Gecersiz bolge: X baslangic < X bitis ve Y baslangic < Y bitis olmali.")

            tc1, tc2 = st.columns(2)
            loiter_seconds = tc1.slider("Loitering esigi (saniye)", 2, 30, 5)
            abandoned_seconds = tc2.slider("Terk edilmis nesne esigi (saniye)", 2, 30, 5)

        if uploaded_video is not None and st.button("▶️ Videoyu isle"):
            # tempfile: isletim sistemine gore dogru gecici klasoru otomatik bulur
            # (Windows'ta 'C:\Users\...\AppData\Local\Temp', Linux/Mac'te '/tmp').
            # Sabit "/tmp/..." yazmak Windows'ta calismaz, bu yuzden boyle yapiyoruz.
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_video.read())
                tmp_path = tmp_file.name

            cap = cv2.VideoCapture(tmp_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            monitor = SecurityMonitor(
                zone=zone, loiter_seconds=loiter_seconds, abandoned_seconds=abandoned_seconds,
            ) if guvenlik_aktif else None
            all_events = []

            progress = st.progress(0, text="Video isleniyor...")
            frame_placeholder = st.empty()
            metric_cols = st.columns(3) if guvenlik_aktif else None
            class_counter = Counter()

            frame_idx = 0
            while frame_idx < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                result = st.session_state.detector.track(
                    frame, conf=conf_threshold, classes=selected_class_ids,
                    persist=True, imgsz=DETECT_IMGSZ,
                )
                for d in result.detections:
                    class_counter[d.class_name] += 1

                annotated = result.annotated_image
                if guvenlik_aktif:
                    video_saniye = frame_idx / video_fps
                    new_events = monitor.process_frame(
                        result.detections, frame_w, frame_h, timestamp=video_saniye,
                    )
                    all_events.extend(new_events)
                    annotated = SecurityMonitor.draw_zone(annotated, zone)
                    kritik = sum(1 for e in all_events if e.severity == "KRITIK")
                    uyari = sum(1 for e in all_events if e.severity == "UYARI")
                    metric_cols[0].metric("İşlenen kare", frame_idx + 1)
                    metric_cols[1].metric("🔴 Kritik olay", kritik)
                    metric_cols[2].metric("🟠 Uyari", uyari)

                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Kare {frame_idx + 1}", use_container_width=True,
                )
                progress.progress((frame_idx + 1) / max_frames)
                frame_idx += 1

            cap.release()
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            progress.empty()
            st.success(f"{frame_idx} kare islendi.")

            if class_counter:
                df = pd.DataFrame({"Sinif": list(class_counter.keys()), "Toplam Tespit": list(class_counter.values())})
                st.plotly_chart(px.bar(df, x="Sinif", y="Toplam Tespit", color="Sinif"), use_container_width=True)

            if guvenlik_aktif:
                if all_events:
                    st.markdown("##### 🚨 Tespit Edilen Guvenlik Olaylari")
                    events_df = pd.DataFrame([{
                        "Saniye": f"{e.timestamp:.1f}",
                        "Tur": e.event_type,
                        "Onem": e.severity,
                        "Aciklama": e.description,
                    } for e in all_events])
                    st.dataframe(events_df, use_container_width=True)
                    log_events(all_events)
                    st.caption("Bu olaylar otomatik olarak 'Guvenlik Olaylari' sayfasina kaydedildi.")
                else:
                    st.info("Bu video isleme oturumunda guvenlik olayi tespit edilmedi.")

    # ---- Webcam sekmesi ----
    with tab_webcam:
        st.info(
            "Streamlit tarayici uzerinden calistigi icin webcam yakalama kare-kare "
            "'Fotograf cek' widget'i ile yapilir (surekli akis icin src/detect.py "
            "CLI scriptini kullan: `python src/detect.py --source 0`)."
        )
        cam_image = st.camera_input("Bir kare yakala")
        if cam_image is not None:
            file_arr = np.frombuffer(cam_image.getvalue(), np.uint8)
            img = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)
            result = st.session_state.detector.detect(
                img, conf=conf_threshold, classes=selected_class_ids,
                imgsz=DETECT_IMGSZ, augment=DETECT_AUGMENT,
            )
            st.image(
                cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB),
                caption="Tespit sonucu", use_container_width=True,
            )
            render_stats_row(result)
            if st.button("💾 Bu sonucu gecmise kaydet", key="save_cam"):
                add_to_session(result, "webcam")
                st.success(f"{result.count} tespit gecmise kaydedildi.")


# --------------------------------------------------------------------------
# SAYFA 2: Analiz & Gecmis
# --------------------------------------------------------------------------
elif page == "📊 Analiz & Gecmis":
    st.title("Analiz & Gecmis")
    st.caption("Kaydedilen tum tespit oturumlarinin ozeti.")

    df = load_history()

    if df.empty:
        st.warning("Henuz kaydedilmis bir tespit yok. 'Tespit Calistir' sayfasindan sonuc kaydet.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam kayit", len(df))
        col2.metric("Farkli sinif", df["class_name"].nunique())
        col3.metric("Ortalama guven", f"%{df['confidence'].mean() * 100:.1f}")

        st.subheader("Sinif dagilimi")
        class_counts = df["class_name"].value_counts().reset_index()
        class_counts.columns = ["Sinif", "Adet"]
        st.plotly_chart(px.pie(class_counts, names="Sinif", values="Adet", hole=0.4),
                         use_container_width=True)

        st.subheader("Zaman icinde tespitler")
        df_time = df.copy()
        df_time["dakika"] = df_time["timestamp"].dt.floor("min")
        timeline = df_time.groupby("dakika").size().reset_index(name="Adet")
        st.plotly_chart(px.line(timeline, x="dakika", y="Adet", markers=True),
                         use_container_width=True)

        st.subheader("Kaynak turune gore dagilim")
        source_counts = df["source_type"].value_counts().reset_index()
        source_counts.columns = ["Kaynak", "Adet"]
        st.plotly_chart(px.bar(source_counts, x="Kaynak", y="Adet", color="Kaynak"),
                         use_container_width=True)

        st.subheader("Ham veri")
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ CSV olarak indir", data=df.to_csv(index=False).encode("utf-8"),
                file_name="tespit_gecmisi.csv", mime="text/csv",
            )
        with col_b:
            if st.button("🗑️ Gecmisi temizle"):
                clear_history()
                st.rerun()


# --------------------------------------------------------------------------
# SAYFA 3: Guvenlik Olaylari
# --------------------------------------------------------------------------
elif page == "🚨 Guvenlik Olaylari":
    st.title("Guvenlik Olaylari")
    st.caption(
        "Video isleme sirasinda 'Guvenlik Izleme' etkinlestirildiginde tespit edilen "
        "davranis-tabanli olaylarin (yasak bolge ihlali, terk edilmis nesne, uzun sure "
        "bekleme) gecmisi."
    )

    edf = load_events()

    if edf.empty:
        st.warning(
            "Henuz kaydedilmis bir guvenlik olayi yok. 'Tespit Calistir' → 'Video' "
            "sekmesinde 'Guvenlik Izleme'yi etkinlestirip bir video isle."
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam olay", len(edf))
        col2.metric("🔴 Kritik", int((edf["severity"] == "KRITIK").sum()))
        col3.metric("🟠 Uyari", int((edf["severity"] == "UYARI").sum()))

        col_pie, col_bar = st.columns(2)
        with col_pie:
            st.subheader("Olay turune gore dagilim")
            type_counts = edf["event_type"].value_counts().reset_index()
            type_counts.columns = ["Tur", "Adet"]
            st.plotly_chart(px.pie(type_counts, names="Tur", values="Adet", hole=0.4),
                             use_container_width=True)
        with col_bar:
            st.subheader("Onem seviyesine gore dagilim")
            sev_counts = edf["severity"].value_counts().reset_index()
            sev_counts.columns = ["Onem", "Adet"]
            renk_map = {"KRITIK": "#FF3B30", "UYARI": "#FFA500", "BILGI": "#4C9AFF"}
            st.plotly_chart(
                px.bar(sev_counts, x="Onem", y="Adet", color="Onem", color_discrete_map=renk_map),
                use_container_width=True,
            )

        st.subheader("Olay gecmisi")
        st.dataframe(
            edf.sort_values("timestamp", ascending=False)[
                ["timestamp", "video_saniye", "event_type", "severity", "class_name", "description"]
            ],
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ CSV olarak indir", data=edf.to_csv(index=False).encode("utf-8"),
                file_name="guvenlik_olaylari.csv", mime="text/csv",
            )
        with col_b:
            if st.button("🗑️ Olay gecmisini temizle"):
                clear_events()
                st.rerun()


# --------------------------------------------------------------------------
# SAYFA 4: Proje Hakkinda
# --------------------------------------------------------------------------
else:
    st.title("Proje Hakkinda")
    st.markdown("""
Bu proje, **YOLOv8** tabanli bir bilgisayarli goru sistemidir. Goruntu, video ve
webcam uzerinde nesne tespiti ve takibi (object tracking) yapar; bunun uzerine
**davranis tabanli guvenlik olayi analizi** katmani ekler ve sonuclari kalici
olarak kaydedip interaktif grafiklerle analiz eder.

### Guvenlik olayi analizi nasil calisir?
Tek bir karedeki nesne tespiti "bir insan var" diyebilir ama "tehlikeli mi"
sorusuna cevap veremez. Gercek guvenlik/perimeter sistemleri bunun yerine
**zaman icindeki davranisa** bakar. Bu proje 3 davranis kalibini modelliyor:

- **Yasak bolge ihlali** — Tanimli bir alana bir kisi girdiginde
- **Terk edilmis nesne** — Bir canta/valiz, sahibi yaninda olmadan uzun sure hareketsiz kaldiginda
- **Uzun sure bekleme (loitering)** — Bir kisi alanda beklenenden uzun sure kaldiginda

Bu mantik `utils/security_events.py` icinde her takip edilen nesne (track ID) icin
bir durum makinesi (state machine) olarak calisir ve 13 birim testiyle dogrulanmistir.

### Kullanilan teknolojiler
- **Ultralytics YOLOv8** — nesne tespiti ve takip (ByteTrack)
- **OpenCV** — goruntu/video isleme, bolge cizimi
- **Streamlit** — web arayuzu
- **Plotly** — interaktif grafikler
- **Pandas** — veri analizi ve loglama
- **pytest + GitHub Actions** — otomatik test ve surekli entegrasyon (CI)

### Mimari
```
app.py                    -> Streamlit web dashboard (4 sayfa)
utils/detector.py          -> YOLOv8 sarmalayicisi (tespit + takip + hassas mod)
utils/security_events.py    -> Davranis tabanli guvenlik olayi durum makinesi
utils/logger.py             -> Tespit + olay gecmisini CSV'ye yazma/okuma
src/detect.py                -> Komut satirindan calisan CLI alternatifi
scripts/benchmark.py         -> Model boyutlarinin hiz/hassasiyet karsilastirmasi
tests/                        -> 32 pytest birim testi
.github/workflows/ci.yml     -> Her push'ta testleri otomatik calistiran CI
Dockerfile                    -> Konteynerde calistirma destegi
```
""")
