"""
tests/test_detector.py
-----------------------
ObjectDetector sinifinin dogru calistigini dogrulayan testler.
Calistirmak icin: pytest tests/ -v
"""

import os
import sys

import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.detector import Detection, FrameResult, ObjectDetector

SAMPLE_IMAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "ornek.jpg"
)


@pytest.fixture(scope="module")
def detector():
    return ObjectDetector("Hizli (nano)")


@pytest.fixture(scope="module")
def sample_image():
    img = cv2.imread(SAMPLE_IMAGE)
    assert img is not None, f"Ornek goruntu bulunamadi: {SAMPLE_IMAGE}"
    return img


class TestObjectDetector:
    def test_model_yuklenebiliyor(self, detector, sample_image):
        """Model hata vermeden yuklenip en az bir tahmin uretmeli."""
        result = detector.detect(sample_image, conf=0.25)
        assert isinstance(result, FrameResult)

    def test_bilinen_goruntude_nesne_buluyor(self, detector, sample_image):
        """Ornek goruntu bilinen bir sahne (otobus + insanlar); en az 1 tespit beklenir."""
        result = detector.detect(sample_image, conf=0.25)
        assert result.count > 0, "Ornek goruntude hic nesne tespit edilemedi"

    def test_tespit_alanlari_dolu(self, detector, sample_image):
        """Her tespitin sinif adi, guven skoru ve kutu koordinati olmali."""
        result = detector.detect(sample_image, conf=0.25)
        for det in result.detections:
            assert isinstance(det, Detection)
            assert det.class_name
            assert 0.0 <= det.confidence <= 1.0
            assert len(det.box_xyxy) == 4

    def test_guven_esigi_filtreliyor(self, detector, sample_image):
        """Yuksek guven esigi, dusuk esikten daha az (ya da esit) tespit vermeli."""
        low_conf = detector.detect(sample_image, conf=0.1)
        high_conf = detector.detect(sample_image, conf=0.9)
        assert high_conf.count <= low_conf.count

    def test_sinif_filtresi_calisiyor(self, detector, sample_image):
        """Sadece 'person' sinifi istendiginde, sonucta baska sinif olmamali."""
        person_class_id = [
            cid for cid, name in detector.class_names.items() if name == "person"
        ][0]
        result = detector.detect(sample_image, conf=0.25, classes=[person_class_id])
        for det in result.detections:
            assert det.class_name == "person"

    def test_counts_by_class_dogru_sayiyor(self, detector, sample_image):
        """counts_by_class(), toplam tespit sayisiyla tutarli olmali."""
        result = detector.detect(sample_image, conf=0.25)
        total_from_counts = sum(result.counts_by_class().values())
        assert total_from_counts == result.count

    def test_model_degistirme_calisiyor(self, detector):
        """set_model cagrildiginda model_key guncellenmeli (lazy-load)."""
        original = detector.model_key
        detector.set_model("Dengeli (small)")
        assert detector.model_key == "Dengeli (small)"
        detector.set_model(original)  # sonraki testleri etkilememesi icin geri al
