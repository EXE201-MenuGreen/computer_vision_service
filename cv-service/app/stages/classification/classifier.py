"""
Classification stages — label each detected food crop.

Implementations
~~~~~~~~~~~~~~~
* ``EfficientNetClassifier`` — EfficientNet-B4 via ``timm``.
* ``MockClassifier``         — returns a fixed label for testing.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torchvision.transforms as T

from app.core.base import ClassifiedDetection, PipelineStage, StageContext
from app.core.config import settings
from app.core.logging import get_logger
from app.registry import classifier_registry

logger = get_logger(__name__)

# Food-101 class names (full list)
FOOD101_LABELS: List[str] = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles",
]

# Vietnamese culinary ingredient labels (mirrors keys in FALLBACK_NUTRITION)
VIET_INGREDIENT_LABELS: List[str] = [
    # Gia vị lỏng & Đồ lên men
    "nuoc_mam", "nuoc_mam_nhi", "mam_tom", "mam_ruoc", "mam_nem",
    "mam_tep", "mam_ba_khia", "mam_ca_sac", "mam_ca_linh",
    "tuong_hot", "tuong_ban", "tuong_den", "tuong_ot", "nuoc_tuong", "xi_dau",
    "giam_gao", "giam_nuoi", "com_me", "tai_chua", "qua_doc",
    "chao_trang", "chao_do", "dau_mau_dieu", "dau_me", "dau_hao", "mo_heo", "top_mo",
    # Củ gia vị
    "hanh_tim", "hanh_tay", "toi", "gung", "sa", "rieng", "nghe_tuoi", "cu_nen",
    # Rau thơm cơ bản
    "hanh_la", "ngo_ri", "ngo_gai", "rau_ngo",
    # Rau thơm ăn kèm
    "hung_que", "hung_lui", "tia_to", "kinh_gioi", "rau_ram", "dieu_ca", "thi_la", "rau_hung_cay",
    # Lá gia vị
    "la_lot", "la_chanh", "la_dua", "la_mac_mat", "la_cach", "la_giang",
    # Ớt & Tiêu
    "ot_hiem", "ot_sung", "tieu_den", "tieu_so", "tieu_xanh",
    # Rau lá
    "rau_muong", "rau_cai_xanh", "rau_cai_ngot", "rau_cai_cuc", "rau_cai_thia",
    "mong_toi", "rau_ngot", "xa_lach", "rau_dang", "cai_bap", "rau_den",
    # Quả nấu kèm
    "kho_qua", "bau", "bi_xanh", "bi_do", "muop_huong", "su_su",
    "ca_chua", "ca_phao", "ca_tim",
    # Củ
    "ca_rot", "cu_cai_trang", "su_hao", "khoai_tay", "khoai_lang",
    "khoai_mon", "khoai_so", "cu_san",
    # Đặc thù
    "hoa_chuoi", "ngo_sen", "gia_do", "mang_tuoi", "mang_kho", "mang_chua", "doc_mung",
    # Nấm
    "nam_rom", "moc_nhi", "nam_huong", "nam_kim_cham", "nam_bao_ngu",
    # Tinh bột & Sợi
    "gao_te", "gao_nep", "com", "cot_m",
    "bun", "banh_pho", "banh_canh", "hu_tieu",
    "mien_dong", "mi_trung", "banh_trang", "banh_da_cua", "banh_hoi",
    "bot_gao", "bot_nang", "bot_nep", "bot_chien_xu", "bot_banh_xeo",
    # Thịt heo
    "thit_ba_chi", "thit_nac_vai", "thit_nac_dam", "suon_non",
    "mong_gio", "da_heo", "tai_heo", "long_heo", "gio_song",
    # Thịt bò
    "thit_bo_than", "bap_bo", "nam_bo", "gan_bo", "duoi_bo",
    # Gia cầm
    "ga_ta", "vit", "ngan", "chim_bo_cau",
    # Trứng
    "trung_ga", "trung_vit", "trung_cut", "trung_vit_lon", "trung_bac_thao", "trung_muoi",
    # Cá
    "ca_loc", "ca_dieu_hong", "ca_ro", "ca_tre", "ca_thu", "ca_ngu", "ca_nuc", "ca_bong",
    # Hải sản khác
    "tom_su", "tom_dat", "tom_the", "cua_dong", "cua_bien", "ghe",
    "muc_la", "muc_ong", "muc_trung", "ngheu", "so_huyet", "oc_buou", "hen", "luon",
    # Thực vật đạm
    "dau_hu_trang", "dau_hu_chien", "tau_hu_ky",
    # Gia vị khô & Hạt
    "hoa_hoi", "que_chi", "thao_qua", "hat_mui", "dinh_huong", "hat_mac_khen", "hat_doi",
    "dau_xanh", "dau_phong", "me_vung", "hat_dieu",
    # Trái cây nấu món mặn
    "me_qua", "dua_thom", "xoai_xanh", "khe_chua", "chuoi_chat", "sung", "qua_sau",
    # Gia vị tinh luyện
    "muoi_hat", "muoi_ham", "duong_cat", "duong_phen", "duong_thot_not", "hat_nem", "bot_ngot",
]

# Combined label set: Food-101 Western + Vietnamese ingredients
ALL_LABELS: List[str] = FOOD101_LABELS + VIET_INGREDIENT_LABELS

# EfficientNet preprocessing
_CLASSIFY_TRANSFORM = T.Compose([
    T.Resize((380, 380)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── EfficientNet-B4 ─────────────────────────────────────────
@classifier_registry.register("efficientnet_b4")
class EfficientNetClassifier(PipelineStage):
    """Classify each crop with EfficientNet-B4 (timm)."""

    def __init__(self) -> None:
        self._model: Optional[torch.nn.Module] = None

    def load(self) -> None:
        import timm

        weights = settings.food_classify_weights
        num_classes = len(ALL_LABELS)

        if not Path(weights).exists():
            logger.warning(
                "classifier_weights_missing",
                path=weights,
                hint="Using timm pretrained as fallback (Food-101 only — Vietnamese labels need fine-tuning)",
            )
            self._model = timm.create_model(
                "efficientnet_b4", pretrained=True, num_classes=101
            )
        else:
            model = timm.create_model(
                "efficientnet_b4", pretrained=False, num_classes=num_classes
            )
            state = torch.load(weights, map_location=settings.device)
            model.load_state_dict(state)
            self._model = model

        assert self._model is not None
        self._model.eval()
        self._model.to(settings.device)
        logger.info("classifier_loaded", num_classes=num_classes)

    def process(self, ctx: StageContext) -> StageContext:
        for det in ctx.detections:
            label, conf = self._classify_crop(det.crop)
            ctx.classifications.append(
                ClassifiedDetection(
                    raw=det,
                    label=label,
                    classify_confidence=conf,
                )
            )
        return ctx

    def _classify_crop(self, crop) -> tuple[str, float]:
        assert self._model is not None, "call load() before process()"
        tensor = _CLASSIFY_TRANSFORM(crop).unsqueeze(0).to(settings.device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1)
            top_prob, top_idx = probs.max(dim=-1)
        idx = int(top_idx.item())
        label = ALL_LABELS[idx] if idx < len(ALL_LABELS) else "unknown"
        return label, float(top_prob.item())


# ── YOLO Passthrough ─────────────────────────────────────────
@classifier_registry.register("yolo_passthrough")
class YoloPassthroughClassifier(PipelineStage):
    """
    Dung truc tiep class label tu YOLO thay vi chay them model phan loai.
    Chi hoat dong khi detector la YoloV8Detector (co yolo_label trong RawDetection).
    """

    def load(self) -> None:
        pass

    def process(self, ctx: StageContext) -> StageContext:
        for det in ctx.detections:
            label = det.yolo_label or "unknown"
            conf  = det.yolo_label_conf or det.detector_confidence
            ctx.classifications.append(
                ClassifiedDetection(
                    raw=det,
                    label=label,
                    classify_confidence=conf,
                )
            )
        logger.info(
            "yolo_passthrough_done",
            request_id=ctx.request_id,
            labels=[c.label for c in ctx.classifications],
        )
        return ctx


# ── Mock (testing) ───────────────────────────────────────────
@classifier_registry.register("mock")
class MockClassifier(PipelineStage):
    """Returns fixed labels for unit tests — alternates Western and Vietnamese."""

    _MOCK_LABELS = ["pizza", "pho", "ca_loc", "rau_muong", "thit_ba_chi"]
    _idx = 0

    def load(self) -> None:
        pass

    def process(self, ctx: StageContext) -> StageContext:
        for i, det in enumerate(ctx.detections):
            label = self._MOCK_LABELS[i % len(self._MOCK_LABELS)]
            ctx.classifications.append(
                ClassifiedDetection(
                    raw=det,
                    label=label,
                    classify_confidence=0.92,
                )
            )
        return ctx
