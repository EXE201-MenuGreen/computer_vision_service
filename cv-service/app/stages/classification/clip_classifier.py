"""
CLIP Zero-Shot Classifier — extends food recognition beyond Food-101.

Implementations
~~~~~~~~~~~~~~~
* ``CLIPZeroShotClassifier`` — openai/clip-vit-base-patch32 via HuggingFace.
  Encodes all food label text prompts at load time, then classifies each crop
  by cosine similarity between image embedding and pre-computed text matrix.
  Supports Food-101 (101 classes) + Vietnamese dishes (50 classes) = 151+ labels.
  Stores visual embeddings to Supabase fire-and-forget after each inference.

Register new label sets by extending CLIP_FOOD_LABELS dict — no model changes needed.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from app.core.base import ClassifiedDetection, PipelineStage, StageContext
from app.core.config import settings
from app.core.logging import get_logger
from app.registry import classifier_registry

logger = get_logger(__name__)

# ── Food label → English text prompt mapping ──────────────────────────────
# Format: "a photo of [food]" is standard CLIP prompt engineering.
# Food-101 labels (formatted as readable names)
_FOOD101_PROMPTS: Dict[str, str] = {
    "apple_pie":              "a photo of apple pie",
    "baby_back_ribs":         "a photo of baby back ribs",
    "baklava":                "a photo of baklava",
    "beef_carpaccio":         "a photo of beef carpaccio",
    "beef_tartare":           "a photo of beef tartare",
    "beet_salad":             "a photo of beet salad",
    "beignets":               "a photo of beignets",
    "bibimbap":               "a photo of bibimbap korean rice bowl",
    "bread_pudding":          "a photo of bread pudding",
    "breakfast_burrito":      "a photo of breakfast burrito",
    "bruschetta":             "a photo of bruschetta",
    "caesar_salad":           "a photo of caesar salad",
    "cannoli":                "a photo of cannoli",
    "caprese_salad":          "a photo of caprese salad",
    "carrot_cake":            "a photo of carrot cake",
    "ceviche":                "a photo of ceviche",
    "cheesecake":             "a photo of cheesecake",
    "cheese_plate":           "a photo of cheese plate",
    "chicken_curry":          "a photo of chicken curry",
    "chicken_quesadilla":     "a photo of chicken quesadilla",
    "chicken_wings":          "a photo of chicken wings",
    "chocolate_cake":         "a photo of chocolate cake",
    "chocolate_mousse":       "a photo of chocolate mousse",
    "churros":                "a photo of churros",
    "clam_chowder":           "a photo of clam chowder soup",
    "club_sandwich":          "a photo of club sandwich",
    "crab_cakes":             "a photo of crab cakes",
    "creme_brulee":           "a photo of creme brulee",
    "croque_madame":          "a photo of croque madame sandwich",
    "cup_cakes":              "a photo of cupcakes",
    "deviled_eggs":           "a photo of deviled eggs",
    "donuts":                 "a photo of donuts",
    "dumplings":              "a photo of dumplings",
    "edamame":                "a photo of edamame",
    "eggs_benedict":          "a photo of eggs benedict",
    "escargots":              "a photo of escargots snails",
    "falafel":                "a photo of falafel",
    "filet_mignon":           "a photo of filet mignon steak",
    "fish_and_chips":         "a photo of fish and chips",
    "foie_gras":              "a photo of foie gras",
    "french_fries":           "a photo of french fries",
    "french_onion_soup":      "a photo of french onion soup",
    "french_toast":           "a photo of french toast",
    "fried_calamari":         "a photo of fried calamari",
    "fried_rice":             "a photo of fried rice",
    "frozen_yogurt":          "a photo of frozen yogurt",
    "garlic_bread":           "a photo of garlic bread",
    "gnocchi":                "a photo of gnocchi pasta",
    "greek_salad":            "a photo of greek salad",
    "grilled_cheese_sandwich":"a photo of grilled cheese sandwich",
    "grilled_salmon":         "a photo of grilled salmon",
    "guacamole":              "a photo of guacamole",
    "gyoza":                  "a photo of gyoza dumplings",
    "hamburger":              "a photo of hamburger",
    "hot_and_sour_soup":      "a photo of hot and sour soup",
    "hot_dog":                "a photo of hot dog",
    "huevos_rancheros":       "a photo of huevos rancheros",
    "hummus":                 "a photo of hummus",
    "ice_cream":              "a photo of ice cream",
    "lasagna":                "a photo of lasagna",
    "lobster_bisque":         "a photo of lobster bisque soup",
    "lobster_roll_sandwich":  "a photo of lobster roll sandwich",
    "macaroni_and_cheese":    "a photo of macaroni and cheese",
    "macarons":               "a photo of macarons",
    "miso_soup":              "a photo of miso soup",
    "mussels":                "a photo of mussels",
    "nachos":                 "a photo of nachos",
    "omelette":               "a photo of omelette",
    "onion_rings":            "a photo of onion rings",
    "oysters":                "a photo of oysters",
    "pad_thai":               "a photo of pad thai noodles",
    "paella":                 "a photo of paella",
    "pancakes":               "a photo of pancakes",
    "panna_cotta":            "a photo of panna cotta dessert",
    "peking_duck":            "a photo of peking duck",
    "pho":                    "a photo of pho Vietnamese noodle soup",
    "pizza":                  "a photo of pizza",
    "pork_chop":              "a photo of pork chop",
    "poutine":                "a photo of poutine",
    "prime_rib":              "a photo of prime rib beef",
    "pulled_pork_sandwich":   "a photo of pulled pork sandwich",
    "ramen":                  "a photo of ramen noodle soup",
    "ravioli":                "a photo of ravioli pasta",
    "red_velvet_cake":        "a photo of red velvet cake",
    "risotto":                "a photo of risotto",
    "samosa":                 "a photo of samosa",
    "sashimi":                "a photo of sashimi raw fish",
    "scallops":               "a photo of scallops",
    "seaweed_salad":          "a photo of seaweed salad",
    "shrimp_and_grits":       "a photo of shrimp and grits",
    "spaghetti_bolognese":    "a photo of spaghetti bolognese",
    "spaghetti_carbonara":    "a photo of spaghetti carbonara",
    "spring_rolls":           "a photo of spring rolls",
    "steak":                  "a photo of steak",
    "strawberry_shortcake":   "a photo of strawberry shortcake",
    "sushi":                  "a photo of sushi",
    "tacos":                  "a photo of tacos",
    "takoyaki":               "a photo of takoyaki octopus balls",
    "tiramisu":               "a photo of tiramisu dessert",
    "tuna_tartare":           "a photo of tuna tartare",
    "waffles":                "a photo of waffles",
}

# Vietnamese dishes (complete dishes — not ingredients)
_VIET_DISH_PROMPTS: Dict[str, str] = {
    "bun_bo_hue":      "a photo of bun bo hue spicy Vietnamese beef noodle soup",
    "bun_rieu":        "a photo of bun rieu Vietnamese crab tomato noodle soup",
    "bun_mam":         "a photo of bun mam Vietnamese fermented fish noodle soup",
    "bun_cha":         "a photo of bun cha grilled pork with Vietnamese noodles",
    "banh_mi":         "a photo of banh mi Vietnamese sandwich baguette",
    "banh_xeo":        "a photo of banh xeo Vietnamese sizzling crepe",
    "banh_cuon":       "a photo of banh cuon Vietnamese steamed rice rolls",
    "com_tam":         "a photo of com tam Vietnamese broken rice with grilled pork",
    "goi_cuon":        "a photo of goi cuon Vietnamese fresh spring rolls",
    "bun_thit_nuong":  "a photo of bun thit nuong Vietnamese grilled pork noodle bowl",
    "canh_chua":       "a photo of canh chua Vietnamese sour soup",
    "ca_kho_to":       "a photo of ca kho to Vietnamese braised fish clay pot",
    "thit_kho":        "a photo of thit kho Vietnamese braised pork eggs caramel",
    "mi_quang":        "a photo of mi quang Vietnamese turmeric noodle",
    "cao_lau":         "a photo of cao lau Hoi An noodle dish",
    "banh_beo":        "a photo of banh beo Vietnamese steamed rice cake",
    "banh_bot_loc":    "a photo of banh bot loc Vietnamese shrimp tapioca dumpling",
    "chao_ga":         "a photo of chao ga Vietnamese chicken congee rice porridge",
    "com_chien":       "a photo of com chien Vietnamese fried rice",
    "lau":             "a photo of lau Vietnamese hot pot",
    "pho_bo":          "a photo of pho bo Vietnamese beef noodle soup",
    "pho_ga":          "a photo of pho ga Vietnamese chicken noodle soup",
    "hu_tieu_nam_vang":"a photo of hu tieu nam vang Phnom Penh noodle soup",
    "bun_ca":          "a photo of bun ca Vietnamese fish noodle soup",
    "nem_ran":         "a photo of nem ran Vietnamese fried spring rolls",
    "cha_gio":         "a photo of cha gio Vietnamese egg rolls",
    "xoi":             "a photo of xoi Vietnamese sticky rice",
    "che":             "a photo of che Vietnamese sweet dessert soup",
    "banh_flan":       "a photo of banh flan Vietnamese caramel flan",
    "goi_ga":          "a photo of goi ga Vietnamese chicken salad",
    "bo_luc_lac":      "a photo of bo luc lac Vietnamese shaking beef",
    "vit_quay":        "a photo of vit quay Vietnamese roast duck",
    "com_suon":        "a photo of com suon Vietnamese broken rice grilled pork chop",
    "banh_trang_nuong":"a photo of banh trang nuong Vietnamese grilled rice paper",
    "bap_xao":         "a photo of bap xao Vietnamese stir fried corn",
    "rau_muong_xao":   "a photo of rau muong xao Vietnamese stir fried water spinach",
    "canh_rau":        "a photo of canh rau Vietnamese vegetable soup",
    "tom_rang_muoi":   "a photo of tom rang muoi Vietnamese salt pepper shrimp",
    "muc_chien":       "a photo of muc chien Vietnamese fried squid",
    "oc_xao":          "a photo of oc xao Vietnamese stir fried snails",
    "banh_mi_op_la":   "a photo of banh mi op la Vietnamese egg sandwich fried egg",
    "chao_long":       "a photo of chao long Vietnamese pork offal congee",
    "sup_cua":         "a photo of sup cua Vietnamese crab asparagus soup",
    "bun_dau_mam_tom": "a photo of bun dau mam tom Vietnamese tofu shrimp paste",
    "banh_khot":       "a photo of banh khot Vietnamese mini savory pancakes",
    "banh_day":        "a photo of banh day Vietnamese sticky rice cake",
    "dau_hu_sot_ca":   "a photo of dau hu sot ca Vietnamese tofu tomato sauce",
    "ca_chien":        "a photo of ca chien Vietnamese pan fried fish",
    "suon_xao_chua_ngot": "a photo of suon xao chua ngot Vietnamese sweet sour pork ribs",
    "tom_kho":         "a photo of tom kho Vietnamese braised shrimp caramel",
    "cua_rang_me":     "a photo of cua rang me Vietnamese tamarind crab",
}

# Combined label → prompt mapping (used to build text embedding matrix)
CLIP_FOOD_LABELS: Dict[str, str] = {**_FOOD101_PROMPTS, **_VIET_DISH_PROMPTS}

# Minimum CLIP cosine similarity to accept a classification
_MIN_CLIP_CONFIDENCE = 0.18


# ── CLIP Zero-Shot Classifier ──────────────────────────────────────────────
@classifier_registry.register("clip_zero_shot")
class CLIPZeroShotClassifier(PipelineStage):
    """
    Zero-shot food classifier using CLIP (openai/clip-vit-base-patch32).

    At load(): encode all food label text prompts into a (N_labels × 512) matrix.
    At process(): for each crop, encode image → cosine sim with text matrix
                  → return top-1 label + similarity score.

    Advantages over EfficientNet-B4:
    - Not limited to 101 Food-101 classes
    - Recognises Vietnamese dishes out-of-the-box via text prompts
    - Extend label set without any model retraining
    - Stores visual embeddings to Supabase for future visual search (Option 3)
    """

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._text_embeds: Optional[torch.Tensor] = None   # (N, 512) normalised
        self._labels: List[str] = []

    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        model_name = settings.clip_model_name
        logger.info("loading_clip_model", model=model_name)

        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name)
        self._model.eval()
        self._model.to(settings.device)

        # Pre-compute text embeddings for all food labels
        self._labels, self._text_embeds = self._build_text_matrix()
        logger.info(
            "clip_loaded",
            model=model_name,
            n_labels=len(self._labels),
        )

    def _build_text_matrix(self) -> Tuple[List[str], torch.Tensor]:
        """Encode all text prompts into a normalised (N × 512) tensor."""
        labels = list(CLIP_FOOD_LABELS.keys())
        prompts = [CLIP_FOOD_LABELS[k] for k in labels]

        # Process in batches of 64 to avoid OOM on large label sets
        batch_size = 64
        all_embeds: List[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i: i + batch_size]
                inputs = self._processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(settings.device)
                text_features = self._model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_embeds.append(text_features)

        text_matrix = torch.cat(all_embeds, dim=0)   # (N, 512)
        logger.debug("text_matrix_built", shape=list(text_matrix.shape))
        return labels, text_matrix

    def process(self, ctx: StageContext) -> StageContext:
        for det in ctx.detections:
            label, confidence, image_embed = self._classify_crop(det.crop)
            ctx.classifications.append(
                ClassifiedDetection(
                    raw=det,
                    label=label,
                    classify_confidence=confidence,
                )
            )
            # Store visual embedding to Supabase (fire-and-forget, best-effort)
            if image_embed is not None and settings.supabase_url:
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(
                        _store_visual_embedding(
                            request_id=ctx.request_id,
                            food_label=label,
                            confidence=confidence,
                            embedding=image_embed,
                        )
                    )
                except RuntimeError:
                    pass  # No running event loop (e.g., Celery worker) — skip

        logger.info(
            "clip_classification_done",
            request_id=ctx.request_id,
            n_classified=len(ctx.classifications),
        )
        return ctx

    def _classify_crop(
        self, crop
    ) -> Tuple[str, float, Optional[List[float]]]:
        """
        Classify a single PIL crop via CLIP.

        Returns (label, confidence, image_embedding_as_list).
        Returns ("unknown", 0.0, None) when below _MIN_CLIP_CONFIDENCE.
        """
        with torch.no_grad():
            inputs = self._processor(
                images=crop,
                return_tensors="pt",
            ).to(settings.device)
            image_features = self._model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity: image (1×512) · text_matrix.T (512×N) → (1×N)
            similarities = (image_features @ self._text_embeds.T).squeeze(0)
            top_idx = int(similarities.argmax().item())
            top_score = float(similarities[top_idx].item())

        if top_score < _MIN_CLIP_CONFIDENCE:
            return "unknown", round(top_score, 3), None

        label = self._labels[top_idx]
        embed_list = image_features.squeeze(0).cpu().tolist()
        return label, round(top_score, 3), embed_list


# ── Mock (testing) ────────────────────────────────────────────────────────
@classifier_registry.register("clip_mock")
class MockCLIPClassifier(PipelineStage):
    """Returns 'pho' for every crop — for unit tests."""

    def load(self) -> None:
        pass

    def process(self, ctx: StageContext) -> StageContext:
        for det in ctx.detections:
            ctx.classifications.append(
                ClassifiedDetection(
                    raw=det,
                    label="pho",
                    classify_confidence=0.76,
                )
            )
        return ctx


# ── Supabase visual embedding store (fire-and-forget) ─────────────────────
async def _store_visual_embedding(
    request_id: str,
    food_label: str,
    confidence: float,
    embedding: List[float],
) -> None:
    """Store a CLIP visual embedding to Supabase (best-effort, never raises)."""
    try:
        from app.services.visual_store import store_visual_embedding
        await store_visual_embedding(
            request_id=request_id,
            food_label=food_label,
            confidence=confidence,
            embedding=embedding,
        )
    except Exception as exc:
        logger.warning("visual_store_failed", food_label=food_label, error=str(exc))
