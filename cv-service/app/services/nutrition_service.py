"""
NutritionService: 5-tier lookup chain with accuracy-first design.

  Tier 0  verified   confidence=1.00  Admin-curated Supabase table (ground truth)
  Tier 1  redis      confidence=—      Shared Redis cache (TTL=24h, serves prior tier result)
  Tier 2  pgvector   confidence=0.90  Semantic vector search, threshold ≥ 0.82
  Tier 3  usda       confidence=0.75  USDA FoodData Central API + name validation
  Tier 4  fallback   confidence=0.30  Hardcoded approximate dict

Rules:
  - Tier 0 always overrides everything and is NOT cached (admin can update any time).
  - Redis is checked after tier 0; a hit returns the source/confidence of original tier.
  - USDA results are accepted only when the returned food name is semantically close
    enough to the query (controlled by USDA_NAME_MATCH_THRESHOLD). Rejected USDA
    results fall through to tier 4 — they are NOT stored in pgvector.
  - Tier 3 & 4 hits are audit-logged so admins can review and promote to tier 0.
  - Every non-verified result is stored in Redis with a TTL so stale data expires
    and tier 2+ is re-checked on the next lookup.
"""
from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.db.food_nutrition import get_verified_food, match_food, upsert_food
from app.schemas.cv_schemas import DetectedFood, FoodNutrition, MacroNutrients
from app.services import redis_cache

logger = get_logger(__name__)

# USDA FoodData Central nutrient IDs
_NUT_ENERGY  = 1008
_NUT_PROTEIN = 1003
_NUT_CARBS   = 1005
_NUT_FAT     = 1004
_NUT_FIBER   = 1079

# Maps Vietnamese ingredient labels → English USDA search terms
VIET_TO_USDA_QUERY: Dict[str, str] = {
    # Gia vị lỏng & Đồ lên men
    "nuoc_mam":        "fish sauce",
    "nuoc_mam_nhi":    "fish sauce premium",
    "mam_tom":         "shrimp paste",
    "mam_ruoc":        "fermented shrimp paste",
    "mam_nem":         "fermented anchovy sauce",
    "mam_tep":         "fermented shrimp",
    "mam_ba_khia":     "fermented crab",
    "mam_ca_sac":      "fermented snakeskin fish",
    "mam_ca_linh":     "fermented freshwater fish",
    "tuong_hot":       "fermented soybean sauce",
    "tuong_ban":       "fermented yellow soybean paste",
    "tuong_den":       "black bean sauce",
    "tuong_ot":        "chili sauce",
    "nuoc_tuong":      "soy sauce",
    "xi_dau":          "soy sauce",
    "giam_gao":        "rice vinegar",
    "giam_nuoi":       "vinegar",
    "com_me":          "fermented rice",
    "tai_chua":        "garcinia cowa sour fruit",
    "qua_doc":         "wild mangosteen sour",
    "chao_trang":      "white fermented tofu",
    "chao_do":         "red fermented tofu",
    "dau_mau_dieu":    "annatto oil",
    "dau_me":          "sesame oil",
    "dau_hao":         "oyster sauce",
    "mo_heo":          "lard",
    "top_mo":          "pork crackling",
    # Củ gia vị
    "hanh_tim":        "shallot",
    "hanh_tay":        "onion",
    "toi":             "garlic",
    "gung":            "ginger root",
    "sa":              "lemongrass",
    "rieng":           "galangal",
    "nghe_tuoi":       "fresh turmeric root",
    "cu_nen":          "chinese shallot",
    # Rau thơm cơ bản
    "hanh_la":         "green onion scallion",
    "ngo_ri":          "cilantro coriander leaves",
    "ngo_gai":         "culantro sawtooth herb",
    "rau_ngo":         "Vietnamese herb ngo om",
    # Rau thơm ăn kèm
    "hung_que":        "Thai basil",
    "hung_lui":        "spearmint Vietnamese",
    "tia_to":          "perilla leaf shiso",
    "kinh_gioi":       "Vietnamese balm herb",
    "rau_ram":         "Vietnamese coriander",
    "dieu_ca":         "fish mint houttuynia",
    "thi_la":          "dill herb",
    "rau_hung_cay":    "peppermint herb",
    # Lá gia vị
    "la_lot":          "betel leaf",
    "la_chanh":        "kaffir lime leaf",
    "la_dua":          "pandan leaf",
    "la_mac_mat":      "litsea cubeba leaf",
    "la_cach":         "Vietnamese herb la cach",
    "la_giang":        "sour leaf soup herb Vietnam",
    # Ớt & Tiêu
    "ot_hiem":         "bird eye chili pepper",
    "ot_sung":         "horn chili pepper",
    "tieu_den":        "black pepper",
    "tieu_so":         "white pepper",
    "tieu_xanh":       "fresh green pepper cluster",
    # Rau lá
    "rau_muong":       "water spinach kangkong",
    "rau_cai_xanh":    "Chinese mustard greens",
    "rau_cai_ngot":    "baby Chinese cabbage",
    "rau_cai_cuc":     "chrysanthemum greens",
    "rau_cai_thia":    "bok choy",
    "mong_toi":        "Malabar spinach",
    "rau_ngot":        "katuk sweet leaf",
    "xa_lach":         "lettuce",
    "rau_dang":        "bitter herb Vietnam",
    "cai_bap":         "cabbage",
    "rau_den":         "amaranth leaves",
    # Quả nấu kèm
    "kho_qua":         "bitter melon",
    "bau":             "bottle gourd",
    "bi_xanh":         "winter melon wax gourd",
    "bi_do":           "pumpkin",
    "muop_huong":      "luffa ridged gourd",
    "su_su":           "chayote",
    "ca_chua":         "tomato",
    "ca_phao":         "pea eggplant Thai",
    "ca_tim":          "eggplant aubergine",
    # Củ
    "ca_rot":          "carrot",
    "cu_cai_trang":    "daikon white radish",
    "su_hao":          "kohlrabi",
    "khoai_tay":       "potato",
    "khoai_lang":      "sweet potato",
    "khoai_mon":       "taro root",
    "khoai_so":        "cocoyam coco yam",
    "cu_san":          "jicama yambean",
    # Đặc thù
    "hoa_chuoi":       "banana blossom flower",
    "ngo_sen":         "lotus root",
    "gia_do":          "mung bean sprouts",
    "mang_tuoi":       "fresh bamboo shoot",
    "mang_kho":        "dried bamboo shoot",
    "mang_chua":       "pickled bamboo shoot sour",
    "doc_mung":        "taro stem Vietnamese",
    # Nấm
    "nam_rom":         "straw mushroom",
    "moc_nhi":         "wood ear mushroom black fungus",
    "nam_huong":       "shiitake mushroom",
    "nam_kim_cham":    "enoki mushroom",
    "nam_bao_ngu":     "oyster mushroom",
    # Tinh bột & Sợi
    "gao_te":          "white rice",
    "gao_nep":         "glutinous sticky rice",
    "com":             "cooked white rice",
    "cot_m":           "young sticky rice com",
    "bun":             "rice vermicelli noodle",
    "banh_pho":        "pho rice noodle",
    "banh_canh":       "thick rice noodle",
    "hu_tieu":         "rice noodle hu tieu",
    "mien_dong":       "glass noodle mung bean",
    "mi_trung":        "egg noodle",
    "banh_trang":      "rice paper wrapper",
    "banh_da_cua":     "red rice noodle crab",
    "banh_hoi":        "thin rice vermicelli cake",
    "bot_gao":         "rice flour",
    "bot_nang":        "tapioca starch",
    "bot_nep":         "glutinous rice flour",
    "bot_chien_xu":    "breadcrumbs tempura",
    "bot_banh_xeo":    "rice flour crispy pancake mix",
    # Thịt heo
    "thit_ba_chi":     "pork belly",
    "thit_nac_vai":    "pork shoulder",
    "thit_nac_dam":    "pork loin",
    "suon_non":        "pork spare ribs",
    "mong_gio":        "pig trotter",
    "da_heo":          "pork skin rind",
    "tai_heo":         "pig ear",
    "long_heo":        "pork intestine",
    "gio_song":        "fresh pork sausage paste",
    # Thịt bò
    "thit_bo_than":    "beef sirloin",
    "bap_bo":          "beef shank",
    "nam_bo":          "beef brisket",
    "gan_bo":          "beef tendon",
    "duoi_bo":         "oxtail",
    # Gia cầm
    "ga_ta":           "free range chicken",
    "vit":             "duck meat",
    "ngan":            "muscovy duck",
    "chim_bo_cau":     "pigeon squab",
    # Trứng
    "trung_ga":        "chicken egg",
    "trung_vit":       "duck egg",
    "trung_cut":       "quail egg",
    "trung_vit_lon":   "balut fertilized egg",
    "trung_bac_thao":  "century egg preserved",
    "trung_muoi":      "salted duck egg",
    # Cá
    "ca_loc":          "snakehead fish",
    "ca_dieu_hong":    "tilapia red",
    "ca_ro":           "climbing perch fish",
    "ca_tre":          "catfish",
    "ca_thu":          "mackerel fish",
    "ca_ngu":          "tuna fish",
    "ca_nuc":          "round scad fish",
    "ca_bong":         "goby fish",
    # Hải sản khác
    "tom_su":          "black tiger shrimp",
    "tom_dat":         "white shrimp",
    "tom_the":         "pacific white shrimp",
    "cua_dong":        "freshwater crab",
    "cua_bien":        "blue swimming crab",
    "ghe":             "mud crab",
    "muc_la":          "squid flat",
    "muc_ong":         "squid tube",
    "muc_trung":       "squid with eggs",
    "ngheu":           "clam",
    "so_huyet":        "blood cockle",
    "oc_buou":         "golden apple snail",
    "hen":             "freshwater mussel",
    "luon":            "eel freshwater",
    # Thực vật đạm
    "dau_hu_trang":    "tofu soft white",
    "dau_hu_chien":    "fried tofu",
    "tau_hu_ky":       "tofu skin yuba",
    # Gia vị khô & Hạt
    "hoa_hoi":         "star anise spice",
    "que_chi":         "cinnamon stick",
    "thao_qua":        "black cardamom",
    "hat_mui":         "coriander seed",
    "dinh_huong":      "cloves spice",
    "hat_mac_khen":    "Sichuan pepper Vietnamese",
    "hat_doi":         "Vietnamese wild pepper doi",
    "dau_xanh":        "mung bean",
    "dau_phong":       "peanut groundnut",
    "me_vung":         "sesame seed",
    "hat_dieu":        "cashew nut",
    # Trái cây nấu món mặn
    "me_qua":          "tamarind",
    "dua_thom":        "pineapple",
    "xoai_xanh":       "green unripe mango",
    "khe_chua":        "star fruit carambola",
    "chuoi_chat":      "unripe green banana",
    "sung":            "fig",
    "qua_sau":         "hog plum spondias",
    # Gia vị tinh luyện
    "muoi_hat":        "salt",
    "muoi_ham":        "salt",
    "duong_cat":       "white sugar",
    "duong_phen":      "rock sugar",
    "duong_thot_not":  "palm sugar",
    "hat_nem":         "seasoning powder umami",
    "bot_ngot":        "monosodium glutamate MSG",
}

# Hardcoded fallback per 100g (Tier 4 — approximate, confidence=0.3)
FALLBACK_NUTRITION: Dict[str, MacroNutrients] = {
    "default":    MacroNutrients(calories_kcal=200, protein_g=8,  carbs_g=25, fat_g=8),
    "pizza":      MacroNutrients(calories_kcal=266, protein_g=11, carbs_g=33, fat_g=10),
    "sushi":      MacroNutrients(calories_kcal=143, protein_g=9,  carbs_g=19, fat_g=3),
    "ramen":      MacroNutrients(calories_kcal=436, protein_g=17, carbs_g=60, fat_g=14),
    "pho":        MacroNutrients(calories_kcal=215, protein_g=15, carbs_g=30, fat_g=4),
    "hamburger":  MacroNutrients(calories_kcal=295, protein_g=17, carbs_g=24, fat_g=14),
    "nuoc_mam":       MacroNutrients(calories_kcal=35,  protein_g=5.0, carbs_g=3.6, fat_g=0.0,  fiber_g=0.0),
    "nuoc_mam_nhi":   MacroNutrients(calories_kcal=40,  protein_g=6.0, carbs_g=3.8, fat_g=0.0,  fiber_g=0.0),
    "mam_tom":        MacroNutrients(calories_kcal=62,  protein_g=7.0, carbs_g=3.0, fat_g=1.5,  fiber_g=0.0),
    "mam_ruoc":       MacroNutrients(calories_kcal=55,  protein_g=6.0, carbs_g=2.0, fat_g=1.0,  fiber_g=0.0),
    "mam_nem":        MacroNutrients(calories_kcal=48,  protein_g=5.5, carbs_g=3.5, fat_g=0.8,  fiber_g=0.0),
    "mam_tep":        MacroNutrients(calories_kcal=50,  protein_g=5.0, carbs_g=3.0, fat_g=1.0,  fiber_g=0.0),
    "mam_ba_khia":    MacroNutrients(calories_kcal=60,  protein_g=7.5, carbs_g=1.0, fat_g=2.0,  fiber_g=0.0),
    "mam_ca_sac":     MacroNutrients(calories_kcal=110, protein_g=15.0,carbs_g=0.0, fat_g=5.0,  fiber_g=0.0),
    "mam_ca_linh":    MacroNutrients(calories_kcal=115, protein_g=14.0,carbs_g=0.0, fat_g=6.0,  fiber_g=0.0),
    "tuong_hot":      MacroNutrients(calories_kcal=100, protein_g=6.0, carbs_g=12.0,fat_g=3.0,  fiber_g=1.5),
    "tuong_ban":      MacroNutrients(calories_kcal=98,  protein_g=5.5, carbs_g=11.0,fat_g=3.0,  fiber_g=1.0),
    "tuong_den":      MacroNutrients(calories_kcal=94,  protein_g=4.5, carbs_g=15.0,fat_g=2.0,  fiber_g=0.5),
    "tuong_ot":       MacroNutrients(calories_kcal=56,  protein_g=1.0, carbs_g=12.0,fat_g=0.5,  fiber_g=0.8),
    "nuoc_tuong":     MacroNutrients(calories_kcal=53,  protein_g=8.1, carbs_g=4.9, fat_g=0.1,  fiber_g=0.0),
    "xi_dau":         MacroNutrients(calories_kcal=53,  protein_g=8.1, carbs_g=4.9, fat_g=0.1,  fiber_g=0.0),
    "giam_gao":       MacroNutrients(calories_kcal=18,  protein_g=0.0, carbs_g=0.9, fat_g=0.0,  fiber_g=0.0),
    "giam_nuoi":      MacroNutrients(calories_kcal=18,  protein_g=0.0, carbs_g=0.9, fat_g=0.0,  fiber_g=0.0),
    "com_me":         MacroNutrients(calories_kcal=65,  protein_g=1.5, carbs_g=14.0,fat_g=0.2,  fiber_g=0.0),
    "tai_chua":       MacroNutrients(calories_kcal=38,  protein_g=0.5, carbs_g=9.0, fat_g=0.2,  fiber_g=1.0),
    "qua_doc":        MacroNutrients(calories_kcal=42,  protein_g=0.5, carbs_g=10.0,fat_g=0.3,  fiber_g=0.8),
    "chao_trang":     MacroNutrients(calories_kcal=128, protein_g=9.0, carbs_g=4.0, fat_g=9.0,  fiber_g=0.5),
    "chao_do":        MacroNutrients(calories_kcal=132, protein_g=8.5, carbs_g=5.0, fat_g=9.5,  fiber_g=0.5),
    "dau_mau_dieu":   MacroNutrients(calories_kcal=884, protein_g=0.0, carbs_g=0.0, fat_g=100.0,fiber_g=0.0),
    "dau_me":         MacroNutrients(calories_kcal=884, protein_g=0.0, carbs_g=0.0, fat_g=100.0,fiber_g=0.0),
    "dau_hao":        MacroNutrients(calories_kcal=79,  protein_g=0.8, carbs_g=17.5,fat_g=0.3,  fiber_g=0.0),
    "mo_heo":         MacroNutrients(calories_kcal=900, protein_g=0.0, carbs_g=0.0, fat_g=100.0,fiber_g=0.0),
    "top_mo":         MacroNutrients(calories_kcal=540, protein_g=12.0,carbs_g=0.0, fat_g=55.0, fiber_g=0.0),
    "hanh_tim":       MacroNutrients(calories_kcal=72,  protein_g=2.5, carbs_g=16.8,fat_g=0.1,  fiber_g=3.2),
    "hanh_tay":       MacroNutrients(calories_kcal=40,  protein_g=1.1, carbs_g=9.3, fat_g=0.1,  fiber_g=1.7),
    "toi":            MacroNutrients(calories_kcal=149, protein_g=6.4, carbs_g=33.1,fat_g=0.5,  fiber_g=2.1),
    "gung":           MacroNutrients(calories_kcal=80,  protein_g=1.8, carbs_g=17.8,fat_g=0.8,  fiber_g=2.0),
    "sa":             MacroNutrients(calories_kcal=99,  protein_g=1.8, carbs_g=25.3,fat_g=0.5,  fiber_g=0.0),
    "rieng":          MacroNutrients(calories_kcal=63,  protein_g=1.0, carbs_g=15.0,fat_g=0.5,  fiber_g=0.0),
    "nghe_tuoi":      MacroNutrients(calories_kcal=80,  protein_g=2.0, carbs_g=17.0,fat_g=1.0,  fiber_g=2.0),
    "cu_nen":         MacroNutrients(calories_kcal=70,  protein_g=2.2, carbs_g=15.5,fat_g=0.2,  fiber_g=2.0),
    "hanh_la":        MacroNutrients(calories_kcal=32,  protein_g=1.8, carbs_g=7.3, fat_g=0.2,  fiber_g=2.6),
    "ngo_ri":         MacroNutrients(calories_kcal=23,  protein_g=2.1, carbs_g=3.7, fat_g=0.5,  fiber_g=2.8),
    "ngo_gai":        MacroNutrients(calories_kcal=25,  protein_g=2.0, carbs_g=4.0, fat_g=0.5,  fiber_g=2.5),
    "rau_ngo":        MacroNutrients(calories_kcal=26,  protein_g=2.0, carbs_g=4.5, fat_g=0.4,  fiber_g=2.0),
    "hung_que":       MacroNutrients(calories_kcal=22,  protein_g=3.2, carbs_g=2.7, fat_g=0.6,  fiber_g=1.6),
    "hung_lui":       MacroNutrients(calories_kcal=44,  protein_g=3.3, carbs_g=8.4, fat_g=0.7,  fiber_g=6.8),
    "tia_to":         MacroNutrients(calories_kcal=37,  protein_g=3.8, carbs_g=7.0, fat_g=0.1,  fiber_g=7.0),
    "kinh_gioi":      MacroNutrients(calories_kcal=20,  protein_g=1.5, carbs_g=4.0, fat_g=0.3,  fiber_g=1.5),
    "rau_ram":        MacroNutrients(calories_kcal=25,  protein_g=2.0, carbs_g=5.0, fat_g=0.2,  fiber_g=2.0),
    "dieu_ca":        MacroNutrients(calories_kcal=20,  protein_g=1.5, carbs_g=4.0, fat_g=0.2,  fiber_g=1.8),
    "thi_la":         MacroNutrients(calories_kcal=43,  protein_g=3.5, carbs_g=7.0, fat_g=1.1,  fiber_g=2.1),
    "rau_hung_cay":   MacroNutrients(calories_kcal=70,  protein_g=3.8, carbs_g=15.0,fat_g=0.9,  fiber_g=8.0),
    "la_lot":         MacroNutrients(calories_kcal=40,  protein_g=3.0, carbs_g=6.0, fat_g=1.0,  fiber_g=3.5),
    "la_chanh":       MacroNutrients(calories_kcal=30,  protein_g=1.5, carbs_g=6.0, fat_g=0.5,  fiber_g=3.0),
    "la_dua":         MacroNutrients(calories_kcal=85,  protein_g=2.0, carbs_g=20.0,fat_g=0.1,  fiber_g=3.0),
    "la_mac_mat":     MacroNutrients(calories_kcal=35,  protein_g=1.5, carbs_g=7.0, fat_g=0.5,  fiber_g=2.5),
    "la_cach":        MacroNutrients(calories_kcal=30,  protein_g=2.5, carbs_g=5.0, fat_g=0.5,  fiber_g=3.0),
    "la_giang":       MacroNutrients(calories_kcal=28,  protein_g=1.8, carbs_g=5.5, fat_g=0.3,  fiber_g=2.5),
    "ot_hiem":        MacroNutrients(calories_kcal=40,  protein_g=2.0, carbs_g=9.0, fat_g=0.4,  fiber_g=1.5),
    "ot_sung":        MacroNutrients(calories_kcal=35,  protein_g=1.5, carbs_g=8.0, fat_g=0.3,  fiber_g=1.2),
    "tieu_den":       MacroNutrients(calories_kcal=255, protein_g=10.4,carbs_g=64.8,fat_g=3.3,  fiber_g=25.3),
    "tieu_so":        MacroNutrients(calories_kcal=296, protein_g=10.4,carbs_g=68.6,fat_g=2.1,  fiber_g=26.2),
    "tieu_xanh":      MacroNutrients(calories_kcal=250, protein_g=10.0,carbs_g=62.0,fat_g=3.0,  fiber_g=24.0),
    "rau_muong":      MacroNutrients(calories_kcal=19,  protein_g=2.6, carbs_g=3.1, fat_g=0.2,  fiber_g=2.1),
    "rau_cai_xanh":   MacroNutrients(calories_kcal=22,  protein_g=2.2, carbs_g=4.0, fat_g=0.2,  fiber_g=1.8),
    "rau_cai_ngot":   MacroNutrients(calories_kcal=15,  protein_g=1.4, carbs_g=2.6, fat_g=0.2,  fiber_g=1.2),
    "rau_cai_cuc":    MacroNutrients(calories_kcal=22,  protein_g=1.6, carbs_g=4.2, fat_g=0.3,  fiber_g=2.5),
    "rau_cai_thia":   MacroNutrients(calories_kcal=13,  protein_g=1.5, carbs_g=2.2, fat_g=0.2,  fiber_g=1.0),
    "mong_toi":       MacroNutrients(calories_kcal=19,  protein_g=1.8, carbs_g=3.4, fat_g=0.3,  fiber_g=0.9),
    "rau_ngot":       MacroNutrients(calories_kcal=60,  protein_g=6.0, carbs_g=12.0,fat_g=0.5,  fiber_g=5.0),
    "xa_lach":        MacroNutrients(calories_kcal=15,  protein_g=1.4, carbs_g=2.9, fat_g=0.2,  fiber_g=1.3),
    "rau_dang":       MacroNutrients(calories_kcal=25,  protein_g=2.0, carbs_g=5.0, fat_g=0.2,  fiber_g=2.0),
    "cai_bap":        MacroNutrients(calories_kcal=25,  protein_g=1.3, carbs_g=5.8, fat_g=0.1,  fiber_g=2.5),
    "rau_den":        MacroNutrients(calories_kcal=23,  protein_g=2.5, carbs_g=4.0, fat_g=0.2,  fiber_g=2.2),
    "kho_qua":        MacroNutrients(calories_kcal=17,  protein_g=1.0, carbs_g=3.7, fat_g=0.2,  fiber_g=2.8),
    "bau":            MacroNutrients(calories_kcal=14,  protein_g=0.6, carbs_g=3.4, fat_g=0.0,  fiber_g=0.5),
    "bi_xanh":        MacroNutrients(calories_kcal=13,  protein_g=0.4, carbs_g=3.0, fat_g=0.2,  fiber_g=0.5),
    "bi_do":          MacroNutrients(calories_kcal=26,  protein_g=1.0, carbs_g=6.5, fat_g=0.1,  fiber_g=0.5),
    "muop_huong":     MacroNutrients(calories_kcal=20,  protein_g=1.2, carbs_g=4.4, fat_g=0.2,  fiber_g=0.5),
    "su_su":          MacroNutrients(calories_kcal=19,  protein_g=0.8, carbs_g=4.5, fat_g=0.1,  fiber_g=1.7),
    "ca_chua":        MacroNutrients(calories_kcal=18,  protein_g=0.9, carbs_g=3.9, fat_g=0.2,  fiber_g=1.2),
    "ca_phao":        MacroNutrients(calories_kcal=25,  protein_g=1.0, carbs_g=5.9, fat_g=0.2,  fiber_g=3.0),
    "ca_tim":         MacroNutrients(calories_kcal=25,  protein_g=1.0, carbs_g=5.9, fat_g=0.2,  fiber_g=3.0),
    "ca_rot":         MacroNutrients(calories_kcal=41,  protein_g=0.9, carbs_g=9.6, fat_g=0.2,  fiber_g=2.8),
    "cu_cai_trang":   MacroNutrients(calories_kcal=18,  protein_g=0.6, carbs_g=4.1, fat_g=0.1,  fiber_g=1.6),
    "su_hao":         MacroNutrients(calories_kcal=27,  protein_g=1.7, carbs_g=6.2, fat_g=0.1,  fiber_g=3.6),
    "khoai_tay":      MacroNutrients(calories_kcal=77,  protein_g=2.0, carbs_g=17.0,fat_g=0.1,  fiber_g=2.2),
    "khoai_lang":     MacroNutrients(calories_kcal=86,  protein_g=1.6, carbs_g=20.1,fat_g=0.1,  fiber_g=3.0),
    "khoai_mon":      MacroNutrients(calories_kcal=112, protein_g=1.5, carbs_g=26.5,fat_g=0.2,  fiber_g=4.1),
    "khoai_so":       MacroNutrients(calories_kcal=105, protein_g=1.4, carbs_g=24.8,fat_g=0.2,  fiber_g=3.5),
    "cu_san":         MacroNutrients(calories_kcal=38,  protein_g=0.7, carbs_g=8.8, fat_g=0.1,  fiber_g=4.9),
    "hoa_chuoi":      MacroNutrients(calories_kcal=37,  protein_g=1.6, carbs_g=9.0, fat_g=0.6,  fiber_g=5.7),
    "ngo_sen":        MacroNutrients(calories_kcal=74,  protein_g=2.6, carbs_g=17.2,fat_g=0.1,  fiber_g=4.9),
    "gia_do":         MacroNutrients(calories_kcal=30,  protein_g=3.0, carbs_g=5.9, fat_g=0.2,  fiber_g=1.8),
    "mang_tuoi":      MacroNutrients(calories_kcal=27,  protein_g=2.6, carbs_g=5.2, fat_g=0.3,  fiber_g=2.2),
    "mang_kho":       MacroNutrients(calories_kcal=260, protein_g=25.0,carbs_g=49.0,fat_g=3.0,  fiber_g=15.0),
    "mang_chua":      MacroNutrients(calories_kcal=22,  protein_g=2.0, carbs_g=4.5, fat_g=0.2,  fiber_g=2.0),
    "doc_mung":       MacroNutrients(calories_kcal=13,  protein_g=0.8, carbs_g=3.0, fat_g=0.1,  fiber_g=0.8),
    "nam_rom":        MacroNutrients(calories_kcal=22,  protein_g=3.0, carbs_g=4.0, fat_g=0.3,  fiber_g=1.0),
    "moc_nhi":        MacroNutrients(calories_kcal=25,  protein_g=1.5, carbs_g=6.8, fat_g=0.2,  fiber_g=6.5),
    "nam_huong":      MacroNutrients(calories_kcal=34,  protein_g=2.2, carbs_g=6.8, fat_g=0.5,  fiber_g=2.5),
    "nam_kim_cham":   MacroNutrients(calories_kcal=37,  protein_g=2.7, carbs_g=8.1, fat_g=0.3,  fiber_g=2.7),
    "nam_bao_ngu":    MacroNutrients(calories_kcal=33,  protein_g=3.3, carbs_g=6.1, fat_g=0.4,  fiber_g=2.3),
    "gao_te":         MacroNutrients(calories_kcal=365, protein_g=7.1, carbs_g=80.0,fat_g=0.7,  fiber_g=1.3),
    "gao_nep":        MacroNutrients(calories_kcal=360, protein_g=6.8, carbs_g=79.0,fat_g=0.6,  fiber_g=0.5),
    "com":            MacroNutrients(calories_kcal=130, protein_g=2.7, carbs_g=28.2,fat_g=0.3,  fiber_g=0.4),
    "cot_m":          MacroNutrients(calories_kcal=120, protein_g=3.5, carbs_g=25.0,fat_g=0.5,  fiber_g=0.5),
    "bun":            MacroNutrients(calories_kcal=109, protein_g=2.0, carbs_g=24.2,fat_g=0.2,  fiber_g=0.4),
    "banh_pho":       MacroNutrients(calories_kcal=109, protein_g=2.0, carbs_g=24.0,fat_g=0.2,  fiber_g=0.4),
    "banh_canh":      MacroNutrients(calories_kcal=130, protein_g=2.0, carbs_g=29.0,fat_g=0.3,  fiber_g=0.3),
    "hu_tieu":        MacroNutrients(calories_kcal=109, protein_g=2.0, carbs_g=24.0,fat_g=0.2,  fiber_g=0.4),
    "mien_dong":      MacroNutrients(calories_kcal=352, protein_g=0.2, carbs_g=86.0,fat_g=0.0,  fiber_g=0.8),
    "mi_trung":       MacroNutrients(calories_kcal=138, protein_g=5.0, carbs_g=25.0,fat_g=2.0,  fiber_g=0.8),
    "banh_trang":     MacroNutrients(calories_kcal=335, protein_g=2.5, carbs_g=81.0,fat_g=0.5,  fiber_g=0.5),
    "banh_da_cua":    MacroNutrients(calories_kcal=320, protein_g=5.0, carbs_g=73.0,fat_g=0.5,  fiber_g=0.3),
    "banh_hoi":       MacroNutrients(calories_kcal=115, protein_g=2.2, carbs_g=26.0,fat_g=0.2,  fiber_g=0.3),
    "bot_gao":        MacroNutrients(calories_kcal=366, protein_g=6.0, carbs_g=80.0,fat_g=0.6,  fiber_g=2.4),
    "bot_nang":       MacroNutrients(calories_kcal=357, protein_g=0.2, carbs_g=88.0,fat_g=0.0,  fiber_g=0.9),
    "bot_nep":        MacroNutrients(calories_kcal=362, protein_g=6.5, carbs_g=79.5,fat_g=0.5,  fiber_g=1.0),
    "bot_chien_xu":   MacroNutrients(calories_kcal=395, protein_g=13.5,carbs_g=72.0,fat_g=5.8,  fiber_g=4.0),
    "bot_banh_xeo":   MacroNutrients(calories_kcal=360, protein_g=5.0, carbs_g=79.0,fat_g=1.0,  fiber_g=1.5),
    "thit_ba_chi":    MacroNutrients(calories_kcal=518, protein_g=9.0, carbs_g=0.0, fat_g=53.0, fiber_g=0.0),
    "thit_nac_vai":   MacroNutrients(calories_kcal=216, protein_g=19.0,carbs_g=0.0, fat_g=15.0, fiber_g=0.0),
    "thit_nac_dam":   MacroNutrients(calories_kcal=242, protein_g=20.0,carbs_g=0.0, fat_g=17.0, fiber_g=0.0),
    "suon_non":       MacroNutrients(calories_kcal=321, protein_g=15.0,carbs_g=0.0, fat_g=28.0, fiber_g=0.0),
    "mong_gio":       MacroNutrients(calories_kcal=248, protein_g=17.0,carbs_g=0.0, fat_g=20.0, fiber_g=0.0),
    "da_heo":         MacroNutrients(calories_kcal=346, protein_g=20.0,carbs_g=0.0, fat_g=29.0, fiber_g=0.0),
    "tai_heo":        MacroNutrients(calories_kcal=176, protein_g=22.5,carbs_g=0.4, fat_g=9.0,  fiber_g=0.0),
    "long_heo":       MacroNutrients(calories_kcal=106, protein_g=16.0,carbs_g=0.0, fat_g=4.7,  fiber_g=0.0),
    "gio_song":       MacroNutrients(calories_kcal=265, protein_g=15.0,carbs_g=2.0, fat_g=22.0, fiber_g=0.0),
    "thit_bo_than":   MacroNutrients(calories_kcal=207, protein_g=26.0,carbs_g=0.0, fat_g=11.0, fiber_g=0.0),
    "bap_bo":         MacroNutrients(calories_kcal=180, protein_g=27.0,carbs_g=0.0, fat_g=7.0,  fiber_g=0.0),
    "nam_bo":         MacroNutrients(calories_kcal=258, protein_g=17.0,carbs_g=0.0, fat_g=21.0, fiber_g=0.0),
    "gan_bo":         MacroNutrients(calories_kcal=150, protein_g=28.0,carbs_g=0.0, fat_g=4.0,  fiber_g=0.0),
    "duoi_bo":        MacroNutrients(calories_kcal=263, protein_g=18.0,carbs_g=0.0, fat_g=21.0, fiber_g=0.0),
    "ga_ta":          MacroNutrients(calories_kcal=189, protein_g=20.0,carbs_g=0.0, fat_g=12.0, fiber_g=0.0),
    "vit":            MacroNutrients(calories_kcal=337, protein_g=19.0,carbs_g=0.0, fat_g=28.0, fiber_g=0.0),
    "ngan":           MacroNutrients(calories_kcal=220, protein_g=19.0,carbs_g=0.0, fat_g=15.0, fiber_g=0.0),
    "chim_bo_cau":    MacroNutrients(calories_kcal=294, protein_g=18.0,carbs_g=0.0, fat_g=24.0, fiber_g=0.0),
    "trung_ga":       MacroNutrients(calories_kcal=155, protein_g=13.0,carbs_g=1.1, fat_g=11.0, fiber_g=0.0),
    "trung_vit":      MacroNutrients(calories_kcal=185, protein_g=13.0,carbs_g=1.4, fat_g=14.0, fiber_g=0.0),
    "trung_cut":      MacroNutrients(calories_kcal=158, protein_g=13.1,carbs_g=0.4, fat_g=11.1, fiber_g=0.0),
    "trung_vit_lon":  MacroNutrients(calories_kcal=182, protein_g=13.6,carbs_g=6.0, fat_g=12.0, fiber_g=0.0),
    "trung_bac_thao": MacroNutrients(calories_kcal=170, protein_g=13.8,carbs_g=1.2, fat_g=12.5, fiber_g=0.0),
    "trung_muoi":     MacroNutrients(calories_kcal=203, protein_g=14.0,carbs_g=1.8, fat_g=16.0, fiber_g=0.0),
    "ca_loc":         MacroNutrients(calories_kcal=84,  protein_g=18.0,carbs_g=0.0, fat_g=0.8,  fiber_g=0.0),
    "ca_dieu_hong":   MacroNutrients(calories_kcal=96,  protein_g=20.0,carbs_g=0.0, fat_g=1.7,  fiber_g=0.0),
    "ca_ro":          MacroNutrients(calories_kcal=90,  protein_g=20.0,carbs_g=0.0, fat_g=1.2,  fiber_g=0.0),
    "ca_tre":         MacroNutrients(calories_kcal=144, protein_g=18.0,carbs_g=0.0, fat_g=8.0,  fiber_g=0.0),
    "ca_thu":         MacroNutrients(calories_kcal=205, protein_g=19.0,carbs_g=0.0, fat_g=14.0, fiber_g=0.0),
    "ca_ngu":         MacroNutrients(calories_kcal=144, protein_g=23.0,carbs_g=0.0, fat_g=6.0,  fiber_g=0.0),
    "ca_nuc":         MacroNutrients(calories_kcal=166, protein_g=20.0,carbs_g=0.0, fat_g=9.0,  fiber_g=0.0),
    "ca_bong":        MacroNutrients(calories_kcal=96,  protein_g=20.0,carbs_g=0.0, fat_g=1.5,  fiber_g=0.0),
    "tom_su":         MacroNutrients(calories_kcal=85,  protein_g=20.0,carbs_g=0.0, fat_g=0.6,  fiber_g=0.0),
    "tom_dat":        MacroNutrients(calories_kcal=99,  protein_g=21.0,carbs_g=0.0, fat_g=0.9,  fiber_g=0.0),
    "tom_the":        MacroNutrients(calories_kcal=99,  protein_g=21.0,carbs_g=0.0, fat_g=0.9,  fiber_g=0.0),
    "cua_dong":       MacroNutrients(calories_kcal=92,  protein_g=18.0,carbs_g=0.0, fat_g=1.8,  fiber_g=0.0),
    "cua_bien":       MacroNutrients(calories_kcal=97,  protein_g=18.0,carbs_g=0.0, fat_g=2.3,  fiber_g=0.0),
    "ghe":            MacroNutrients(calories_kcal=97,  protein_g=18.5,carbs_g=0.0, fat_g=2.0,  fiber_g=0.0),
    "muc_la":         MacroNutrients(calories_kcal=92,  protein_g=16.0,carbs_g=3.1, fat_g=1.4,  fiber_g=0.0),
    "muc_ong":        MacroNutrients(calories_kcal=92,  protein_g=16.0,carbs_g=3.1, fat_g=1.4,  fiber_g=0.0),
    "muc_trung":      MacroNutrients(calories_kcal=95,  protein_g=15.5,carbs_g=4.0, fat_g=1.5,  fiber_g=0.0),
    "ngheu":          MacroNutrients(calories_kcal=74,  protein_g=13.0,carbs_g=2.7, fat_g=1.0,  fiber_g=0.0),
    "so_huyet":       MacroNutrients(calories_kcal=80,  protein_g=14.0,carbs_g=4.0, fat_g=1.0,  fiber_g=0.0),
    "oc_buou":        MacroNutrients(calories_kcal=92,  protein_g=16.0,carbs_g=4.0, fat_g=1.4,  fiber_g=0.0),
    "hen":            MacroNutrients(calories_kcal=86,  protein_g=12.0,carbs_g=3.7, fat_g=2.2,  fiber_g=0.0),
    "luon":           MacroNutrients(calories_kcal=184, protein_g=18.0,carbs_g=0.0, fat_g=12.0, fiber_g=0.0),
    "dau_hu_trang":   MacroNutrients(calories_kcal=76,  protein_g=8.0, carbs_g=1.9, fat_g=4.8,  fiber_g=0.3),
    "dau_hu_chien":   MacroNutrients(calories_kcal=175, protein_g=12.0,carbs_g=4.0, fat_g=13.0, fiber_g=0.5),
    "tau_hu_ky":      MacroNutrients(calories_kcal=196, protein_g=17.0,carbs_g=7.0, fat_g=12.0, fiber_g=0.5),
    "hoa_hoi":        MacroNutrients(calories_kcal=337, protein_g=18.0,carbs_g=50.0,fat_g=16.0, fiber_g=15.0),
    "que_chi":        MacroNutrients(calories_kcal=247, protein_g=4.0, carbs_g=81.0,fat_g=1.2,  fiber_g=53.1),
    "thao_qua":       MacroNutrients(calories_kcal=311, protein_g=11.0,carbs_g=68.0,fat_g=7.0,  fiber_g=28.0),
    "hat_mui":        MacroNutrients(calories_kcal=298, protein_g=12.4,carbs_g=55.0,fat_g=17.8, fiber_g=41.9),
    "dinh_huong":     MacroNutrients(calories_kcal=274, protein_g=6.0, carbs_g=66.0,fat_g=13.0, fiber_g=34.2),
    "hat_mac_khen":   MacroNutrients(calories_kcal=285, protein_g=10.0,carbs_g=65.0,fat_g=8.0,  fiber_g=25.0),
    "hat_doi":        MacroNutrients(calories_kcal=280, protein_g=10.0,carbs_g=60.0,fat_g=8.0,  fiber_g=25.0),
    "dau_xanh":       MacroNutrients(calories_kcal=347, protein_g=23.9,carbs_g=62.6,fat_g=1.2,  fiber_g=16.3),
    "dau_phong":      MacroNutrients(calories_kcal=567, protein_g=25.8,carbs_g=16.1,fat_g=49.2, fiber_g=8.5),
    "me_vung":        MacroNutrients(calories_kcal=573, protein_g=17.7,carbs_g=23.5,fat_g=49.7, fiber_g=11.8),
    "hat_dieu":       MacroNutrients(calories_kcal=553, protein_g=18.2,carbs_g=30.2,fat_g=43.8, fiber_g=3.3),
    "me_qua":         MacroNutrients(calories_kcal=239, protein_g=2.8, carbs_g=62.5,fat_g=0.6,  fiber_g=5.1),
    "dua_thom":       MacroNutrients(calories_kcal=50,  protein_g=0.5, carbs_g=13.1,fat_g=0.1,  fiber_g=1.4),
    "xoai_xanh":      MacroNutrients(calories_kcal=60,  protein_g=0.8, carbs_g=15.0,fat_g=0.4,  fiber_g=1.6),
    "khe_chua":       MacroNutrients(calories_kcal=31,  protein_g=1.0, carbs_g=7.0, fat_g=0.3,  fiber_g=2.8),
    "chuoi_chat":     MacroNutrients(calories_kcal=89,  protein_g=1.1, carbs_g=22.8,fat_g=0.3,  fiber_g=2.6),
    "sung":           MacroNutrients(calories_kcal=74,  protein_g=0.8, carbs_g=19.2,fat_g=0.3,  fiber_g=2.9),
    "qua_sau":        MacroNutrients(calories_kcal=50,  protein_g=0.8, carbs_g=12.0,fat_g=0.2,  fiber_g=2.0),
    "muoi_hat":       MacroNutrients(calories_kcal=0,   protein_g=0.0, carbs_g=0.0, fat_g=0.0,  fiber_g=0.0),
    "muoi_ham":       MacroNutrients(calories_kcal=0,   protein_g=0.0, carbs_g=0.0, fat_g=0.0,  fiber_g=0.0),
    "duong_cat":      MacroNutrients(calories_kcal=387, protein_g=0.0, carbs_g=100.0,fat_g=0.0, fiber_g=0.0),
    "duong_phen":     MacroNutrients(calories_kcal=387, protein_g=0.0, carbs_g=100.0,fat_g=0.0, fiber_g=0.0),
    "duong_thot_not": MacroNutrients(calories_kcal=380, protein_g=0.0, carbs_g=98.0,fat_g=0.0,  fiber_g=0.0),
    "hat_nem":        MacroNutrients(calories_kcal=180, protein_g=15.0,carbs_g=25.0,fat_g=2.0,  fiber_g=0.0),
    "bot_ngot":       MacroNutrients(calories_kcal=0,   protein_g=0.0, carbs_g=0.0, fat_g=0.0,  fiber_g=0.0),
}

# Confidence constants per tier
_CONF = {
    "verified": 1.00,
    "pgvector":  0.90,
    "usda":      0.75,
    "fallback":  0.30,
}

# ── Internal type alias ──────────────────────────────────────
_LookupResult = Tuple[MacroNutrients, Optional[str], str, float]
# (macros_per_100g, fdc_id, data_source, confidence)


def _scale(macro: MacroNutrients, grams: float) -> MacroNutrients:
    ratio = grams / 100.0
    return MacroNutrients(
        calories_kcal=round(macro.calories_kcal * ratio, 1),
        protein_g=round(macro.protein_g * ratio, 1),
        carbs_g=round(macro.carbs_g * ratio, 1),
        fat_g=round(macro.fat_g * ratio, 1),
        fiber_g=round((macro.fiber_g or 0) * ratio, 1) if macro.fiber_g is not None else None,
    )


def _extract_macros(food_json: dict) -> MacroNutrients:
    nutrients = {n["nutrientId"]: n.get("value", 0)
                 for n in food_json.get("foodNutrients", [])}
    return MacroNutrients(
        calories_kcal=nutrients.get(_NUT_ENERGY, 200),
        protein_g=nutrients.get(_NUT_PROTEIN, 0),
        carbs_g=nutrients.get(_NUT_CARBS, 0),
        fat_g=nutrients.get(_NUT_FAT, 0),
        fiber_g=nutrients.get(_NUT_FIBER),
    )


def _usda_name_matches(query_text: str, usda_description: str) -> bool:
    """
    Accept a USDA result only if the returned food description is semantically
    close enough to what we queried.  Two acceptance criteria (OR logic):
      1. SequenceMatcher ratio >= threshold
      2. At least one meaningful word (≥4 chars) from query_text appears in
         the USDA description
    This prevents cases like query="tom_su" (shrimp) matching "Tomato sauce".
    """
    threshold = settings.usda_name_match_threshold
    q = query_text.lower()
    d = usda_description.lower()
    ratio = SequenceMatcher(None, q, d).ratio()
    word_hit = any(w in d for w in q.split() if len(w) >= 4)
    return ratio >= threshold or word_hit


def _macros_to_cache_dict(
    macros: MacroNutrients,
    fdc_id: Optional[str],
    data_source: str,
    confidence: float,
) -> dict:
    return {
        "macros": {
            "calories_kcal": macros.calories_kcal,
            "protein_g": macros.protein_g,
            "carbs_g": macros.carbs_g,
            "fat_g": macros.fat_g,
            "fiber_g": macros.fiber_g,
        },
        "fdc_id": fdc_id,
        "data_source": data_source,
        "confidence": confidence,
    }


def _macros_from_cache_dict(data: dict) -> _LookupResult:
    return (
        MacroNutrients(**data["macros"]),
        data.get("fdc_id"),
        data.get("data_source", "unknown"),
        data.get("confidence", 1.0),
    )


class NutritionService:

    async def _fetch_nutrition(self, query: str) -> _LookupResult:
        """
        5-tier lookup.  Returns (macros_per_100g, fdc_id, data_source, confidence).
        """
        search_text = VIET_TO_USDA_QUERY.get(query, query.replace("_", " "))

        # ── Tier 0: admin-verified (always checked first, never cached) ───────
        verified = await get_verified_food(query)
        if verified is not None:
            logger.info("nutrition_tier0_verified", label=query)
            return verified, None, "verified", _CONF["verified"]

        # ── Tier 1: Redis cache (stores result from tier 2/3/4) ───────────────
        cached = await redis_cache.get_nutrition(query)
        if cached is not None:
            logger.debug("nutrition_tier1_cache", label=query,
                         source=cached.get("data_source"))
            return _macros_from_cache_dict(cached)

        # ── Tier 2: Supabase pgvector semantic search ─────────────────────────
        vector_result = await match_food(label=query, search_text=search_text)
        if vector_result is not None:
            macros, fdc_id = vector_result
            result: _LookupResult = (macros, fdc_id, "pgvector", _CONF["pgvector"])
            await redis_cache.set_nutrition(
                query, _macros_to_cache_dict(*result), settings.nutrition_cache_ttl
            )
            return result

        # ── Tier 3: USDA FoodData Central ────────────────────────────────────
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{settings.usda_base_url}/foods/search",
                    params={
                        "query": search_text,
                        "api_key": settings.usda_api_key,
                        "pageSize": 1,
                        "dataType": "Foundation,SR Legacy",
                    },
                )
                resp.raise_for_status()
                foods = resp.json().get("foods", [])

                if foods:
                    food = foods[0]
                    usda_desc = food.get("description", "")

                    if _usda_name_matches(search_text, usda_desc):
                        macros = _extract_macros(food)
                        fdc_id = str(food.get("fdcId"))
                        result = (macros, fdc_id, "usda", _CONF["usda"])

                        # Persist to pgvector for future tier-2 hits
                        asyncio.ensure_future(
                            upsert_food(
                                label=query,
                                display_name=search_text,
                                search_text=search_text,
                                macros=macros,
                                fdc_id=fdc_id,
                                source="usda",
                            )
                        )
                        await redis_cache.set_nutrition(
                            query,
                            _macros_to_cache_dict(*result),
                            settings.nutrition_cache_ttl,
                        )
                        logger.info("nutrition_tier3_usda", label=query,
                                    usda_desc=usda_desc, fdc_id=fdc_id)
                        return result
                    else:
                        # Step 4 audit: USDA returned unrelated food — log for admin review
                        logger.warning(
                            "nutrition_usda_name_mismatch",
                            label=query,
                            search_text=search_text,
                            usda_returned=usda_desc,
                            action="rejected_falling_to_fallback",
                        )
                else:
                    logger.warning("nutrition_usda_no_results",
                                   label=query, search_text=search_text)

        except Exception as exc:
            logger.warning("nutrition_usda_failed", label=query, error=str(exc))

        # ── Tier 4: hardcoded fallback ────────────────────────────────────────
        fallback = FALLBACK_NUTRITION.get(query, FALLBACK_NUTRITION["default"])
        # Step 4 audit: log every fallback hit for admin review
        logger.warning(
            "nutrition_tier4_fallback",
            label=query,
            search_text=search_text,
            has_specific_entry=query in FALLBACK_NUTRITION,
            action="review_and_add_to_verified_table",
        )
        result = (fallback, None, "fallback", _CONF["fallback"])
        await redis_cache.set_nutrition(
            query, _macros_to_cache_dict(*result), settings.nutrition_cache_ttl
        )
        return result

    async def lookup_batch(self, detected: List[DetectedFood]) -> List[FoodNutrition]:
        results = []
        for food in detected:
            macros_per_100g, fdc_id, data_source, confidence = \
                await self._fetch_nutrition(food.label)
            scaled = _scale(macros_per_100g, food.estimated_grams)
            results.append(FoodNutrition(
                food_label=food.label,
                estimated_grams=food.estimated_grams,
                macros=scaled,
                usda_fdc_id=fdc_id,
                data_source=data_source,
                confidence=confidence,
            ))
        return results

    def sum_macros(self, breakdown: List[FoodNutrition]) -> MacroNutrients:
        return MacroNutrients(
            calories_kcal=round(sum(f.macros.calories_kcal for f in breakdown), 1),
            protein_g=round(sum(f.macros.protein_g for f in breakdown), 1),
            carbs_g=round(sum(f.macros.carbs_g for f in breakdown), 1),
            fat_g=round(sum(f.macros.fat_g for f in breakdown), 1),
            fiber_g=round(sum(f.macros.fiber_g or 0 for f in breakdown), 1),
        )


# Singleton
nutrition_service = NutritionService()
