"""
Test case để verify fix cho Pydantic ValidationError.

Bug: AI model trả về nguyên liệu thiếu field `do_chinh_xac_uoc_tinh`
Fix: IngredientItem model có default value và response_utils normalize đúng cách
"""
import pytest
from app.schemas.cv_schemas import AIInferenceResponse, IngredientItem


class TestIngredientItemSchema:
    """Test IngredientItem schema với Pydantic v2."""

    def test_ingredient_item_with_do_chinh_xac(self):
        """Ingredient có đầy đủ field do_chinh_xac_uoc_tinh."""
        item = IngredientItem(
            id_nguyen_lieu="raw_1",
            ten_nguyen_lieu="Ức gà",
            ten_nguyen_lieu_ky_thuat="uc_ga",
            khoi_luong_uoc_tinh_g=150.0,
            do_chinh_xac_uoc_tinh="95%",
        )
        assert item.do_chinh_xac_uoc_tinh == "95%"

    def test_ingredient_item_without_do_chinh_xac_uses_default(self):
        """Ingredient thiếu do_chinh_xac_uoc_tinh sẽ dùng default value."""
        item = IngredientItem(
            id_nguyen_lieu="raw_1",
            ten_nguyen_lieu="Ức gà",
            ten_nguyen_lieu_ky_thuat="uc_ga",
            khoi_luong_uoc_tinh_g=150.0,
            # Không truyền do_chinh_xac_uoc_tinh
        )
        # Pydantic v2: default value "unknown" được sử dụng
        assert item.do_chinh_xac_uoc_tinh == "unknown"

    def test_ingredient_item_with_missing_field(self):
        """Ingredient không có field do_chinh_xac_uoc_tinh sẽ dùng default value.
        
        ĐÂY LÀ SCENARIO THỰC TẾ: AI model trả JSON thiếu field,
        không phải truyền explicit None.
        """
        item = IngredientItem(
            id_nguyen_lieu="raw_1",
            ten_nguyen_lieu="Ức gà",
            ten_nguyen_lieu_ky_thuat="uc_ga",
            khoi_luong_uoc_tinh_g=150.0,
            # KHÔNG truyền do_chinh_xac_uoc_tinh - field bị thiếu trong JSON
        )
        # Pydantic v2 với default value "unknown" sẽ tự điền
        assert item.do_chinh_xac_uoc_tinh == "unknown"


class TestAIInferenceResponseWithIncompleteIngredients:
    """Test AIInferenceResponse với dữ liệu không đầy đủ từ AI model."""

    def test_response_with_incomplete_ingredients(self):
        """AI model trả về ingredients thiếu do_chinh_xac_uoc_tinh - không crash."""
        # Simulate AI response với ingredient thiếu field
        payload = {
            "job_id": "test-job-123",
            "request_id": "req-456",
            "status": "done",
            "nguyen_lieu_tho_quet_duoc": [
                {
                    "id_nguyen_lieu": "raw_1",
                    "ten_nguyen_lieu": "Ức gà",
                    "ten_nguyen_lieu_ky_thuat": "uc_ga",
                    "khoi_luong_uoc_tinh_g": 150.0,
                    # THIẾU: do_chinh_xac_uoc_tinh
                },
                {
                    "id_nguyen_lieu": "raw_2",
                    "ten_nguyen_lieu": "Cơm trắng",
                    "ten_nguyen_lieu_ky_thuat": "com_trang",
                    "khoi_luong_uoc_tinh_g": 200.0,
                    "do_chinh_xac_uoc_tinh": "90%",
                },
            ],
        }

        # Trước fix: ValidationError crash
        # Sau fix: Không crash, dùng default value
        response = AIInferenceResponse(**payload)
        
        assert response.job_id == "test-job-123"
        assert response.status == "done"
        assert len(response.nguyen_lieu_tho_quet_duoc) == 2
        
        # Ingredient 1 dùng default
        assert response.nguyen_lieu_tho_quet_duoc[0].do_chinh_xac_uoc_tinh == "unknown"
        # Ingredient 2 có giá trị
        assert response.nguyen_lieu_tho_quet_duoc[1].do_chinh_xac_uoc_tinh == "90%"

    def test_response_with_multiple_incomplete_ingredients(self):
        """Test với nhiều ingredients thiếu field - index 9, 13, 14 như trong logs."""
        ingredients = []
        for i in range(15):
            ing = {
                "id_nguyen_lieu": f"raw_{i}",
                "ten_nguyen_lieu": f"Nguyên liệu {i}",
                "ten_nguyen_lieu_ky_thuat": f"nguyen_lieu_{i}",
                "khoi_luong_uoc_tinh_g": 100.0,
            }
            # Chỉ một số ingredients có do_chinh_xac_uoc_tinh
            if i not in [9, 13, 14]:
                ing["do_chinh_xac_uoc_tinh"] = f"{90 + i}%"
            # else: thiếu field như lỗi trong logs
            ingredients.append(ing)

        payload = {
            "job_id": "test-job-complex",
            "request_id": "req-789",
            "status": "done",
            "nguyen_lieu_tho_quet_duoc": ingredients,
        }

        # Không crash!
        response = AIInferenceResponse(**payload)
        
        assert len(response.nguyen_lieu_tho_quet_duoc) == 15
        # Verify các ingredient thiếu dùng default
        assert response.nguyen_lieu_tho_quet_duoc[9].do_chinh_xac_uoc_tinh == "unknown"
        assert response.nguyen_lieu_tho_quet_duoc[13].do_chinh_xac_uoc_tinh == "unknown"
        assert response.nguyen_lieu_tho_quet_duoc[14].do_chinh_xac_uoc_tinh == "unknown"


class TestResponseNormalization:
    """Test response_utils normalization."""

    def test_normalize_ingredient_item_handles_missing_field(self):
        """normalize_ingredient_item xử lý đúng khi field thiếu."""
        from app.services.response_utils import _normalize_ingredient_item
        
        # Ingredient thiếu do_chinh_xac_uoc_tinh
        raw_item = {
            "id_nguyen_lieu": "raw_1",
            "ten_nguyen_lieu": "Thịt bò",
            "ten_nguyen_lieu_ky_thuat": "thit_bo",
            "khoi_luong_uoc_tinh_g": 200.0,
            # THIẾU: do_chinh_xac_uoc_tinh
        }
        
        normalized = _normalize_ingredient_item(raw_item)
        
        # Đảm bảo field được fill với default
        assert "do_chinh_xac_uoc_tinh" in normalized
        assert normalized["do_chinh_xac_uoc_tinh"] == "unknown"

    def test_normalize_ingredient_item_handles_none_value(self):
        """normalize_ingredient_item xử lý đúng khi field là None."""
        from app.services.response_utils import _normalize_ingredient_item
        
        raw_item = {
            "id_nguyen_lieu": "raw_1",
            "ten_nguyen_lieu": "Thịt bò",
            "ten_nguyen_lieu_ky_thuat": "thit_bo",
            "khoi_luong_uoc_tinh_g": 200.0,
            "do_chinh_xac_uoc_tinh": None,  # Explicit None
        }
        
        normalized = _normalize_ingredient_item(raw_item)
        
        # Đảm bảo None được convert thành "unknown"
        assert normalized["do_chinh_xac_uoc_tinh"] == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
