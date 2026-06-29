# Tài liệu các Models trong Hệ thống MenuGreen

Tài liệu này tổng hợp toàn bộ các mô hình dữ liệu (Models/Entities/DTOs) đang được sử dụng trong hệ thống backend của MenuGreen (bao gồm .NET Web API và Python CV Service).

---

## 1. Database Entities (MenuGreen.DataAccessLayer)
Các thực thể này đại diện cho các bảng trong cơ sở dữ liệu PostgreSQL và được ánh xạ qua Entity Framework Core (EF Core).

### 1.1 Quản lý Tài khoản & Bảo mật (Auth & Accounts)
* **`User`** (`User.cs`): Lưu trữ thông tin tài khoản cơ bản.
  * *Các trường chính:* `Id` (Guid), `Email` (string), `PasswordHash` (string), `EmailConfirmed` (bool), `IsActive` (bool), `RoleId` (Guid), `LastSignInAt`, `CreatedAt`, `UpdatedAt`.
* **`Role`** (`Role.cs`): Phân quyền người dùng (ví dụ: `Admin`, `User`).
  * *Các trường chính:* `Id` (Guid), `Name` (string), `Description` (string).
* **`Session`** (`Session.cs`): Quản lý các phiên đăng nhập và Refresh Token.
  * *Các trường chính:* `Id` (Guid), `UserId` (Guid), `RefreshToken` (string), `ExpiresAt` (DateTime), `IsRevoked` (bool), `CreatedAt`.
* **`EmailVerification`** (`EmailVerification.cs`): OTP xác thực email đăng ký.
* **`PasswordResetToken`** (`PasswordResetToken.cs`): OTP/Token khôi phục mật khẩu.

### 1.2 Hồ sơ & Sức khỏe (Profile & Health)
* **`Profile`** (`Profile.cs`): Hồ sơ cá nhân của người dùng.
  * *Các trường chính:* `Id` (Guid), `UserId` (Guid), `FullName` (string), `PhoneNumber` (string), `Gender` (string), `DateOfBirth` (DateTime?), `AvatarUrl` (string).
* **`HealthProfile`** (`HealthProfile.cs`): Thông số sức khỏe, mục tiêu dinh dưỡng và calo.
  * *Các trường chính:* `Id` (Guid), `UserId` (Guid), `Height` (double), `Weight` (double), `TargetWeight` (double), `ActivityLevel` (string), `DietaryGoal` (string), `DailyCalorieLimit` (double), `DailyProteinLimit`, `DailyCarbsLimit`, `DailyFatLimit`.
* **`WeightLog`** (`WeightLog.cs`): Nhật ký cân nặng theo thời gian để vẽ biểu đồ xu hướng.
* **`NutritionSnapshot`** (`NutritionSnapshot.cs`): Ảnh chụp nhanh lượng calo/macros đã tiêu thụ hàng ngày.

### 1.3 Dị ứng & Danh mục món ăn (Allergies, Foods & Recipes)
* **`Allergy`** (`Allergy.cs`): Danh mục dị ứng master (ví dụ: Hải sản, Đậu phộng, Trứng, Sữa...).
* **`UserAllergy`** (`UserAllergy.cs`): Quan hệ nhiều-nhiều giữa Người dùng và Dị ứng của họ.
* **`Food`** (`Food.cs`): Bảng danh mục thực phẩm cơ bản.
  * *Các trường chính:* `Id` (Guid), `Name` (string), `Barcode` (string), `Brand` (string), `ServingSize` (double), `Calories`, `Protein`, `Carbs`, `Fat`, `Fiber`, `IsVerified`.
* **`Ingredient`** (`Ingredient.cs`): Nguyên liệu thô dùng trong nấu ăn.
* **`Recipe`** (`Recipe.cs`): Công thức nấu ăn chi tiết.
  * *Các trường chính:* `Id` (Guid), `Title` (string), `Description` (string), `Instructions` (string - các bước làm), `PrepTimeMinutes` (int), `CookTimeMinutes` (int), `Difficulty` (string), `Servings` (int).
* **`RecipeIngredient`** (`RecipeIngredient.cs`): Quan hệ nhiều-nhiều giữa công thức và nguyên liệu kèm khối lượng sử dụng.
* **`FoodAllergenTag`** (`FoodAllergenTag.cs`): Bảng gắn thẻ dị ứng (`allergen_key`) vào món ăn để quét nhanh độ an toàn.

### 1.4 Nhật ký Ăn uống & Kế hoạch (Meal Tracking & Planning)
* **`MealLog`** (`MealLog.cs`): Nhật ký bữa ăn hàng ngày của người dùng.
  * *Các trường chính:* `Id` (Guid), `UserId` (Guid), `MealType` (string - `Breakfast`, `Lunch`, `Dinner`, `Snack`), `LogDate` (DateTime), `FoodId` (Guid?), `CustomFoodName` (string), `GramsConsumed` (double), `Calories`, `Protein`, `Carbs`, `Fat`, `Fiber`.
* **`MealPlanHeader`** (`MealPlanHeader.cs`) & **`MealPlanItem`** (`MealPlanItem.cs`): Kế hoạch ăn uống được lên lịch trước cho người dùng.

### 1.5 Giao dịch & Gói thành viên (Subscriptions & Payments)
* **`SubscriptionPlan`** (`SubscriptionPlan.cs`): Các gói dịch vụ trả phí (Premium, Pro...).
* **`UserSubscription`** (`UserSubscription.cs`): Quản lý thời hạn gói dịch vụ đang hoạt động của người dùng.
* **`SepayTransaction`** (`SepayTransaction.cs`): Lưu vết giao dịch tự động đồng bộ từ cổng SePay.

---

## 2. Business Logic DTOs (MenuGreen.BusinessLogicLayer)
Các Data Transfer Objects (DTO) định nghĩa cấu trúc dữ liệu truyền nhận qua API, phân tách giữa Request (nhận từ Client) và Response (trả về Client).

### 2.1 Các Response DTOs nổi bật
* **`CvInferenceResponse`**: Trả về từ quá trình phân tích ảnh AI.
* **`CvSuggestedDish`**: Món ăn gợi ý nhận diện kèm theo cờ kiểm tra dị ứng (`IsSafeForUser`, `MatchedAllergens`).
* **`MealDaySummaryResponse`**: Báo cáo tổng quan dinh dưỡng tiêu thụ trong ngày.
* **`DashboardMetricsResponse`**: Số liệu phân tích dinh dưỡng hiển thị trên trang chủ Dashboard của user.
* **`AllergenRiskResult`**: Kết quả đánh giá mức độ rủi ro dị ứng của món ăn đối với một user cụ thể.
* **`SepayOrderResponse`**: Dữ liệu hóa đơn thanh toán thông qua cổng giao dịch SePay.

---

## 3. Pydantic Schemas (Python CV Service)
Các model định nghĩa bằng thư viện Pydantic (Pydantic v2) trong dịch vụ xử lý ảnh AI FastAPI (`cv-service/app/schemas/cv_schemas.py`).

* **`BoundingBox`**: Tọa độ pixel phát hiện đối tượng trên ảnh (`x1`, `y1`, `x2`, `y2`).
* **`DetectedFood`**: Món ăn/Nguyên liệu được nhận diện bởi mô hình thị giác máy tính.
* **`MacroNutrients`**: Lượng dinh dưỡng đa lượng tiêu chuẩn (`calories_kcal`, `protein_g`, `carbs_g`, `fat_g`, `fiber_g`).
* **`FoodNutrition`**: Thông tin dinh dưỡng đầy đủ của một món ăn cụ thể được định lượng.
* **`AnalysisResult`**: Đầu ra của luồng xử lý ảnh cục bộ (Local Object Detection).
* **`AIInferenceResponse`**: Dữ liệu đầu ra của cuộc hội thoại/phân tích từ mô hình Generative AI (Gemini).
* **`JobStatusResponse`**: Quản lý thông tin trạng thái hàng đợi bất đồng bộ Celery (`queued` | `processing` | `done` | `failed`).
