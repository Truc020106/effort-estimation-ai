# 🚀 Effort Estimation using NLP

## 📌 Giới thiệu

Dự án sử dụng **Machine Learning + NLP** để:

* Phân tích **User Story**
* Dự đoán kích thước công việc: `XS, S, M, L, XL`
* Ước lượng thời gian thực hiện

---

## 🧠 Công nghệ sử dụng

* Python
* TF-IDF (Text Vectorization)
* LinearSVC (Machine Learning)
* KMeans (Noise Filtering)

---

## ⚙️ Cách hoạt động

1. Nhập yêu cầu (User Story)
2. Hệ thống xử lý văn bản
3. So sánh với dữ liệu đã học
4. Dự đoán:

   * Kích thước (Size)
   * Thời gian (Time)

---

## ▶️ Cách chạy chương trình

```bash
python effort_estimation.py
```

---

## 💡 Ví dụ

```text
Input: implement authentication with JWT

Output:
Size: M
Time: 2–3 days
```

---

## 📊 Kết quả mô hình

| Metric   | Trước Filtering | Sau Filtering |
| -------- | --------------- | ------------- |
| F1 Score | 0.79            | 0.85          |

👉 Noise Filtering giúp tăng độ chính xác.

---

## 📁 Cấu trúc project

```
effort-estimation-ai/
│
├── effort_estimation.py
├── README.md
├── requirements.txt
```

---

## 🔥 Ý nghĩa

Giúp:

* Ước lượng công việc nhanh
* Hỗ trợ Agile/Scrum
* Giảm sai sót khi planning

---
