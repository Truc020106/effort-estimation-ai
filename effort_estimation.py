# ============================================================
# ENHANCING AGILE EFFORT ESTIMATION (FINAL CLEAN VERSION)
# ============================================================

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

np.random.seed(42)

print("🚀 CHẠY PHIÊN BẢN CHUẨN")

# ============================================================
# 1. TẠO DỮ LIỆU
# ============================================================

vocab = {
    'XS': ["fix typo", "change color", "update label", "fix css", "adjust margin"],
    'S': ["add validation", "fix login bug", "create api endpoint", "add search feature"],
    'M': ["implement authentication", "build dashboard", "create reporting module"],
    'L': ["build full system", "implement microservices", "design architecture"],
    'XL': ["rewrite entire system", "build ML pipeline", "global deployment system"]
}

def generate_data(n_per_class=400, noise_ratio=0.2):
    texts, labels = [], []
    classes = list(vocab.keys())

    for label in classes:
        for i in range(n_per_class):
            base = np.random.choice(vocab[label])
            text = f"As a user I want to {base}"

            # thêm noise
            if np.random.rand() < noise_ratio:
                wrong_label = np.random.choice([c for c in classes if c != label])
                labels.append(wrong_label)
            else:
                labels.append(label)

            texts.append(text)

    return pd.DataFrame({'text': texts, 'label': labels})

df = generate_data()

print(f"Tổng số mẫu: {len(df)}")
print(df['label'].value_counts())

# ============================================================
# 2. TIỀN XỬ LÝ
# ============================================================

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

df['clean'] = df['text'].apply(preprocess)

# ============================================================
# 3. MODEL KHÔNG FILTER
# ============================================================

print("\n===== THÍ NGHIỆM 1: KHÔNG FILTER =====")

tfidf_1 = TfidfVectorizer()
X1 = tfidf_1.fit_transform(df['clean'])
y1 = df['label']

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, stratify=y1, random_state=42)

model1 = LinearSVC()
model1.fit(X_train1, y_train1)

y_pred1 = model1.predict(X_test1)

p1 = precision_score(y_test1, y_pred1, average='macro')
r1 = recall_score(y_test1, y_pred1, average='macro')
f1_1 = f1_score(y_test1, y_pred1, average='macro')

print(f"Precision: {p1:.2f}")
print(f"Recall: {r1:.2f}")
print(f"F1: {f1_1:.2f}")

# ============================================================
# 4. NOISE FILTERING
# ============================================================

print("\n===== NOISE FILTERING =====")

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X1)

df['cluster'] = clusters

cluster_map = {}
for c in range(5):
    mask = df['cluster'] == c
    dominant = df.loc[mask, 'label'].value_counts().idxmax()
    cluster_map[c] = dominant

df['cluster_label'] = df['cluster'].map(cluster_map)

filtered = []

for lbl in ['XS','S','M','L','XL']:
    subset = df[df['label'] == lbl]

    good = subset[subset['cluster_label'] == lbl]

    # nếu bị mất quá nhiều thì giữ thêm 30%
    if len(good) < 0.5 * len(subset):
        extra = subset.sample(frac=0.3, random_state=42)
        good = pd.concat([good, extra])

    filtered.append(good)

df_filtered = pd.concat(filtered)

print(f"Sau filtering: {len(df_filtered)} mẫu")
print(df_filtered['label'].value_counts())

# ============================================================
# 5. MODEL SAU FILTER (MODEL CHÍNH)
# ============================================================

print("\n===== THÍ NGHIỆM 2: CÓ FILTER =====")

tfidf_final = TfidfVectorizer()
X2 = tfidf_final.fit_transform(df_filtered['clean'])
y2 = df_filtered['label']

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y2, test_size=0.2, stratify=y2, random_state=42)

model_final = LinearSVC()
model_final.fit(X_train2, y_train2)

y_pred2 = model_final.predict(X_test2)

p2 = precision_score(y_test2, y_pred2, average='macro')
r2 = recall_score(y_test2, y_pred2, average='macro')
f1_2 = f1_score(y_test2, y_pred2, average='macro')

print(f"Precision: {p2:.2f}")
print(f"Recall: {r2:.2f}")
print(f"F1: {f1_2:.2f}")

# ============================================================
# 6. SO SÁNH
# ============================================================

print("\n===== KẾT QUẢ =====")
print(f"F1 trước: {f1_1:.2f}")
print(f"F1 sau:   {f1_2:.2f}")
print(f"Tăng:     {(f1_2 - f1_1):.2f}")

# ============================================================
# 7. DỰ ĐOÁN THỰC TẾ
# ============================================================

print("\n" + "="*60)
print("DỰ ĐOÁN KÍCH THƯỚC CÔNG VIỆC")
print("="*60)

SIZE_TO_TIME = {
    "XS": "1–2 giờ",
    "S":  "1 ngày",
    "M":  "2–3 ngày",
    "L":  "1 tuần",
    "XL": "2+ tuần"
}

def predict_story(text):
    text_clean = preprocess(text)
    vec = tfidf_final.transform([text_clean])
    pred = model_final.predict(vec)[0]
    return pred, SIZE_TO_TIME[pred]

while True:
    user_input = input("\n👉 Nhập user story (hoặc 'exit'): ")

    if user_input.lower() == 'exit':
        print("👋 Thoát!")
        break

    size, time_est = predict_story(user_input)

    print("\n📊 KẾT QUẢ:")
    print(f"👉 Size: {size}")
    print(f"⏱️ Thời gian: {time_est}")