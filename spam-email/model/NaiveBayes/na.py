# Mô hình Naive Bayes
# --- Bước 1: Import thư viện ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.stem import WordNetLemmatizer

# --- Bước 2: Đọc và làm sạch dữ liệu ---
data = pd.read_csv('email.csv')

# Loại bỏ dòng có nhãn lỗi (không phải ham/spam)
data = data[data['Category'].isin(['ham', 'spam'])].copy()

# Xử lý dữ liệu thiếu (nếu có)
data.dropna(subset=['Message'], inplace=True)

# --- Tiền xử lý văn bản (khởi tạo đối tượng để xử lí văn bản)---
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Chuyển văn bản về chữ thường
    text = text.lower()

    # Tách từ
    words = text.split()

    # Lemmatization: giảm từ về dạng gốc
    words = [lemmatizer.lemmatize(w) for w in words]

    # Trả về văn bản đã xử lý (kết hợp lại các từ thành chuỗi)
    return ' '.join(words)

# Áp dụng tiền xử lý vào cột 'Message'
data['Message'] = data['Message'].apply(preprocess)

# Encode nhãn 'spam' -> 1, 'ham' -> 0
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Category'])

# Chia dữ liệu thành features và labels
X = data['Message']
y = data['label']

# --- Bước 3: Chia train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Bước 4: Vector hóa TF-IDF ---
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, token_pattern=r'\b\w+\b')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Bước 5: Tìm tham số laplace tốt nhất ---
param_grid = {'alpha': np.arange(0.1, 10.0 + 0.1, 0.1)}
grid = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1')
grid.fit(X_train_tfidf, y_train)

# Mô hình tốt nhất và giá trị laplace tốt nhất
best_model = grid.best_estimator_
best_laplace = grid.best_params_['alpha']

# --- Bước 6: Huấn luyện mô hình Naive Bayes ---
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# --- Bước 7: Dự đoán ---
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)[:,1]  # Xác suất dự đoán lớp 1 (spam)

# --- Bước 8: Đánh giá mô hình ---
train_accuracy = accuracy_score(y_train, model.predict(X_train_tfidf))
test_accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Mô hình Naive Bayes")
print(f'Tham số laplace tốt nhất: {best_laplace}')
print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'ROC-AUC Score: {roc_auc:.4f}')
print('Classification Report:')
print(class_report)

# --- Bước 9: Vẽ Confusion Matrix ---
plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- Bước 10: Vẽ ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()
