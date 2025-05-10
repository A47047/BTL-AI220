import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import string
import re

# Bước 1: Đọc dữ liệu từ file CSV
data = pd.read_csv('email.csv')

# Bước 2: Tiền xử lý dữ liệu
# Loại bỏ dòng có nhãn lỗi (không phải ham/spam)
data = data[data['Category'].isin(['ham', 'spam'])].copy()

# Tiền xử lý văn bản (loại bỏ dấu câu, chuyển thành chữ thường, xóa các từ dừng)
def preprocess_text(text):
    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Chuyển sang chữ thường
    text = text.lower()
    # Loại bỏ các ký tự không phải chữ
    text = re.sub(r'\d+', '', text)
    return text

# Áp dụng tiền xử lý cho cột 'Message'
data['cleaned_message'] = data['Message'].apply(preprocess_text)

# Encode nhãn 'spam' -> 1, 'ham' -> 0
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Category'])

# Chia dữ liệu thành features và labels
X = data['cleaned_message']
y = data['label']

# Bước 3: Chia dữ liệu thành tập huấn luyện và tập kiểm tra (an toàn dùng stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Bước 4: Chuyển đổi văn bản thành các đặc trưng (features)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Xuất mô hình Naive Bayes
import joblib

# Lưu vectorizer, mô hình, và label encoder
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

print("Đã lưu vectorizer và label encoder thành công.")
