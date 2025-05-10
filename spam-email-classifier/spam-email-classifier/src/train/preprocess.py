import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import string
import re
import joblib

# Tiền xử lý văn bản (loại bỏ dấu câu, chuyển thành chữ thường, xóa các từ dừng)
def preprocess_text(text):
    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Chuyển sang chữ thường
    text = text.lower()
    # Loại bỏ các ký tự không phải chữ
    text = re.sub(r'\d+', '', text)
    return text

# Đọc dữ liệu từ file CSV
def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Tiền xử lý dữ liệu cho cột 'Message'
    data['cleaned_message'] = data['Message'].apply(preprocess_text)

    # Encode nhãn 'spam' -> 1, 'ham' -> 0
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['Category'])
    
    X = data['cleaned_message']
    y = data['label']
    
    return X, y, label_encoder

def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

def train_model(file_path):
    # Đọc và tiền xử lý dữ liệu
    data = load_data(file_path)
    X, y, label_encoder = preprocess_data(data)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Vector hóa dữ liệu
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_data(X_train, X_test)

    # Lưu vectorizer và label encoder
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')

    print("Đã lưu vectorizer và label encoder thành công.")

# Thực thi quá trình huấn luyện mô hình
if __name__ == "__main__":
    train_model('/workspaces/BTL-AI220/spam-email-classifier/data/email.csv')
