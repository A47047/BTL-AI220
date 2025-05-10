import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from preprocess import preprocess_data, vectorize_data
from sklearn.model_selection import train_test_split

def evaluate_model(model_type, data):
    # Tiền xử lý dữ liệu
    X, y, label_encoder = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Tải mô hình và vectorizer đã huấn luyện
    model = joblib.load(f'{model_type}_model.joblib')
    vectorizer = joblib.load(f'{model_type}_vectorizer.joblib')
    
    # Chuyển đổi dữ liệu vào dạng tf-idf
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Dự đoán với mô hình
    y_pred = model.predict(X_test_tfidf)
    
    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    
    print(f"Đánh giá mô hình: {model_type}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

if __name__ == '__main__':
    # Đọc dữ liệu
    data = pd.read_csv('email.csv')
    
    # Đánh giá các mô hình
    models = ['naive_bayes', 'logistic_regression', 'svm', 'decision_tree', 'knn']
    for model in models:
        evaluate_model(model, data)
