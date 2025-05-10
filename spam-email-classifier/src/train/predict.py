import joblib
import pandas as pd
from preprocess import preprocess_data, vectorize_data

def predict(model_type, input_data):
    # Tải mô hình đã huấn luyện và các đối tượng liên quan
    model = joblib.load(f'{model_type}_model.joblib')
    vectorizer = joblib.load(f'{model_type}_vectorizer.joblib')
    label_encoder = joblib.load('label_encoder.joblib')

    # Tiền xử lý dữ liệu
    X_new, _, _ = preprocess_data(input_data)  # Chỉ cần xử lý X, không cần Y cho dự đoán
    X_new_tfidf = vectorizer.transform(X_new)  # Chuyển đổi dữ liệu đầu vào thành tf-idf

    # Dự đoán
    predictions = model.predict(X_new_tfidf)

    # Chuyển đổi nhãn dự đoán thành tên (ham/spam)
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    return predicted_labels

if __name__ == '__main__':
    # Ví dụ dự đoán với mô hình 'naive_bayes'
    input_data = ['Free money!!! Click now!', 'Meeting at 10am tomorrow.']
    model_type = 'naive_bayes'  # Có thể thay đổi thành các mô hình khác: 'logistic_regression', 'svm', v.v.
    
    result = predict(model_type, input_data)
    for text, label in zip(input_data, result):
        print(f'Message: {text} \nPredicted: {label}\n')
