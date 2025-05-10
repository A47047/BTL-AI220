from flask import Flask, render_template, request
import joblib
from preprocess import preprocess_data

app = Flask(__name__)

# Tải mô hình và các đối tượng cần thiết
model = joblib.load('naive_bayes_model.joblib')
vectorizer = joblib.load('naive_bayes_vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

@app.route('/')
def home():
    return render_template('index.html')  # Trang chủ của web

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu đầu vào từ người dùng
    message = request.form['message']
    X_new, _, _ = preprocess_data(pd.Series([message]))
    X_new_tfidf = vectorizer.transform(X_new)
    
    # Dự đoán với mô hình
    prediction = model.predict(X_new_tfidf)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    return render_template('result.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
