import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load các file .pkl đúng đường dẫn
tfidf_vectorizer = joblib.load(os.path.join(BASE_DIR, 'model', 'tfidf_vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'model', 'label_encoder.pkl'))
svc_model = joblib.load(os.path.join(BASE_DIR, 'model', 'SVC', 'svc_best_model.pkl'))


# Giao diện người dùng
st.title("🔍 Email Spam Classifier")
st.write("Nhập nội dung email bạn muốn kiểm tra:")

# Nhập nội dung email
user_input = st.text_area("✉️ Nội dung email:")

# Khi người dùng nhấn nút Dự đoán
if st.button("Dự đoán"):
    if user_input.strip() == "":
        st.warning("⚠️ Vui lòng nhập nội dung email.")
    else:
        # Tiền xử lý văn bản
        input_tfidf = tfidf_vectorizer.transform([user_input])
        
        # Dự đoán với mô hình đã huấn luyện
        prediction = svc_model.predict(input_tfidf)
        result_label = label_encoder.inverse_transform(prediction)[0]
        
        # Hiển thị kết quả
        if result_label == "spam":
            st.error("🚫 Kết quả: SPAM")
        else:
            st.success("✅ Kết quả: HAM (Không phải spam)")
