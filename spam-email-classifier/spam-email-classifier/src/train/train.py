from src.train.preprocess import load_data, preprocess_data, vectorize_data
# from preprocess import load_data, preprocess_data, vectorize_data
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split

def train_model(model_type='naive_bayes'):
    # Đọc và tiền xử lý dữ liệu
    data = load_data('/workspaces/BTL-AI220/spam-email-classifier/data/email.csv')
    X, y, label_encoder = preprocess_data(data)
    data = data[data['Category'] != '{"mode":"full"']






    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Vector hóa dữ liệu
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_data(X_train, X_test)

    # Chọn mô hình dựa trên tham số model_type
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'svc':
        model = SVC()
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    elif model_type == 'knn':
        model = KNeighborsClassifier()
    else:
        raise ValueError(f"Model {model_type} không hợp lệ!")

    # Huấn luyện mô hình
    model.fit(X_train_tfidf, y_train)

    # Đánh giá mô hình
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{class_report}')

    # Lưu mô hình và vectorizer
    joblib.dump(model, f'{model_type}_model.joblib')
    joblib.dump(vectorizer, f'{model_type}_vectorizer.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')

    print(f"Đã lưu mô hình {model_type} thành công.")

if __name__ == '__main__':
    model_choice = 'naive_bayes'  # Thay đổi mô hình ở đây: 'logistic_regression', 'naive_bayes', 'svm', 'decision_tree', 'knn'
    train_model(model_choice)
