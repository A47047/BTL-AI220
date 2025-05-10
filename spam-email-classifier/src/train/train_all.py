# from preprocess import load_data, preprocess_data, vectorize_data
from src.train.preprocess import load_data, preprocess_data  # Đã bỏ vectorize_data
from src.train.train import train_model


def train_all_models():
    # Danh sách các mô hình cần huấn luyện
    models = [
        'naive_bayes', 
        'logistic_regression', 
        'svc', 
        'decision_tree', 
        'knn'
    ]

    # Huấn luyện tất cả các mô hình
    for model in models:
        print(f"\nĐang huấn luyện mô hình: {model}")
        train_model(model)

if __name__ == '__main__':
    train_all_models()
