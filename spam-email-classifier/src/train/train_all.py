from train import train_model

def train_all_models():
    # Danh sách các mô hình cần huấn luyện
    models = [
        'naive_bayes', 
        'logistic_regression', 
        'svm', 
        'decision_tree', 
        'knn'
    ]

    # Huấn luyện tất cả các mô hình
    for model in models:
        print(f"\nĐang huấn luyện mô hình: {model}")
        train_model(model)

if __name__ == '__main__':
    train_all_models()
