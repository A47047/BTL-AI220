# Mô hình Neural Network (MLPClassifier)
# --- Bước 1: Import thêm thư viện ---
from sklearn.neural_network import MLPClassifier

# --- Bước 5: Tìm tham số Neural Network tốt nhất ---
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Kiến trúc mạng: 1 lớp 50 node, 1 lớp 100 node, 2 lớp 50 node
    'alpha': [0.0001, 0.001, 0.01],  # L2 regularization (để tránh overfitting)
    'learning_rate_init': [0.001, 0.01]  # Learning rate
}

grid = GridSearchCV(MLPClassifier(max_iter=300, random_state=42), param_grid, cv=5, scoring='f1')
grid.fit(X_train_tfidf, y_train)

# Mô hình tốt nhất và tham số tốt nhất
best_model = grid.best_estimator_
best_params = grid.best_params_

# --- Bước 6: Huấn luyện mô hình Neural Network ---
model = MLPClassifier(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    alpha=best_params['alpha'],
    learning_rate_init=best_params['learning_rate_init'],
    max_iter=300,
    random_state=42
)
model.fit(X_train_tfidf, y_train)

# --- Bước 7: Dự đoán ---
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)[:,1]  # Xác suất lớp spam

# --- Bước 8: Đánh giá mô hình ---
train_accuracy = accuracy_score(y_train, model.predict(X_train_tfidf))
test_accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Mô hình Neural Network (MLPClassifier)")
print(f'Tham số tốt nhất: {best_params}')
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
