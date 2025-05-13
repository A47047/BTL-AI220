Sau khi download file zip về máy
giải nén file đó
Vào terminal và chạy:
chuyển sang thư mục đang lưu trữ file spam-email bằng lệch: cd "dường dẫn tời thư mục"
pip install -r requirements.txt
chạy app
streamlit run spam-email/app/app.py

nếu chưa có thư viện streamlit dùng lệnh : pip install streamlit
nếu vẫn không được chạy thẳng trực tiếp:
"$env:USERPROFILE\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0
\LocalCache\local-packages\Python311\Scripts\streamlit.exe" run spam-email/app/app.py
