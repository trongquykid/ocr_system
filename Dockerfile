# Sử dụng image chính thức của Python
FROM python:3.9.10

# Đặt biến môi trường để đảm bảo các output của Python không bị buffer
ENV PYTHONUNBUFFERED=1

# Tạo và đặt thư mục làm việc cho container
WORKDIR /app

# Sao chép file requirements.txt vào thư mục làm việc
COPY requirements.txt .

# Cài đặt các package trong requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào thư mục làm việc
COPY . .

# Chạy lệnh để khởi động
CMD ["sh"]
