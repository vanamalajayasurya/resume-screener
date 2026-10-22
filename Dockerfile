From python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "streamlit run app.py --server.port=8000 --server.address=0.0.0.0"]
