FROM python:3.10.5
WORKDIR /recom-app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]