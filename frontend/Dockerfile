FROM python:3.10

COPY .env .
COPY requirements.txt .
COPY src/ /src

RUN pip install -r requirements.txt

WORKDIR /src

CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]