FROM python:3.9-slim

EXPOSE 8501

WORKDIR /main

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies.
COPY requirements.txt /main/

# Install dependencies.
RUN pip install -r requirements.txt

# Copy project.
COPY . /main/

CMD streamlit run --server.port $PORT src/main.py
