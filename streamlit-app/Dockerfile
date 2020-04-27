# API_VERSION 1.0
FROM python:3.7-alpine
MAINTAINER reidkr876@gmail.com

WORKDIR /usr/src/app

# Install dependencies:
# RUN apt update && \
# apt install -y build-essential cmake

COPY requirements.txt ./

RUN pip install --upgrade pip && \
pip install -r requirements.txt

COPY . .

# Expose port:
EXPOSE 8501

# Run application:
ENTRYPOINT "streamlit run hair_cam.py"