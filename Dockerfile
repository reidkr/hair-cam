# API_VERSION 1.0
FROM python:3.7-slim-buster
MAINTAINER reidkr876@gmail.com

WORKDIR /usr/src/app

# Install dependencies:
RUN apt update && \
apt install -y build-essential cmake

# COPY requirements.txt ./
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu \
-f https://download.pytorch.org/whl/torch_stable.html
RUN pip install streamlit python-dotenv flask fastai

COPY . .

# Expose port:
# EXPOSE 8501

# This allows Heroku bind its PORT the Apps port
# since Heroku needs to use its own PORT before the App can be made accessible to the World
EXPOSE $PORT

# Run application:
ENTRYPOINT "streamlit run hair_cam.py --server.port $PORT"