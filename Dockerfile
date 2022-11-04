# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common

COPY . /app
RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/bash"]

CMD ["./run_app.sh"]
