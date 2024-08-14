# app/Dockerfile

FROM python:3.9-slim-buster

EXPOSE 8801

WORKDIR /app

RUN apt-get update && apt-get install --fix-missing -y \
    build-essential \
    software-properties-common

# Copy only the requirements file first
COPY requirements.txt /app/

# Install dependencies (this layer will be cached unless requirements.txt changes)
RUN pip3 install -r requirements.txt

# Now copy the rest of the application files
COPY . /app

ENTRYPOINT ["/bin/bash"]

CMD ["./run_app.sh"]
