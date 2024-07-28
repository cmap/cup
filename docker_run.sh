#!/usr/bin/env bash

docker run -p 8801:8801 \
-e API_KEY=XXXXXXX \
-e API_URL="https://api.clue.io/api/" \
-e AWS_ACCESS_KEY_ID=XXXXXXX \
-e AWS_SECRET_ACCESS_KEY=XXXXXXXX \
-e AWS_DEFAULT_REGION=us-east-1 \
-it prismcmap/cup-dev
