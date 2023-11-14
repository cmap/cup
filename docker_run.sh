#!/usr/bin/env bash

docker run -p 8801:8801 \
-e AWS_ACCESS_KEY_ID="AKIATAWTSI6KLR3VO5NQ" \
-e AWS_SECRET_ACCESS_KEY="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" \
-e API_KEY="1ce653c27e4ae77b060c0364be79db0b" \
-it prismcmap/cup-dev