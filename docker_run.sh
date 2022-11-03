#!/usr/bin/env bash
#!/usr/bin/env bash
docker run -p 8501:8501 \
-e AWS_ACCESS_KEY_ID=XXXXXXX \
-e AWS_SECRET_ACCESS_KEY=XXXXXX \
-e API_URL="https://api.clue.io/api/" \
-e API_KEY=XXXXXXXXX \
-it prism/ctg