docker run -p 8801:8801 \
-v ~/.aws/credentials:/root/.aws/credentials \
-e AWS_DEFAULT_REGION=us-east-1 \
-it prismcmap/cup