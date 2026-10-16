#!/bin/bash

# Build the image
docker buildx build --platform linux/amd64 -t radix-container-amd64 .

# Get the exact uncompressed image size
IMAGE_SIZE=$(docker image inspect radix-container-amd64 --format='{{.Size}}')

# Save, show progress, and gzip in one continuous stream
docker save radix-container-amd64 | pv -s $IMAGE_SIZE | gzip -9 > radix-container-amd64.tar.gz
