#!/bin/bash

# Build the dashboard image from the ./dashboard directory
docker buildx build --platform linux/amd64 -t radix-dashboard-amd64 ./dashboard

# Get the exact uncompressed image size
IMAGE_SIZE=$(docker image inspect radix-dashboard-amd64 --format='{{.Size}}')

# Save, show progress, and gzip in one continuous stream
docker save radix-dashboard-amd64 | pv -s $IMAGE_SIZE | gzip -9 > radix-dashboard-amd64.tar.gz
