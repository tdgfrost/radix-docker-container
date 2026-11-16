#!/bin/bash

# First, make sure the image is available. Use a local hash to check whether the image has been updated since the last load.

IMAGE_NAME="radix-dashboard-amd64"
HASH_FILE=".${IMAGE_NAME}_hash"
ARCHIVE_FILE=""
PORT=8501

if [ -f "${IMAGE_NAME}.tar.gz" ]; then
    ARCHIVE_FILE="${IMAGE_NAME}.tar.gz"
elif [ -f "${IMAGE_NAME}.tar" ]; then
    ARCHIVE_FILE="${IMAGE_NAME}.tar"
fi

if [ -n "$ARCHIVE_FILE" ]; then
    echo "Archive found. Checking hash..."
    CURRENT_HASH=$(sha256sum "$ARCHIVE_FILE" | awk '{ print $1 }')

    if [ -f "$HASH_FILE" ]; then
        CACHED_HASH=$(cat "$HASH_FILE")
    else
        CACHED_HASH=""
    fi

    # Load the image if the hash differs OR if the image is not in the local registry
    if [ "$CURRENT_HASH" != "$CACHED_HASH" ] || ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        echo "Update detected or image missing. Loading '$ARCHIVE_FILE'..."

        # Remove the old image and prune dangling blobs BEFORE loading to maximise free disk space
        docker rm -f radix-dashboard >/dev/null 2>&1
        docker rmi -f "$IMAGE_NAME" >/dev/null 2>&1
        docker image prune -f >/dev/null 2>&1

        # Decompress on the fly if .tar.gz so Docker does not duplicate the compressed archive in /var/lib/docker/tmp
        if echo "$ARCHIVE_FILE" | grep -q '\.gz$'; then
            gzip -dc "$ARCHIVE_FILE" | docker load || exit 1
        else
            docker load -i "$ARCHIVE_FILE" || exit 1
        fi
        docker image prune -f >/dev/null 2>&1

        echo "$CURRENT_HASH" > "$HASH_FILE"
    else
        echo "Archive unchanged. Using existing local image."
    fi
else
    # No archive found, check if image exists locally
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        echo "Error: No archive found, and image '$IMAGE_NAME' is not loaded locally."
        exit 1
    fi
    echo "No archive found. Using existing local image '$IMAGE_NAME'."
fi

# Second, make sure there is read/write access in our intended mount folder (experiments), and fail gracefully if not

HOST_DIR="$HOME/experiments"
CONTAINER_DIR="/experiments"

if ! docker run --rm --user "$(id -u):$(id -g)" --entrypoint bash -v "$HOST_DIR:$CONTAINER_DIR" "$IMAGE_NAME" -c "touch $CONTAINER_DIR/.rw_test && rm $CONTAINER_DIR/.rw_test" 2>/dev/null; then
    echo "ERROR: The Docker container lacks read/write access to $HOST_DIR."
    echo "To fix this, run the following command on your host machine:"
    echo "  chmod -R u+rwX $HOST_DIR"
    exit 1
fi

# Start the container in the background and run the interactive model dashboard
# --rm --replace will replace any old containers, and delete the container after it finishes
# -d detaches the container
# --name sets a fixed/consistent container name
# --user will set the user/group id to match the host user, so we preserve read/write access
# -v virtually mounts the experiments directory within the container
# -p does the port re-direct for the dashboard web server

docker run --rm --replace -d --name radix-dashboard --user "$(id -u):$(id -g)" -v "$HOST_DIR:/experiments" -p "${PORT}:${PORT}" "$IMAGE_NAME" -c "pixi run dashboard"

# Wait a few seconds for the dashboard server to initialise
while ! curl -s "http://127.0.0.1:${PORT}" > /dev/null; do
    sleep 1
done

# Open the browser on the host machine
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://127.0.0.1:${PORT}"
elif command -v open >/dev/null 2>&1; then
    open "http://127.0.0.1:${PORT}"
else
    echo "Dashboard is running at http://127.0.0.1:${PORT}"
fi
