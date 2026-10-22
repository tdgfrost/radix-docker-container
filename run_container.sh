#!/bin/bash

# First, make sure the image is available. Use a local hash to check whether the image has been updated since the last load.

IMAGE_NAME="radix-container-amd64"
HASH_FILE=".${IMAGE_NAME}_hash"
ARCHIVE_FILE=""

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
        docker load -i "$ARCHIVE_FILE" || exit 1
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

# Second, make sure there is read/write access in our intended mount folder (experiments) and fail gracefully if not
HOST_DIR="$HOME/experiments"
CONTAINER_DIR="/experiments"

if ! docker run --rm --entrypoint bash -v "$HOST_DIR:$CONTAINER_DIR" "$IMAGE_NAME" -c "touch $CONTAINER_DIR/.rw_test && rm $CONTAINER_DIR/.rw_test" 2>/dev/null; then
    echo "ERROR: The Docker container lacks read/write access to $HOST_DIR."
    echo "To fix this, run the following command on your host machine:"
    echo "  chmod -R a+rwX $HOST_DIR"
    exit 1
fi

# If you need superuser access, include the `-u root` flag
docker run --name radix-container --rm --replace -v "$HOST_DIR:/experiments" -it -p 8888:8888 "$IMAGE_NAME"
