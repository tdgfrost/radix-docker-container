#!/bin/bash

# First, make sure there is read/write access in our intended mount folder (experiments), and fail gracefully if not

HOST_DIR="$HOME/experiments"
CONTAINER_DIR="/experiments"

if ! docker run --rm --entrypoint bash -v "$HOST_DIR:$CONTAINER_DIR" radix -c "touch $CONTAINER_DIR/.rw_test && rm $CONTAINER_DIR/.rw_test" 2>/dev/null; then
    echo "ERROR: The Docker container lacks read/write access to $HOST_DIR."
    echo "To fix this, run the following command on your host machine:"
    echo "  chmod -R a+rwx $HOST_DIR"
    exit 1
fi

# Start the container in the background and run the notebook
docker run --rm -d --name radix-notebook --replace -v /home/rmhitfr/experiments:/experiments -p 8888:8888 radix -c "pixi run notebook"

# Wait a few seconds for the Jupyter server to initialise
while ! curl -s http://127.0.0.1:8888 > /dev/null; do
	sleep 1
done

# Open the browser on the host machine
xdg-open "http://127.0.0.1:8888"
