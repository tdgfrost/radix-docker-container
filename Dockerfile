# 1. Base operating system
FROM debian:bookworm-slim

# 2. Prevent interactive package configuration prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install basic tools needed to fetch code and build Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the Pixi executable directly from the official Prefix container
COPY --from=ghcr.io/prefix-dev/pixi:latest /usr/local/bin/pixi /usr/local/bin/pixi

# 5. Create a standard user named "devuser"
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# 6. Set the working directory
WORKDIR /home/$USERNAME/app
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME \
    && chmod -R a+rwX /home/$USERNAME

# 7. Switch from root to devuser
USER $USERNAME

# 8. Copy your project configuration and install dependencies
COPY --chown=$USERNAME:$USERNAME pixi.toml pixi.lock ./
RUN pixi install --locked

# 9. Copy remaining project files (scripts, source files)
COPY --chown=$USERNAME:$USERNAME . .

# 10. Enforce offline mode permanently
ENV PIXI_FROZEN=true
ENV PIXI_OFFLINE=true

# 11. Default command when container runs
ENTRYPOINT ["pixi", "run", "nu"]
