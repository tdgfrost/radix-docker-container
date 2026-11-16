# Code for building and running the RADIX ML Docker containers

This repository contains two separate Docker container configurations for use within the TRE:

1. **Training / Notebook Container (`radix-container-amd64`)** — defined at the repository root (`Dockerfile`, `pixi.toml`).
   - To build and save the training Docker image, run `sh build_docker.sh`.
   - To run the Jupyter notebook in RADIX (port `8888`), run `sh run_notebook.sh` (or `sh run_container.sh` to open an interactive shell inside the container).

2. **Interactive Model Dashboard Container (`radix-dashboard-amd64`)** — defined in `./dashboard` (`dashboard/Dockerfile`, `dashboard/pixi.toml`, `dashboard/app.py`).
   - To build and save the dashboard Docker image, run `sh build_dashboard.sh`.
   - To launch the interactive model dashboard in RADIX (port `8501`) and open it in the browser, run `sh run_dashboard.sh`.
