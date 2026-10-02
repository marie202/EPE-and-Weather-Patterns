# Linux image for HPC (e.g. Levante); build on Mac with:
#   docker buildx build --platform linux/amd64 -t marie202/paper3-detect-weather-patterns:latest --push .
FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml && conda clean -afy

ENV PATH=/opt/conda/envs/weather-regimes/bin:$PATH

SHELL ["conda", "run", "-n", "weather-regimes", "/bin/bash", "-c"]

RUN python --version && python -c "import numpy, xarray; print('ok')"

CMD ["bash"]
