FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04
WORKDIR /app

# Install basic dependencies
RUN set -xe \
 && apt update -q \
 && apt install -fyq python3 python3-pip \
 && apt clean

# Install python packages
COPY requirements.txt .
RUN set -xe \
 && pip install -r requirements.txt \
 && pip cache purge \
 && rm -R ~/.cache/pip

# Copy all sources
COPY . .

# Init entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
