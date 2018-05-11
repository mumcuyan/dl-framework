#!/usr/bin/env bash

docker build -t dl-framework:latest .

docker run --rm -d -v "$(pwd):/app" -p 8888:8888 dl-framework