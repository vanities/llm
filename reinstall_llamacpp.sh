#!/bin/bash

CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip3 install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
