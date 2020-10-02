#!/bin/bash

CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda
