#!/bin/bash

# Default - simple goto
# CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda --exp_dir exp/demo --n_per_file 10000
CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda --exp_dir exp/goto

# Default - simple pickup
CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda --exp_dir exp/pickup --demos demos/BabyAI-Pickup-v0.pkl

CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda --exp_dir exp/synth --demos demos/BabyAI-Synth-v0.pkl --n_per_file 1000000

CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda --exp_dir exp/boss_big --demos demos/BabyAI-BossLevel-v0.pkl --n_per_file 1000000
