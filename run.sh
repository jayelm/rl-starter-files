#!/bin/bash

# Default - simple goto
CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda --exp_dir exp/demo --n_per_file 10000

# Default - simple pickup
# CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda --exp_dir exp/pickup --demos demos/BabyAI-Pickup-v0.pkl

# CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda --exp_dir exp/goto-pickup --demos demos/BabyAI-Pickup-v0.pkl demos/BabyAI-GoTo-v0.pkl

# CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda --exp_dir exp/goto-pickup-putnext --demos demos/BabyAI-Pickup-v0.pkl demos/BabyAI-GoTo-v0.pkl demos/BabyAI-PutNext-v0.pkl

# CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda --exp_dir exp/boss --demos demos/BabyAI-BossLevel-v0.pkl --n_per_file 500000

# CUDA_VISIBLE_DEVICES="$1" python -m scripts.caption --cuda --exp_dir exp/boss_big --demos demos/BabyAI-BossLevel-v0.pkl --n_per_file 1000000
