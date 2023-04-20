#!/usr/bin/env bash

nvidia-smi

sudo apt-get update -y
sudo apt-get install git-lfs -y

python3 -m venv hf_env
source hf_env/bin/activate
echo "source ~/hf_env/bin/activate" >> ~/.bashrc

pip install -r requirements.txt

git config --global credential.helper store
huggingface-cli login

