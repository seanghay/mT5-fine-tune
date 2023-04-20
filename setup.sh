#!/usr/bin/env bash

echo "1. Check for GPU"
nvidia-smi

echo "2. Install Git LFS"
sudo apt-get update -y
sudo apt-get install git-lfs -y

echo "3. Create venv"
python3 -m venv hf_env
source hf_env/bin/activate
echo "source $(pwd)/hf_env/bin/activate" >> ~/.bashrc

echo "4. Install deps"
pip install -r requirements.txt

echo "5. Authenticate"
git config --global credential.helper store
huggingface-cli login

