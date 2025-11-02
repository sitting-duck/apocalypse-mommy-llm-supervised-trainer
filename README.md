# apocalypse-mommy-llm-supervised-trainer

switch to NVIDIA’s proprietary driver so PyTorch can use the GPU.
```bash
sudo apt update

# see what drivers are available for cpu and gpu
sudo ubuntu-drivers list 
sudo ubuntu-drivers list --gpgpu


# install latest driver for your setep. Author used 580 at time of writing but yours may be different. Check compatibility based on your card.
sudo apt install -y nvidia-driver-580
# cleanup leftover dependencies
sudo apt autoremove -y


sudo reboot

# python setup
python --version

# if you don't have Python installed yet - 3.11 is a perfect version for our stack
sudo apt update
sudo apt install -y python3.11
sudo apt install -y python3.11-venv
ssudo apt install -y python3-pip

sudo apt install -y python-is-python3

python3.11 -m venv .venv
source .venv/bin/activate
python -V # should show 3.11

pip3 install -U pip

pip install --index-url https://download.pytorch.org/whl/cu124 torch
pip install --index-url https://download.pytorch.org/whl/cu124 torchvision
pip install --index-url https://download.pytorch.org/whl/cu124 torchaudio
```
Verify like this: 

```python
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("bundled CUDA:", torch.version.cuda)           # e.g., '12.4' or '12.1'
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

```
and I get some output like this: 

```bash
torch: 2.6.0+cu124
bundled CUDA: 12.4
cuDNN: 90100
CUDA available: True
GPU: NVIDIA GeForce RTX 3080

```
install python deps
```bash
pip install -r requirements.txt
```








