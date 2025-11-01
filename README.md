# apocalypse-mommy-llm-supervised-trainer

switch to NVIDIA’s proprietary driver so PyTorch can use the GPU.
```bash
sudo apt update

# see what drivers are available for cpu and gpu
sudo ubuntu-drivers list 
sudo ubuntu-drivers list --gpgpu


ubuntu-drivers devices
# Pick the recommended, e.g. nvidia-driver-550 (or 555/560+ if shown as "recommended")

sudo apt install -y nvidia-driver-580
sudo reboot

```
