# apocalypse-mommy-llm-supervised-trainer

switch to NVIDIA’s proprietary driver so PyTorch can use the GPU.
```bash
sudo apt update
ubuntu-drivers devices
# Pick the recommended, e.g. nvidia-driver-550 (or 555/560+ if shown as "recommended")
sudo apt install -y nvidia-driver-550
sudo reboot

```
