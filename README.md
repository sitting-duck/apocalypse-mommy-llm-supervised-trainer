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

```
