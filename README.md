# apocalypse-mommy-llm-supervised-trainer

linux only

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
pip install rank_bm25

pip install -r requirements.txt
pip install -U "sentence-transformers>=3.0.0"
pip install datasets
pip install transformers[torch]
pip install scikit-learn
pip install rank-bm25
```


### beef up the training dataset with the Survival Corpus
```bash
# clone upstream outside or in ./external (but ignored)
git clone https://github.com/PR0M3TH3AN/Survival-Data.git external/Survival-Data

pip install beautifulsoup4 pdfminer.six
sudo apt install ocrmypdf

# OCR pass (only when needed): 
# ocrmypdf --skip-text in.pdf out.pdf

# (build corpus + queries with our scripts)
python scripts/build_corpus_from_survival_repo.py --src external/Survival-Data --out data/corpus.jsonl
python scripts/inspect_corpus.py --file data/corpus.jsonl --sample 8
python scripts/clean_rechunk_corpus.py --infile data/corpus.jsonl --outfile data/corpus_clean.jsonl --max_tokens 180
python scripts/make_rerank_data_from_corpus.py --corpus data/corpus_clean.jsonl --queries data/queries.jsonl --top_k 20


python scripts/make_queries_from_survival_repo.py  --src external/Survival-Data --out data/queries.jsonl

# record the exact upstream commit you used
git -C external/Survival-Data rev-parse HEAD > data/source_commit.txt

```





