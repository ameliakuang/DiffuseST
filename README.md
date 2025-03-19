<h1>DiffuseST: Unleashing the Capability of the Diffusion Model for Style Transfer</h1>

Ying Hu, [Chenyi Zhuang](https://chenyi-zhuang.github.io/), Pan Gao

[I2ML](https://i2-multimedia-lab.github.io/), Nanjing University of Aeronautics and Astronautics

[Paper](https://arxiv.org/abs/2410.15007)

### ⚙️ Setup and Usage
```bash
conda create --name DiffuseST python=3.8
conda activate DiffuseST

# Install requirements
pip install -r requirements.txt
```

Download the pre-trained [blipdiffusion](https://huggingface.co/salesforce/blipdiffusion). 

Put the content images in `images/content` and the style images in `images/style`.

```bash
# Run DiffuseST, alpha default to 0.1
python run.py

# Perform style injection for more steps
python run.py --alpha 0.2
```
