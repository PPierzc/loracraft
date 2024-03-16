# loracraft
Brainhack 2024 


# Env setup

```bash

conda create --prefix ./<your_env> python=3.10
conda activate ./<your_env>

conda install pytorch=2.2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


conda install -c conda-forge diffusers
conda install xformers::xformers
conda install conda-forge::transformers

conda install conda-forge::accelerate

pip install tomesd==0.1.3
pip install peft==0.9.0

conda install conda-forge::wandb
```


