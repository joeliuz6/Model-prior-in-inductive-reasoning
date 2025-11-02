# ✨ [EMNLP2025] On the Role of Model Prior in Real-World Inductive Reasoning
[**🔬 Paper**](https://arxiv.org/abs/2412.13645)

## 🧩 Overview

This project explores how model priors affect real-world inductive reasoning tasks.  
We provide experiments demonstrating how different model configurations and data setups impact reasoning performance under various conditions.

## 🚀 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/joeliuz6/Model-prior-in-inductive-reasoning.git
cd Model-prior-in-inductive-reasoning
```

### 2. Create and Activate Environment
You can use **conda**.

**Using Conda:**
```bash
conda create -n modelprior python=3.10
conda activate modelprior
```

### 3. Set Your OpenAI API Key
**Linux / macOS:**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

**Windows (PowerShell):**
```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

### 4. Run `main.py`
**Basic Run:**
```bash
python main.py
```

**With demonstrations and label configurations:**
```bash
python main.py --with_data data --with_label label
```

### 5. Arguments Explanation

| Argument | Type | Description |
|-----------|------|-------------|
| `--with_data` | flag | Include in-context demonstrations in reasoning. |
| `--with_label` | flag | Specify label configuration for reasoning tasks. |
| `--seed` | int | *(Optional)* Random seed for reproducibility. Default: 2 |
| `--model` | str | *(Optional)* Model name, e.g. `gpt-4-turbo`, `gpt-3.5-turbo`. |

**Example:**
```bash
python main.py --with_data --with_label --model gpt-4-turbo
```



## Dataset

Full datasets in the paper.
[Google Drive](https://drive.google.com/file/d/1zojab3wzJ942X6qMAmVRQcE_m-kMO9Wa/view?usp=drive_link)


## ✏️ Citation
```bibtex
@article{liu2024role,
  title={On the Role of Model Prior in Real-World Inductive Reasoning},
  author={Liu, Zhuo and Yu, Ding and He, Hangfeng},
  journal={arXiv preprint arXiv:2412.13645},
  year={2024}
}
```
