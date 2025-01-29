# Flow-guided Fine-Tuning for Diverse LLM-based Recommenders

This repository hosts the official PyTorch-based implementation of the method presented in the paper "Flow-guided Fine-Tuning for Diverse LLM-based Recommenders".

## Installation

To install the project, follow these steps:

1. Clone this git repository and change directory to this repository.

2. Create a conda environment and activate.
```conda create --name Flower python=3.9 -y```
```conda activate Flower```

3. Install the required modules from pip.
```pip install -r requirements.txt```

## Data processing
The following steps use the Video Games dataset for example. The data required for training has already been provided in this repository.

1. Download the dataset
```wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Video_Games.json.gz```
```wget wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz```

2. Unzip
```gunzip Video_Games.json.gz```
```gunzip meta_Video_Games.json.gz```

3. Preprocess for BIGRec and IFairLRS
```python ./code/process.py --category "Video_Games"```

4. Preprocess for SASRec
```bash to_SASRec.sh```

5. Preprocess for Flower
run the code in process.ipynb

## Next-item Recommendation Results (RQ2)

To reproduce the results in RQ2, follow these steps:

1. Train SASRec
```bash run_SASRec.sh```

2. Train and evaluate BIGRec
```bash run_sft.sh```
```bash evaluate_sft.sh```

3. Train Flower
```bash run_sft-gfn_logp_div_s.sh```

4. Train IFairLRS
```bash item_side_reweight.sh```

## Analysis of Key Factors in Flower (RQ4)

To reproduce the results in RQ4, follow these steps:

1. Effects of Reward Setting
```bash run_sft-gfn_logp_div_s.sh```
```bash run_sft-gfn_logp_add_logs.sh```
```bash run_sft-gfn_logp.sh```

2. Impact of Supervision Granularity
```bash run_sft-gfn_logp_n.sh```

3. Performance Varying 𝜆
```bash run_sft-gfn_logp_div_s_lambda.sh```
