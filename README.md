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
The following steps use the Video Games dataset for example. Due to file size limitations of GitHub, the files of training set are not uploaded to the repository, other files are available.

1. Download the dataset

```wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Video_Games.json.gz```

```wget wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz```

2. Unzip

```gunzip Video_Games.json.gz```

```gunzip meta_Video_Games.json.gz```

```cd /with_history```

3.Process for BIGRec and IFairLRS

```python ./code/process.py --category "Video_Games"```

4. Process for SASRec

```bash to_SASRec.sh```

5. Process for Flower

run the code in process.ipynb

## Next-item Recommendation Results (RQ2)

To reproduce the results in RQ2, follow these steps:

```cd /with_history```

1. Train SASRec

```bash run_SASRec.sh```

2. Train and evaluate BIGRec

```bash run_sft.sh```

```bash evaluate_sft.sh```

3. Train Flower

```bash run_sft-gfn_logp_div_s.sh```

4. Train IFairLRS

```bash item_side_reweight.sh```

## Flower as a Reference Policy (RQ3)

To reproduce the results in RQ3, follow these steps:

```cd /with_history/dpo```

1. Data processing

run the code in dpo_dataset.ipynb

Due to file size limitations of GitHub, the files of training set are not uploaded to the repository, other files are available.

2. BIGRec as a Reference Policy

```bash dmpo.sh```

```bash sdpo.sh```

```bash ppo.sh```

```bash rosedpo.sh```

3. Flower as a Reference Policy

```bash dmpo_gfn.sh```

```bash sdpo_gfn.sh```

```bash ppo_gfn.sh```

```bash rosedpo_gfn.sh```

## Analysis of Key Factors in Flower (RQ4)

To reproduce the results in RQ4, follow these steps:

```cd /with_history```

1. Effects of Reward Setting

```bash run_sft-gfn_logp_div_s.sh```

```bash run_sft-gfn_logp_add_logs.sh```

```bash run_sft-gfn_logp.sh```

2. Impact of Supervision Granularity

```bash run_sft-gfn_logp_n.sh```

3. Performance Varying 𝜆

```bash run_sft-gfn_logp_div_s_lambda.sh```
