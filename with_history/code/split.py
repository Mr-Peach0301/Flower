import fire
import os
import pandas as pd

def split(input_path, output_path):
    df = pd.read_csv(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_len = len(df)
    cuda = [6]
    for i in range(len(cuda)):
        start = i * df_len // len(cuda)
        end = (i+1) * df_len // len(cuda)
        df[start:end].to_csv(f'{output_path}/{cuda[i]}.csv', index=True)

if __name__ == '__main__':
    fire.Fire(split)
