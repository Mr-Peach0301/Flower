category=Movies
num=100 # 100 or 1500
train_set=./data/train_${num}.json
possible_output=../${category}/${category}_${num}_title.txt
title_distri=../${category}/select_${num}_${category}.json
save_path=../${category}/${num}/ppo
base_model=./model/${category}_${num}
CUDA_VISIBLE_DEVICES=4 nohup python ./ppo.py > log_${category}_ppo_${num}.out \
    --base_model $base_model \
    --category $category \
    --train_set $train_set \
    --possible_output_path $possible_output \
    --title_distri $title_distri \
    --save_path $save_path