output_dir=./model/
base_model=YOUR_MODEL_PATH_1.5B
category=Movies
num=100
train_data=./data/train_${num}.json
val_data=./data/test_${num}.json
possible_output_path=../${category}/${category}_${num}_title.txt
real_token_distri=../${category}/select_${num}_${category}.json
save_path=../${category}/${num}/bigrec
CUDA_VISIBLE_DEVICES=4 nohup python ./train_gfn.py > log_${category}_bigrec_${num}.out \
    --base_model $base_model \
    --train_data_path [\'$train_data\']   \
    --val_data_path [\'$val_data\'] \
    --output_dir ${output_dir}${category}_${num} \
    --batch_size 32 \
    --micro_batch_size 4 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --train_on_inputs \
    --group_by_length \
    --resume_from_checkpoint \
    --seed 33 \
    --sample -1 \
    --possible_output_path $possible_output_path \
    --real_token_distri $real_token_distri \
    --save_path $save_path \
    --category $category