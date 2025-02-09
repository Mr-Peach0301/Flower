category=Movies
num=100 # 100 or 1500
output_dir=./dpo_model/
base_model=./model/${category}_${num}
train_file=./data/dpo/${category}_${num}_dpo_train_dataset.json
test_file=./data/test_${num}.json
possible_output_path=../${category}/${category}_${num}_title.txt
real_token_distri=../${category}/select_${num}_${category}.json
save_path=../${category}/${num}/dpo
CUDA_VISIBLE_DEVICES=4 nohup python ./dpo.py > log_${category}_dpo_${num}.out \
    --base_model $base_model \
    --train_file $train_file \
    --output_dir ${output_dir}${category}_${num} \
    --category $category  \
    --test_data_path $test_file \
    --real_token_distri $real_token_distri \
    --save_path $save_path \
    --possible_output_path $possible_output_path