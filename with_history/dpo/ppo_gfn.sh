declare -A start_years=(
    ["CDs_and_Vinyl"]=2015
    ["Video_Games"]=2015
    ["Movies_and_TV"]=2017
)
declare -A train_theta=(
    ["CDs_and_Vinyl"]=0.995
    ["Movies_and_TV"]=0.995
    ["Video_Games"]=0.9995
)
for category in "CDs_and_Vinyl" "Video_Games" "Movies_and_TV"
do
    start_year=${start_years[$category]}
    group_num=8
    train_theta=${train_theta[$category]}
    train_file=./data/${category}/${category}_dpo_train_dataset.json
    test_file=$(ls -f ../test/${category}_5_${start_year}*11.csv)
    info_file=$(ls -f ../info/${category}_5_${start_year}*11.txt)
    group_title=$(ls -f ../info/${category}_group_title_${group_num}_${start_year}*11.json)
    possible_output=$(ls -f ../info/${category}_title_interaction_5_${start_year}*11.json)
    sasrec_logits_path=$(ls -f ../result/${category}_${start_year}_SASRec/train*.npy)
    echo ${category} ${train_theta}
    CUDA_VISIBLE_DEVICES=7 nohup python ./train_ppo_gfn.py > ./out/${category}_ppo_gfn.out \
        --base_model /output_dir_sft-gfn_logp_div_s/${category}_${start_year}/${train_theta}_0 \
        --train_file ${train_file} \
        --output_dir ./model/${category}_${start_year}/gfn_ppo \
        --category ${category}  \
        --test_data_path ${test_file} \
        --info_path ${info_file} \
        --group_title_path ${group_title} \
        --title_distri ${possible_output} \
        --sasrec_logits_path ${sasrec_logits_path} \
        --temperature 1
done