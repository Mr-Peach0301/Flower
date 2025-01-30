declare -A start_years=(
    ["CDs_and_Vinyl"]=2015
    ["Video_Games"]=2015
    ["Movies_and_TV"]=2017
)
for category in "CDs_and_Vinyl" "Video_Games" "Movies_and_TV"
do
    start_year=${start_years[$category]}
    group_num=8
    train_file=./data/${category}/${category}_rosedpo_train_dataset.json
    eval_file=./data/${category}/${category}_rosedpo_eval_dataset.json
    test_file=$(ls -f ../test/${category}_5_${start_year}*11.csv)
    info_file=$(ls -f ../info/${category}_5_${start_year}*11.txt)
    group_title=$(ls -f ../info/${category}_group_title_${group_num}_${start_year}*11.json)
    title_token_distribution=$(ls -f ../info/${category}_token_distribution_5_${start_year}*11.json)
    possible_output=$(ls -f ../info/${category}_title_interaction_5_${start_year}*11.json)
    echo ${category}
    CUDA_VISIBLE_DEVICES=0 nohup python ./train_rosedpo.py > ./out/${category}_rosedpo.out \
        --base_model /output_dir_sft/${category}_${start_year} \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --output_dir ./model/${category}_${start_year}/sft_rosedpo \
        --category ${category}  \
        --test_data_path ${test_file} \
        --info_path ${info_file} \
        --group_title_path ${group_title} \
        --title_token_distribution ${title_token_distribution} \
        --possible_output ${possible_output} \
        --temperature 1
done