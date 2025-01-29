declare -A start_years=(
    ["CDs_and_Vinyl"]=2015
    ["Video_Games"]=2015
    ["Movies_and_TV"]=2017
)
for category in "CDs_and_Vinyl" "Video_Games" "Movies_and_TV"
do
    for train_theta in 0.99 0.995 0.999 0.9995 0.9999
    do
        n=1
        start_year=${start_years[$category]}
        group_num=8
        gfn_theta=0
        train_file=$(ls -f ./train/${category}_5_${start_year}*11.csv)
        eval_file=$(ls -f ./valid/${category}_5_${start_year}*11.csv)
        test_file=$(ls -f ./test/${category}_5_${start_year}*11.csv)
        info_file=$(ls -f ./info/${category}_5_${start_year}*11.txt)
        group_title=$(ls -f ./info/${category}_group_title_${group_num}_${start_year}*11.json)
        title_token_distribution=$(ls -f ./info/${category}_token_distribution_5_${start_year}*11.json)
        n_token_distribution=$(ls -f ./info/${category}_token_distribution_${n}_5_${start_year}*11.json)
        possible_output=$(ls -f ./info/${category}_title_interaction_5_${start_year}*11.json)
        echo ${category} ${train_theta} ${gfn_theta} ${n} ${title_token_distribution} ${n_token_distribution}
        CUDA_VISIBLE_DEVICES=7 nohup python ./code/train_sft-gfn_logp.py  > ./out_sft-gfn/${category}_${train_theta}_${gfn_theta}_logp_${n}.out \
            --base_model YOUR_MODEL_PATH \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ../output_dir_sft-gfn_logp_${n}/${category}_${start_year} \
            --category ${category}  \
            --min_sentence_len 1 \
            --max_sentence_len 100 \
            --test_data_path ${test_file} \
            --info_path ${info_file} \
            --group_title_path ${group_title} \
            --title_token_distribution ${title_token_distribution} \
            --n_token_distribution ${n_token_distribution} \
            --possible_output ${possible_output} \
            --gfn_theta ${gfn_theta} \
            --train_theta ${train_theta} \
            --n ${n} \
            --temperature 1
    done
done