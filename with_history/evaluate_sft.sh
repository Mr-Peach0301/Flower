declare -A start_years=(
    ["CDs_and_Vinyl"]=2015
    ["Video_Games"]=2015
    ["Movies_and_TV"]=2017
)
for category in "CDs_and_Vinyl" "Video_Games" "Movies_and_TV" 
do
    start_year=${start_years[$category]}
    group_num=8
    train_file=$(ls -f ./train/${category}_5_${start_year}*11.csv)
    test_file=$(ls -f ./test/${category}_5_${start_year}*11.csv)
    info_file=$(ls -f ./info/${category}_5_${start_year}*11.txt)
    group_title=$(ls -f ./info/${category}_group_title_${group_num}_${start_year}*11.json)
    logits_file=$(ls -f ./result/${category}_${start_year}_SASRec/test*.npy)
    cudalist="6"
    python ./code/split.py --input_path ${test_file} --output_path ./temp_sft/${category}_base
    for i in ${cudalist}
    do
        echo $i
        CUDA_VISIBLE_DEVICES=$i python ./code/evaluate.py --base_model ../output_dir_sft/${category}_${start_year} --train_file ${train_file} --info_file ${info_file} --category ${category} --test_data_path ./temp_sft/${category}_base/${i}.csv --result_json_data ./temp_sft/${category}_base/${i}.json
    done
    wait
    python ./code/merge.py --input_path ./temp_sft/${category}_base --output_path ./temp_sft/${category}_base/final_result.json
    python ./code/calc.py --path ./temp_sft/${category}_base/final_result.json --item_path ${info_file} --result_data ./temp_sft/${category}_base/sft.txt --group_title_path ${group_title} --category ${category}
    
    
    python ./code/split.py --input_path ${test_file} --output_path ./temp_sft/${category}_base
    for i in ${cudalist}
    do
        echo $i
        CUDA_VISIBLE_DEVICES=$i python ./code/evaluate.py --base_model ../output_dir_sft/${category}_${start_year} --train_file ${train_file} --info_file ${info_file} --category ${category} --test_data_path ./temp_sft/${category}_base/${i}.csv --result_json_data ./temp_sft/${category}_base/${i}.json  --temperature 1.5 &
    done
    wait
    python ./code/merge.py --input_path ./temp_sft/${category}_base --output_path ./temp_sft/${category}_base/final_result_temp.json
    python ./code/calc.py --path ./temp_sft/${category}_base/final_result_temp.json --item_path ${info_file} --result_data ./temp_sft/${category}_base/temp.txt --group_title_path ${group_title} --category ${category}
    
    python ./code/split.py --input_path ${test_file} --output_path ./temp_sft/${category}_base
    for i in ${cudalist}
    do
        echo $i
        CUDA_VISIBLE_DEVICES=$i python ./code/evaluate.py --base_model ../output_dir_sft/${category}_${start_year} --train_file ${train_file} --info_file ${info_file} --category ${category} --test_data_path ./temp_sft/${category}_base/${i}.csv --result_json_data ./temp_sft/${category}_base/${i}.json  --length_penalty 0.0 --logits_file ${logits_file} &
    done
    wait
    python ./code/merge.py --input_path ./temp_sft/${category}_base --output_path ./temp_sft/${category}_base/final_result_D3.json
    python ./code/calc.py --path ./temp_sft/${category}_base/final_result_D3.json --item_path ${info_file} --result_data ./temp_sft/${category}_base/D3.txt --group_title_path ${group_title} --category ${category}
done
