declare -A start_years=(
    ["CDs_and_Vinyl"]=2015
    ["Video_Games"]=2015
    ["Movies_and_TV"]=2017
)
for category in "CDs_and_Vinyl" "Video_Games" "Movies_and_TV"
do
    start_year=${start_years[$category]}
    end_year=2018
    group_num=8
    model=SASRec
    python ./code/train_and_get_logits.py \
        --data ./cache/${category}_${start_year} \
        --cuda 0\
        --model ${model}\
        --category ${category}\
        --result_json_path ./result/${category}_${start_year}_${model}/temp.json\
        --group_ids ./info/${category}_group_title_${group_num}_${start_year}-10-${end_year}-11.json\
        --info_file ./info/${category}_5_${start_year}-10-${end_year}-11.txt
done