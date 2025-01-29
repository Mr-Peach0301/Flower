declare -A start_years=(
    ["CDs_and_Vinyl"]=2015
    ["Video_Games"]=2015
    ["Movies_and_TV"]=2017
)
for category in "CDs_and_Vinyl" "Video_Games" "Movies_and_TV"
do
    start_year=${start_years[$category]}
    train_file=$(ls -f ./train/${category}_5_${start_year}*11.csv)
    eval_file=$(ls -f ./valid/${category}_5_${start_year}*11.csv)
    test_file=$(ls -f ./test/${category}_5_${start_year}*11.csv)
    python ./code/preprocess.py \
        --start_year ${start_year} \
        --category ${category} \
        --train_file ${train_file} \
        --eval_file ${eval_file}\
        --test_file ${test_file}
done