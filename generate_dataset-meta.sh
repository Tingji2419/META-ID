for dataset in Beauty
do
    for indexing in metapath
    do
        python ./src/generate_dataset.py --dataset ${dataset} --metapath_cluster_method kmcos --metapath_cluster_num 100 --data_path ./data/ --item_indexing ${indexing} --user_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt

        python ./src/generate_dataset_eval.py --dataset ${dataset} --metapath_cluster_method kmcos --metapath_cluster_num 100 --data_path ./data/ --item_indexing ${indexing} --user_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode validation --prompt seen:0

        python ./src/generate_dataset_eval.py --dataset ${dataset} --metapath_cluster_method kmcos --metapath_cluster_num 100 --data_path ./data/ --item_indexing ${indexing} --user_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode test --prompt seen:0

        python ./src/generate_dataset_eval.py --dataset ${dataset} --metapath_cluster_method kmcos --metapath_cluster_num 100 --data_path ./data/ --item_indexing ${indexing} --user_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode test --prompt unseen:0
    done
done

# for dataset in Beauty
# do
#     for indexing in metapath
#     do
#         python ./src/generate_dataset.py --dataset ${dataset} --metapath_cluster_method dbscan --metapath_cluster_num 37 --data_path ./data/ --item_indexing ${indexing} --user_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt

#         python ./src/generate_dataset_eval.py --dataset ${dataset} --metapath_cluster_method dbscan --metapath_cluster_num 37 --data_path ./data/ --item_indexing ${indexing} --user_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode validation --prompt seen:0

#         python ./src/generate_dataset_eval.py --dataset ${dataset} --metapath_cluster_method dbscan --metapath_cluster_num 37 --data_path ./data/ --item_indexing ${indexing} --user_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode test --prompt seen:0

#         python ./src/generate_dataset_eval.py --dataset ${dataset} --metapath_cluster_method dbscan --metapath_cluster_num 37 --data_path ./data/ --item_indexing ${indexing} --user_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode test --prompt unseen:0
#     done
# done
