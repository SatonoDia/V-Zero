Base_model=$1
prefix=$2
data_source=$3

echo "Using $data_source for self-training"

python utils/questioner_data.py --local_dataset_path /root/autodl-tmp/data/OpenVLThinker
bash scripts/questioner_train.sh $Base_model $Base_model ${prefix}_questioner_v1
bash scripts/solver_train.sh $Base_model /root/autodl-tmp/models/${prefix}_questioner_v1 ${data_source} ${prefix}_solver_v1

for i in {2..2}; do
    prev=$((i-1))
    
    bash scripts/questioner_train.sh \
        /root/autodl-tmp/models/${prefix}_solver_v${prev} \
        /root/autodl-tmp/models/${prefix}_questioner_v${prev} \
        ${prefix}_questioner_v${i}

    bash scripts/solver_train.sh \
        /root/autodl-tmp/models/${prefix}_solver_v${prev} \
        /root/autodl-tmp/models/${prefix}_questioner_v${i} \
        ${data_source} \
        ${prefix}_solver_v${i}
done

