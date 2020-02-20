#!/bin/bash

python3_cmd=python3.6

stage=2
use_gpu=cuda:2

model=bert  # (bert|xlnet)
model_path=/home/M10815022/Models/roberta-wwm-ext  #/home/M10815022/Models/roberta-wwm-ext  #/home/M10815022/Models/roberta-base-lm-finetune
save_path=./models/roberta-1.7.2-woASR

train_datasets="DRCD_train Lee_train Kaggle_train FGC_release_all_train"
dev_datasets="DRCD_dev Lee_dev Kaggle_dev FGC_release_all_dev"
test_datasets="DRCD_test Lee_test Kaggle_test FGC_release_all_test"

#train_datasets="DRCD_train Lee_train Kaggle_train ASR_train FGC_release_all_train DRCD_dev Lee_dev Kaggle_dev ASR_dev DRCD_test Lee_test Kaggle_test ASR_test"
#dev_datasets="FGC_release_all_dev"
#test_datasets="FGC_release_all_test"


if [ $stage -le 0 ]; then
  echo "==================================================="
  echo "     Convert traditional Chinese to simplified     "
  echo "==================================================="
  for dataset in $train_datasets $dev_datasets $test_datasets; do
    file=dataset/$dataset.json
    echo "Converting '$file'..."
    opencc -i $file -o $file -c t2s.json || exit 1
  done
  echo "Done."
fi


if [ $stage -le 1 ]; then
  echo "======================"
  echo "     Prepare data     "
  echo "======================"
  rm -rf data
  for split in train dev test; do
    for dir in passage passage_no_unk question question_no_unk answer span; do
      mkdir -p data/$split/$dir
    done
  done
  echo "Preparing dev set..."
  $python3_cmd scripts/prepare_${model}_data.py $model_path dev $dev_datasets || exit 1
  echo "Preparing test set..."
  $python3_cmd scripts/prepare_${model}_data.py $model_path test $test_datasets || exit 1
  echo "Preparing train set..."
  $python3_cmd scripts/prepare_${model}_data.py $model_path train $train_datasets || exit 1
fi

exit 0


if [ $stage -le 2 ]; then
  echo "================================="
  echo "     Train and test QA model     "
  echo "================================="
  if [ -d $save_path ]; then
    echo "'$save_path' already exists! Please remove it and try again."; exit 1
  fi
  mkdir -p $save_path
  $python3_cmd scripts/train_${model}.py $use_gpu $model_path $save_path
fi
