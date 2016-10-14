
# Training a model from scratch.

```shell
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/Training/logs
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3 \
    --save_summaries_secs=600 \
    --save_interval_secs=600 \
    --weight_decay=0.00004 \
    --optimizer=rmsprop \
    --learning_rate=0.01 \
    --batch_size=32
```
