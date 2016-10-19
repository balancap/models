
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
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.045 \
    --batch_size=32
```

DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/Training/logs
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v3.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=lead_inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=600 \
    --save_interval_secs=600 \
    --weight_decay=0.00004 \
    --optimizer=rmsprop \
    --learning_rate=0.0001 \
    --batch_size=54


CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v3.ckpt
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/logs
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=lead_inception_v3



DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_resnet_v2_2016_08_30.ckpt
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_resnet_v2

