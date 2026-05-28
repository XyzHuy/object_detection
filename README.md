# Object Detection Submission

## Train

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

The best checkpoint is saved to `./models/best.pth`.

## Predict

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

