## Cài đặt môi trường
- Với môi trường chỉ có CPU:
```bash
pip install -r requirements.txt     
```
- Với môi trường có GPU
Lệnh dưới cài đặt với phiên bản CUDA 12.1 cho torch, có thể không tương thích với các phần cứng máy khác

```bash
pip install -r requirements.txt
pip uninstall torch torchvision  -y
pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu121
```

## Lệnh huấn luyện 
```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/  
```

## Lệnh suy luận bắt buộc

### Tập val

- Predict: 
```bash
python predict.py \
  --image_dir .public/val/images \
  --output predictions.json \
  --checkpoint ./models/best.pth \
```

- Evaluate:
```bash
python ./public/tools/evaluate_predictions.py 
  --ground_truth ./public/annotations/val.json 
  --predictions predictions.json 
  --output score.json
```



### Tập Test

Với giả định file annotation và tập ảnh của tập test có cùng cấu trúc với train và val

- Predict: 
```bash
python predict.py \
  --image_dir ./public/test/images \
  --output predictions.json \
  --checkpoint ./models/best.pth \
```

- Evaluate:
```bash
python ./public/tools/evaluate_predictions.py 
  --ground_truth ./public/annotations/test.json 
  --predictions predictions.json 
  --output score.json
```


## P/s:
predict.py có các flag để điều chỉnh threshold để tăng precsion nhưng đánh đổi bằng recall

--conf_threshold :
--nms_iou :
--max_det :

Nếu train bị crash, lỗi có thể do pin_memory trong dataloader