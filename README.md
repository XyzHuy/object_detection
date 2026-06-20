## Tổng quan dự án

Đây là dự án bài cuối kì cho môn học [Xử lý ảnh 2526II_AIT3002#_1](https://portal.uet.vnu.edu.vn/courses/5889) tại UET. Mục tiêu của project là tự xây dựng một mô hình object detection from scratch cho bài toán phát hiện 5 lớp đối tượng: `person`, `car`, `dog`, `cat`, `chair`.

Mô hình được triển khai theo hướng custom YOLOv8, gồm backbone `ConvNeXtV2Tiny` dùng làm feature extractor, neck YOLOv8 và detection head tự xây dựng. Repo bao gồm pipeline huấn luyện, suy luận, đánh giá, tuning threshold và các module loss/metrics/dataloader phục vụ huấn luyện end-to-end.

Dữ liệu huấn luyện và validation là bộ train/val được cung cấp trong cuộc thi. Theo thống kê trong `dataset_stats`, tập train có 7.500 ảnh, 10.642 bounding boxes và bị mất cân bằng đáng kể giữa các lớp: lớp `person` chiếm khoảng 54,8% số object, trong khi lớp ít nhất là `cat` chỉ khoảng 7,8%; tỉ lệ chênh lệch lớn nhất giữa các lớp xấp xỉ 7 lần. Tập val có 1.500 ảnh, 2.021 bounding boxes và phân bố lớp tương tự.

Kết quả đạt được:
- Khoảng `0.9 mAP@0.5` trên validation set (`score.json`: `0.894114`).
- Top 2/62 trên Kaggle Community Prediction Competition: [From Scratch Object Detection Challenge](https://www.kaggle.com/competitions/from-scratch-object-detection-challenge/leaderboard).
- `mAP@0.5 = 0.86` trên private set của competition.

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
