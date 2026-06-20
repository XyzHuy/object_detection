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



## Tune threshold tối ưu F1

Script `utils/tune_thresholds.py` tự dùng GPU nếu `torch.cuda.is_available()` trả về `True`. Khi mục tiêu là tối đa F1 và chấp nhận đánh đổi mAP@0.5, đặt `--selection_metric micro_f1` và `--min_map50 0.0`.

Chạy tune trực tiếp từ checkpoint trên GPU:
```bash
PYTHONPATH=. python3 utils/tune_thresholds.py \
  --checkpoint ./models/best.pth \
  --source ./public/val/images \
  --data_root ./public \
  --split val \
  --ground_truth ./public/annotations/val.json \
  --official_evaluator utils/official_evaluator/evaluate_predictions.py \
  --selection_metric micro_f1 \
  --min_map50 0.0 \
  --base_conf 0.001 \
  --threshold_values 0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95 \
  --nms_iou_values 0.45,0.50,0.55,0.60,0.65,0.70,0.75 \
  --max_det 100 \
  --nms_max_det 300 \
  --rounds 3 \
  --output_dir threshold_sweeps_max_f1
```

Nếu đã có file dự đoán ngưỡng thấp như `predictions.json`, có thể tune offline nhanh hơn:
```bash
PYTHONPATH=. python3 utils/tune_thresholds.py \
  --predictions predictions.json \
  --data_root public \
  --split val \
  --ground_truth public/annotations/val.json \
  --official_evaluator utils/official_evaluator/evaluate_predictions.py \
  --selection_metric micro_f1 \
  --min_map50 0.0 \
  --base_conf 0.001 \
  --threshold_values 0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95 \
  --nms_iou 0.65 \
  --max_det 100 \
  --rounds 3 \
  --output_dir threshold_sweeps_max_f1
```

Kết quả tune offline hiện tại trên `predictions.json`:
- Threshold theo class: `person=0.55`, `car=0.50`, `dog=0.70`, `cat=0.60`, `chair=0.70`.
- `micro_f1 = 0.850890`, `macro_f1 = 0.850691`, `weighted_f1 = 0.848808`.
- `micro_precision = 0.875851`, `micro_recall = 0.827313`.
- `mAP@0.5 = 0.806175`, số prediction sau lọc: `1909`.
- F1 theo lớp: `person=0.866888`, `car=0.792727`, `dog=0.932693`, `cat=0.948276`, `chair=0.712871`.

Khi suy luận với bộ threshold này, truyền thêm `--thresholds threshold_sweeps_max_f1/best_thresholds.json` cho `predict.py`.

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
