# Explanations & Visualizations Directory

Thư mục này dùng để lưu trữ các kết quả phân tích giải thích mô hình (model explanations), attention heatmaps, và các biểu đồ visualize phục vụ việc debug các lượt inference theo từng phiên (`session_id`).

## Cấu trúc thư mục

```
explanations/
└── <session_id>/                          # Ví dụ: 20261015T162257+0700_797af814
    ├── cycle_000015_rollout_step0_dim0/   # Thư mục xuất ra từ scripts/explain_session.py
    │   ├── cam_head.png                   # Attention heatmap overlay trên ảnh camera đầu
    │   ├── cam_head.npz                   # Attention tensor + RGB + token maps
    │   ├── cam_left_wrist.png             # Heatmap camera cổ tay trái
    │   ├── cam_left_wrist.npz
    │   ├── cam_right_wrist.png            # Heatmap camera cổ tay phải
    │   ├── cam_right_wrist.npz
    │   └── metadata.json                  # Replay error, method, tolerance, cycle info
    └── ...
```

## Lệnh tạo visualization & giải thích cho một cycle

```bash
/home/anhnb9/miniconda3/envs/pelican_vla/bin/python scripts/explain_session.py \
  logs/sessions/20261015T162257+0700_797af814 \
  --cycle 15 \
  --method rollout
```
Kết quả sẽ tự động lưu vào `explanations/20261015T162257+0700_797af814/` (đồng thời có symlink liên kết hai chiều từ thư mục session tương ứng).
