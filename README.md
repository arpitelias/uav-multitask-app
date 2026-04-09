# UAV Multi-Task Interpretation — Streamlit Demo App

Interactive deployment application for the MSc AI project **"Attention-Guided Multi-Task Learning for Unified UAV Image Interpretation"** by Arpit Joshua Elias (24257567), National College of Ireland, 2026.

## Features

- **Single Model Inference** — Upload drone images and run inference with any of 6 trained models
- **Model Comparison** — Compare two models side-by-side on the same input
- **Visual Outputs** — Segmentation maps, scene classification probabilities, detection heatmaps
- **Project Overview** — Full ablation results and key findings

## Folder Structure

```
streamlit_app/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/                 # Trained model weights (.pth files)
│   ├── resnet50_scene.pth
│   ├── resnet50_seg.pth
│   ├── mtl_standard.pth
│   ├── mtl_cross_attention.pth
│   ├── swin_mtl.pth
│   └── three_task_mtl_best.pth
└── samples/                # Sample drone images for demo
    ├── sample_1.jpg
    ├── sample_2.jpg
    └── ...
```

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model weights

The trained model weights are produced by the Kaggle notebooks in this project. After running the committed notebooks:

- From `/kaggle/working/single_task/`: download `resnet50_scene.pth`, `resnet50_seg.pth`
- From `/kaggle/working/multi_task/`: download `mtl_standard.pth`, `mtl_cross_attention.pth`, `swin_mtl.pth`, `three_task_mtl_best.pth`

Place all `.pth` files in the `models/` directory.

### 3. Copy sample images

The notebook 05 commit output includes a folder `app_samples/` with sample drone images. Copy its contents to the `samples/` directory.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Models Included

| Model | Architecture | Parameters | Speed (CPU) |
|-------|--------------|-----------|-------------|
| Single-Task (ResNet-50) | ResNet-50 classifier | 23.5M | ~5.7 ms |
| Single-Task Segmentation | ResNet-50 U-Net | 31.5M | ~7.4 ms |
| Standard MTL | Shared backbone, 2 heads | 24.6M | ~5.5 ms |
| Cross-Task Attention | Novel attention module | 25.7M | ~6.4 ms |
| Swin Transformer MTL | Swin-T backbone | 27.8M | ~8.2 ms |
| 3-Task MTL | Seg + Cls + Det | 26.3M | ~6.9 ms |

## Notes

- The app runs entirely on CPU, so no GPU is required for inference
- Models load lazily on first selection and are cached thereafter
- If a model's `.pth` file is missing, the app will still load the architecture but warn that predictions will not be meaningful (random weights)

## Author

**Arpit Joshua Elias** (24257567)  
MSc Artificial Intelligence  
National College of Ireland  
Supervisor: Prof. Furqan Rustam
