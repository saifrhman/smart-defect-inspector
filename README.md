# smart-defect-inspector

## Quickstart

### 1) Install dependencies
```bash
pip install -r requirements.txt


## Pipeline (so far)

1. Generate a sample image  
   `python -m src.utils.smoke_test`

2. Preprocess the image + log metrics to SQLite  
   `python -m src.preprocess.demo_preprocess`

3. Query latest runs + metrics  
   `python -m src.db.query_latest_runs`

4. Save a before/after comparison figure  
   `python -m src.eval.visualize_preprocess`

5. Run YOLOv8 inference on the example image (saves overlay + logs to SQLite)  
   `python -m src.detection.infer_yolo`

6. Generate a synthetic segmentation mask (ground truth)  
   `python -m src.segmentation.make_synthetic_mask`

7. Visualize mask overlay + log mask stats to SQLite  
   `python -m src.segmentation.visualize_mask_overlay`


## Generated outputs

- `outputs/smoke_test.png`
- `outputs/smoke_test_preprocessed.png`
- `outputs/preprocess_comparison.png`
- `outputs/experiments.sqlite` (runs + metrics)
- `outputs/yolo_example_pred.png`
- `data/processed/masks/example_mask.png`
- `outputs/segmentation_mask_overlay.png`

