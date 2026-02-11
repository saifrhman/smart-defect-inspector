-- Experiments table
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  task TEXT NOT NULL,                -- e.g., "preprocess", "detection", "segmentation"
  model_name TEXT,                   -- e.g., "yolov8n", "unet"
  dataset_name TEXT,
  notes TEXT
);

-- Metrics table (one row per metric per run)
CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  name TEXT NOT NULL,                -- e.g., "iou", "dice", "map50"
  value REAL NOT NULL,
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
