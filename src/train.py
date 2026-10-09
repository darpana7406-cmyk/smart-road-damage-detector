from ultralytics import YOLO
import torch
import os

if __name__ == "__main__":
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    if device == 0:
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    last_weights = r"D:\PBEL\runs\yolov8n_rdd2022\weights\last.pt"

    if os.path.exists(last_weights):
        print(f"\n>> Resuming from: {last_weights}")
        model = YOLO(last_weights)
        resume = True
    else:
        print("\n>> Starting fresh")
        model = YOLO("yolov8n.pt")
        resume = False

    model.train(
        data=r"D:\PBEL\data\yolo\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,                    # 16x more than before — GPU has headroom
        patience=20,
        project=r"D:\PBEL\runs",
        name="yolov8n_rdd2022",
        device=device,
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        plots=True,
        cache="ram",                 # 12 GB RAM → safe
        amp=True,
        val=True,
        save=True,
        resume=resume,
        exist_ok=True,
    )