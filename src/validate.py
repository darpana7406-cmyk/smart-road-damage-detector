from ultralytics import YOLO


def main():
    model = YOLO(r"D:\PBEL\runs\yolov8n_rdd2022\weights\best.pt")

    # workers=0 is required on Windows to avoid multiprocessing spawn errors
    metrics = model.val(
        data=r"D:\PBEL\data\yolo\data.yaml",
        workers=0,
        batch=8,
        imgsz=640,
        split="val",
    )

    print("\n===== VALIDATION METRICS =====")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP@50-95 : {metrics.box.map:.4f}")
    print("==============================\n")

    print("Per-class results:")
    for i, name in metrics.names.items():
        print(f"  {name:25s}  P={metrics.box.p[i]:.3f}  "
              f"R={metrics.box.r[i]:.3f}  mAP50={metrics.box.ap50[i]:.3f}")


if __name__ == "__main__":
    main()