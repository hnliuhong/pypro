from ultralytics import YOLO

def train():
    # 1. 加载轻量预训练模型（YOLOv8n 最适合教学演示，训练速度快）
    model = YOLO('last.pt')

    # 2. 训练配置
    results = model.train(
        data='neu_dataset.yaml',  # 指定数据集配置文件
        epochs=70,               # 入门 Demo 30 轮即可查看明显的效果
        imgsz=200,               # NEU-DET 图像原始分辨率一般为 200x200
        batch=16,                # 批次大小
        workers=2,               # 数据加载线程数
        name='neu_fault_exp'     # 实验输出保存目录名
    )
    print("训练结束！权重与评估图表保存在 runs/detect/neu_fault_exp 中。")

if __name__ == '__main__':
    train()