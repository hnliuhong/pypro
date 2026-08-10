import cv2
from ultralytics import YOLO

def predict_single_image(img_path):
    # 1. 加载训练好的最佳权重
    model = YOLO('runs/detect/neu_fault_exp-2/weights/best.pt')
    # 2. 模型推理
    results = model.predict(source=img_path, conf=0.25)
    # 3. 打印预测缺陷类别与置信度
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            print(f"检测到缺陷目标: {cls_name} | 置信度: {conf:.2f}")
    # 4. 绘制检测框并显示
    annotated_img = results[0].plot()
    cv2.imshow("NEU Defect Detection", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 传入一张训练集或测试集中的图片进行推理测试
    test_image = 'NEU-DET/train/images/inclusion/inclusion_1.jpg'
    predict_single_image(test_image)