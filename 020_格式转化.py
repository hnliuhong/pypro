import os
import xml.etree.ElementTree as ET

# NEU-DET 的 6 种标准缺陷类别映射
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def convert_box(size, box):
    """将 VOC 的 [xmin, ymin, xmax, ymax] 归一化转换为 YOLO 的 [x_center, y_center, w, h]"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2]) / 2.0 - 1
    y_center = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x_center * dw, y_center * dh, w * dw, h * dh)


def convert_annotation(xml_path, output_txt_path):
    """解析单张 XML 标注文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(output_txt_path, 'w', encoding='utf-8') as out_file:
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip()
            if cls_name not in CLASSES:
                continue
            cls_id = CLASSES.index(cls_name)

            xml_box = obj.find('bndbox')
            b = (float(xml_box.find('xmin').text),
                 float(xml_box.find('ymin').text),
                 float(xml_box.find('xmax').text),
                 float(xml_box.find('ymax').text))

            # 坐标转换与写入
            bb = convert_box((w, h), b)
            out_file.write(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bb]) + '\n')


def process_dataset_split(split_dir):
    """处理训练集或验证集的标注"""
    xml_dir = os.path.join(split_dir, 'annotations')
    txt_dir = os.path.join(split_dir, 'labels')
    os.makedirs(txt_dir, exist_ok=True)

    if not os.path.exists(xml_dir):
        print(f"未找到标注目录: {xml_dir}")
        return

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"开始转换 {split_dir} 中的 {len(xml_files)} 个 XML 标注文件...")

    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        txt_path = os.path.join(txt_dir, xml_file.replace('.xml', '.txt'))
        convert_annotation(xml_path, txt_path)


if __name__ == '__main__':
    # 分别处理 train 和 validation 目录
    process_dataset_split('NEU-DET/train')
    if os.path.exists('NEU-DET/validation'):
        process_dataset_split('NEU-DET/validation')
    print("格式转换完成！Labels 已保存在对应的 labels/ 文件夹中。")