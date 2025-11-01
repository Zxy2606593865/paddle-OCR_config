import os
import cv2
import json
import yaml  # 用于读取 yml 配置文件
import numpy as np
import paddle
from paddleocr import PaddleOCR
import sys
from tqdm import tqdm
import gc  # 用于强制进行垃圾回收
import xml.etree.ElementTree as ET
from xml.dom import minidom


def get_user_input(available_languages):
    """
    通过命令行交互，获取所有本次任务需要用户手动指定的、可变的参数。
    """
    print("===== 多语言OCR预标注工具 (灵活路径版) =====")

    while True:
        input_dir = input("\n>>> 请输入本次要处理的图片文件夹路径: ").strip()
        if os.path.isdir(input_dir):
            break
        else:
            print(f"错误: 路径 '{input_dir}' 无效！")

    output_visual = input(">>> 请输入可视化结果输出文件夹路径 (留空自动): ").strip()
    output_xml_json = input(">>> 请输入XML/JSON输出文件夹路径 (留空自动): ").strip()

    print("\n--- 可用的语言模型 (来自 settings.yml) ---")
    for i, lang in enumerate(available_languages, 1):
        print(f"{i}. {lang}")

    selected_languages = []
    while True:
        lang_input = input("\n>>> 请输入语言编号或名称 (多个用逗号隔开，留空全选): ").strip()
        if not lang_input:
            selected_languages = available_languages
            break
        lang_inputs = [x.strip().lower() for x in lang_input.split(",")]
        selected_languages, valid_input = [], True
        for lang in lang_inputs:
            if lang.isdigit():
                idx = int(lang) - 1
                if 0 <= idx < len(available_languages):
                    selected_languages.append(available_languages[idx])
                else:
                    print(f"错误: 无效编号 {lang}。")
                    valid_input = False
                    break
            elif lang in [l.lower() for l in available_languages]:
                selected_languages.append(next(l for l in available_languages if l.lower() == lang))
            else:
                print(f"错误: 不支持的语言 '{lang}'。")
                valid_input = False
                break
        if valid_input:
            break

    return input_dir, output_visual, output_xml_json, selected_languages


def write_xml(output_path, image_name, image_width, image_height, detections):
    """
    将最终的识别结果写入一个标准格式的XML文件。
    """
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = os.path.basename(os.path.dirname(output_path))
    filename = ET.SubElement(root, "filename")
    filename.text = image_name
    path = ET.SubElement(root, "path")
    path.text = os.path.abspath(output_path)
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image_width)
    height = ET.SubElement(size, "height")
    height.text = str(image_height)
    depth = ET.SubElement(root, "depth")
    depth.text = "3"
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    for detection in detections:
        bbox, text = detection["bbox"], detection["text"]
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = "text"
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(bbox[0][0]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(bbox[0][1]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(bbox[1][0]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(bbox[1][1]))
        text_elem = ET.SubElement(obj, "text")
        text_elem.text = text
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


def main():
    CONFIG_FILE_PATH = "settings.yml"
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"--- 成功加载应用配置文件: {CONFIG_FILE_PATH} ---")
    except Exception as e:
        print(f"错误: 加载或解析配置文件 '{CONFIG_FILE_PATH}' 失败: {e}\n程序无法继续。")
        return

    post_proc_config = config.get('PostProcessing', {})
    LINE_MIN_CONFIDENCE = post_proc_config.get('min_confidence', 0.2)
    LINE_MIN_TEXT_LENGTH = post_proc_config.get('min_text_length', 1)
    LINE_MIN_AREA_RATIO = post_proc_config.get('min_area_ratio', 0.00001)

    all_available_experts_configs = config.get('Models', {})
    if not all_available_experts_configs:
        print("错误: 配置文件中 'Models' 部分为空。")
        return

    available_languages = list(all_available_experts_configs.keys())
    INPUT_DIR, output_visual, output_xml_json, selected_languages = get_user_input(available_languages)

    input_dir_name = os.path.basename(os.path.normpath(INPUT_DIR))
    OUTPUT_VISUAL = output_visual or os.path.join(INPUT_DIR, f"result_{input_dir_name}_visual")
    OUTPUT_XML = output_xml_json or os.path.join(INPUT_DIR, f"result_{input_dir_name}_xml_json")
    OUTPUT_JSON = OUTPUT_XML

    print(
        f"\n--- 配置确认 ---\n  图片输入: {INPUT_DIR}\n  可视化输出: {OUTPUT_VISUAL}\n  XML/JSON输出: {OUTPUT_XML}\n  选用语言: {', '.join(selected_languages)}")
    for d in [OUTPUT_VISUAL, OUTPUT_XML, OUTPUT_JSON]:
        os.makedirs(d, exist_ok=True)

    image_files = [os.path.join(r, f) for r, _, fs in os.walk(INPUT_DIR) for f in fs if
                   f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not image_files:
        print(f"\n警告：目录 '{INPUT_DIR}' 中未找到图片！")
        return
    print(f"\n找到 {len(image_files)} 张图片，开始第一阶段：模型处理...")

    all_images_results = {img_path: {} for img_path in image_files}

    for lang_name in selected_languages:
        config_path = all_available_experts_configs.get(lang_name)
        if not (config_path and os.path.exists(config_path)):
            print(f"\n警告: 找不到 '{lang_name}' 的产业线配置文件 '{config_path}'，已跳过。")
            continue

        print(f"\n--- 正在加载 '{lang_name}' 模型 ---")
        try:
            # 最终的、正确的加载方式
            ocr_engine = PaddleOCR(
                paddlex_config=config_path,
            )
        except Exception as e:
            print(f"错误：加载 '{lang_name}' 模型失败: {e}")
            continue

        for image_path in tqdm(image_files, desc=f"使用[{lang_name}]", unit="张", ncols=100):
            try:
                img_data = np.fromfile(image_path, dtype='uint8')
                original_image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if original_image is None: continue
            except Exception:
                continue

            total_area = original_image.shape[0] * original_image.shape[1]

            try:
                # ================= [ 最终的、正确的推理与解析 ] =================
                # 1. 使用 .predict() 接口，获取包含分离键的原始字典输出
                ocr_results_paddlex = ocr_engine.predict(original_image)

                # 2. 从字典中安全地获取并行的结果列表
                output_dict = ocr_results_paddlex[0] if ocr_results_paddlex else {}
                dt_polys = output_dict.get('dt_polys', [])
                rec_texts = output_dict.get('rec_texts', [])
                rec_scores = output_dict.get('rec_scores', [])

                # 3. 使用 zip 将并行的列表“缝合”成标准格式
                parsed_lines = []
                for bbox, text, score in zip(dt_polys, rec_texts, rec_scores):
                    # `bbox` 已经是我们需要的坐标点列表
                    # `text` 和 `score` 也是现成的
                    parsed_lines.append([bbox, (text, score)])

                # 4. 将转换后的结果包装起来，以完美兼容后续代码
                ocr_results = [parsed_lines]
                # ====================================================================

            except Exception as e:
                tqdm.write(f"\n警告：模型推理失败，已跳过图片 {os.path.basename(image_path)}: {e}")
                continue

            # 您原来优秀的解析代码现在可以无缝衔接了！
            high_quality_lines = []
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    try:
                        bbox, (text, confidence) = line
                        if (confidence < LINE_MIN_CONFIDENCE or len(text.strip()) < LINE_MIN_TEXT_LENGTH):
                            continue
                        area = cv2.contourArea(np.array(bbox, dtype=np.int32))
                        if (area / total_area) < LINE_MIN_AREA_RATIO:
                            continue
                        high_quality_lines.append([bbox, (text, confidence)])
                    except (TypeError, ValueError):
                        continue

            all_images_results[image_path][lang_name] = {"results": high_quality_lines,
                                                         "line_count": len(high_quality_lines)}

        print(f"\n--- '{lang_name}' 模型处理完毕，释放内存 ---")
        del ocr_engine
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
        gc.collect()

    print(f"\n\n所有语言处理完成！开始第二阶段：汇总并保存最佳结果...")

    for image_path in tqdm(image_files, desc="汇总保存", unit="张", ncols=100):
        filename = os.path.basename(image_path)
        lang_results = all_images_results.get(image_path)
        if not lang_results:
            tqdm.write(f"图片 {filename} 未获取到任何有效结果，已跳过。")
            continue

        best_lang = max(lang_results, key=lambda lang: lang_results[lang]['line_count'])
        best_result = lang_results[best_lang]

        tqdm.write(f"\n图片 '{filename}': 最佳语言 '{best_lang}' ({best_result['line_count']}行)")

        if best_result["line_count"] > 0:
            img_data = np.fromfile(image_path, dtype='uint8')
            original_image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if original_image is None: continue

            visual_image = original_image.copy()
            H, W, _ = original_image.shape
            output_data = []

            for line_data in best_result["results"]:
                bbox, (text, _) = line_data
                box = np.array(bbox, dtype=np.int32)
                cv2.polylines(visual_image, [box], True, (0, 255, 0), 2)

                x_coords, y_coords = box[:, 0], box[:, 1]
                simple_bbox = [
                    [int(min(x_coords)), int(min(y_coords))],
                    [int(max(x_coords)), int(max(y_coords))]
                ]
                output_data.append({"bbox": simple_bbox, "text": text})

            rel_path = os.path.relpath(os.path.dirname(image_path), INPUT_DIR)
            if rel_path == '.': rel_path = ''
            output_visual_dir = os.path.join(OUTPUT_VISUAL, rel_path)
            os.makedirs(output_visual_dir, exist_ok=True)
            output_xml_dir = os.path.join(OUTPUT_XML, rel_path)
            os.makedirs(output_xml_dir, exist_ok=True)

            base_filename = os.path.splitext(filename)[0]
            visual_path = os.path.join(output_visual_dir, f"vis_{filename}")
            xml_path = os.path.join(output_xml_dir, f"{base_filename}.xml")
            json_path = os.path.join(output_xml_dir, f"{base_filename}.json")

            try:
                cv2.imencode('.png', visual_image)[1].tofile(visual_path)
            except Exception as e:
                tqdm.write(f"  警告: 保存可视化图片 '{visual_path}' 失败: {e}")

            write_xml(xml_path, filename, W, H, output_data)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
        else:
            tqdm.write(f"  图片 '{filename}' 未识别出有效文本，跳过文件生成。")

    print("\n\n所有图片处理完毕！")


if __name__ == "__main__":
    main()