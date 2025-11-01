import os
import cv2
import json
import numpy as np
import paddle
from paddleocr import PaddleOCR
import argparse
import sys


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多语言OCR文本识别工具")
    parser.add_argument(
        "input_dir", nargs="?", type=str, default=None, help="存储图片文件的根目录路径"
    )
    parser.add_argument(
        "-l",
        "--lang",
        nargs="+",
        type=str,
        default=None,
        help="指定一个或多个语言模型, 例如: --lang russian english. 如果不提供，则使用所有模型。",
    )
    parser.add_argument(
        "--output_visual", type=str, default=None, help="可视化结果输出目录"
    )
    parser.add_argument(
        "--output_json", type=str, default=None, help="JSON结果输出目录"
    )
    return parser.parse_args()


def main():
    """主程序"""
    args = parse_arguments()

    INPUT_DIR = args.input_dir
    if not INPUT_DIR:
        print("\n===== 多语言OCR文本识别工具 =====")
        print("错误：请在命令行中提供图片目录路径。")
        print('用法示例: python your_script_name.py "C:\\Path\\To\\Images" --lang korean')
        return

    if not os.path.isdir(INPUT_DIR):
        print(f"错误: 路径 '{INPUT_DIR}' 不是一个有效的目录!")
        return

    # 如果用户未指定输出目录，则根据输入目录名自动创建
    input_dir_name = os.path.basename(os.path.normpath(INPUT_DIR))
    OUTPUT_VISUAL = (
        args.output_visual if args.output_visual else f"result_{input_dir_name}_visual"
    )
    OUTPUT_JSON = (
        args.output_json if args.output_json else f"result_{input_dir_name}_json"
    )

    print(f"\n输入目录: {INPUT_DIR}")
    print(f"可视化输出目录: {OUTPUT_VISUAL}")
    print(f"JSON输出目录: {OUTPUT_JSON}")

    for d in [OUTPUT_VISUAL, OUTPUT_JSON]:
        if not os.path.exists(d):
            os.makedirs(d)

    LINE_MIN_CONFIDENCE = 0.2
    LINE_MIN_TEXT_LENGTH = 1
    LINE_MIN_AREA_RATIO = 0.00001

    common_params = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
    }

    all_available_experts = {
        "russian": lambda: PaddleOCR(lang="ru", **common_params),
        "arabic": lambda: PaddleOCR(lang="ar", **common_params),
        "english": lambda: PaddleOCR(lang="en", **common_params),
        "japan": lambda: PaddleOCR(lang="japan", **common_params),
        "korean": lambda: PaddleOCR(lang="korean", **common_params),
        "Spanish": lambda: PaddleOCR(lang="es", **common_params),
        "french": lambda: PaddleOCR(lang="french", **common_params),
    }

    selected_languages = args.lang
    if not selected_languages:
        print("未指定语言，将加载所有可用语言模型...")
        selected_languages = list(all_available_experts.keys())

    # 只加载用户选择的语言模型
    ocr_experts = {}
    for lang_name in selected_languages:
        if lang_name in all_available_experts:
            print(f"  > 正在加载 '{lang_name}' 语言模型...")
            ocr_experts[lang_name] = all_available_experts[lang_name]()
        else:
            print(f"\n错误: 不支持的语言 '{lang_name}'。")
            print(f"可用语言包括: {', '.join(all_available_experts.keys())}")
            return

    print("\n所有选定语言模型加载完毕。")

    # --- 主程序：改为递归搜索 ---
    image_files = []
    for root, _, files in os.walk(INPUT_DIR):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(root, filename))

    if not image_files:
        print(f"\n警告：在目录 '{INPUT_DIR}' 及其子目录中未找到任何图片文件！")
        return
    print(f"\n找到 {len(image_files)} 张图片，开始处理...")

    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"\n--- 正在处理图片: {filename} ---")

        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"  > 警告: 无法读取图片 {image_path}, 已跳过。")
            continue

        overall_best_result = {
            "results": [],
            "line_count": -1,
            "language_expert": "None",
        }
        image_height, image_width, _ = original_image.shape
        total_area = image_width * image_height

        for lang_name, ocr_engine in ocr_experts.items():
            print(f"\n  --- 使用语言专家 '{lang_name}' 进行识别 ---")

            # 直接在原始图片上运行OCR，不再有预处理步骤
            ocr_results = ocr_engine.predict(original_image)

            high_quality_lines = []

            # --- 关键改动：恢复您原版的、能稳定运行的结果解析逻辑 ---
            if ocr_results and ocr_results[0]:
                # 兼容旧版 PaddleOCR 可能返回的 dict 结构
                if isinstance(ocr_results[0], dict):
                    results_dict = ocr_results[0]
                    boxes = results_dict.get("dt_polys", [])
                    texts = results_dict.get("rec_texts", [])
                    scores = results_dict.get("rec_scores", [])
                    # 将三个列表压缩成一个进行循环
                    for bbox, text, confidence in zip(boxes, texts, scores):
                        if (
                                confidence < LINE_MIN_CONFIDENCE
                                or len(text.strip()) < LINE_MIN_TEXT_LENGTH
                        ):
                            continue
                        box_np = np.array(bbox).astype(np.int32)
                        if (cv2.contourArea(box_np) / total_area) < LINE_MIN_AREA_RATIO:
                            continue
                        high_quality_lines.append([bbox, (text, confidence)])
                # 兼容新版 PaddleOCR 返回的 list 结构
                else:
                    results_list = ocr_results[0]
                    for line_info in results_list:
                        # 安全地解包，避免出错
                        if isinstance(line_info, list) and len(line_info) == 2:
                            bbox, (text, confidence) = line_info
                            if (
                                    confidence < LINE_MIN_CONFIDENCE
                                    or len(text.strip()) < LINE_MIN_TEXT_LENGTH
                            ):
                                continue
                            box_np = np.array(bbox).astype(np.int32)
                            if (cv2.contourArea(box_np) / total_area) < LINE_MIN_AREA_RATIO:
                                continue
                            high_quality_lines.append([bbox, (text, confidence)])

            line_count = len(high_quality_lines)

            # 对比当前语言模型的结果和已有的最佳结果
            if line_count > overall_best_result["line_count"]:
                print(f"  *** 新的全局最佳结果! ({line_count} 行) ***")
                overall_best_result = {
                    "results": high_quality_lines,
                    "line_count": line_count,
                    "language_expert": lang_name,
                }
            else:
                print(f"  > 找到 {line_count} 行，未超过当前最佳结果 ({overall_best_result['line_count']} 行)。")

        # 结果汇总和文件保存部分，完全保留你的逻辑
        print(f"\n--- 图片 '{filename}' 最终诊断 ---")
        print(f"  > 胜出语言专家: '{overall_best_result['language_expert']}'")
        visual_image = original_image.copy()
        output_data = []
        if overall_best_result["line_count"] > 0:
            for line_data in overall_best_result["results"]:
                bbox, (text, confidence) = line_data
                box = np.array(bbox).astype(np.int32)
                cv2.polylines(
                    visual_image, [box], isClosed=True, color=(0, 255, 0), thickness=2
                )
                min_x, min_y = int(min(p[0] for p in bbox)), int(
                    min(p[1] for p in bbox)
                )
                max_x, max_y = int(max(p[0] for p in bbox)), int(
                    max(p[1] for p in bbox)
                )
                simple_bbox = [[min_x, min_y], [max_x, max_y]]
                output_data.append({"bbox": simple_bbox, "text": text})
                print(f"  > 识别结果: '{text}' (置信度: {confidence:.2f})")

        rel_path = os.path.relpath(image_path, INPUT_DIR)
        output_visual_dir = os.path.join(OUTPUT_VISUAL, os.path.dirname(rel_path))
        output_json_dir = os.path.join(OUTPUT_JSON, os.path.dirname(rel_path))
        os.makedirs(output_visual_dir, exist_ok=True)
        os.makedirs(output_json_dir, exist_ok=True)
        visual_path = os.path.join(output_visual_dir, f"vis_{filename}")
        json_path = os.path.join(
            output_json_dir, f"{os.path.splitext(filename)[0]}.json"
        )

        cv2.imwrite(visual_path, visual_image)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"  > 已生成 {len(output_data)} 个高质量文本行。")
        print(f"  > 可视化结果保存至: {visual_path}")
        print(f"  > JSON结果保存至: {json_path}")

    print("\n\n所有图片处理完毕！")


if __name__ == "__main__":
    main()