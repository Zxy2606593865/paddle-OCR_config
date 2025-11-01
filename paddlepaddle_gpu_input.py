import os
import cv2
import json
import numpy as np
import paddle
from paddleocr import PaddleOCR
import time
import argparse  # 新增：用于解析命令行参数
import sys  # 新增：用于程序交互


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
        print('用法示例: python fr_ru_ar.py "C:\\Path\\To\\Images" --lang russian')
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

    LINE_MIN_CONFIDENCE = 0.15
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

    def preprocess_adaptive_threshold(image):
        """
        文档与扫描件专用：转为灰度图后进行自适应二值化，生成高对比度的黑白图像。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ADAPTIVE_THRESH_GAUSSIAN_C: 阈值是邻域像素的加权和
        # THRESH_BINARY_INV: 反转二值化，使文字为白色(255)，背景为黑色(0)，有时OCR模型更喜欢这种格式
        binary_image = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4
        )
        # 将处理后的单通道灰度图转回BGR，以匹配后续流程
        return cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    def preprocess_clahe_only(image):
        """
        截图与中等质量实景图专用：仅使用CLAHE增强局部对比度。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    def preprocess_low_quality_rescue(image):
        """
        低质量实景图专用（组合管道）：去噪 -> CLAHE -> 锐化
        """
        # 步骤1: 使用fastNlMeansDenoisingColored去噪，效果优于高斯模糊且能保留更多边缘细节
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # 步骤2: 对去噪后的图片进行CLAHE
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        clahe_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        # 步骤3: 对比度增强后进行锐化
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(clahe_img, -1, kernel)

        return sharpened

    # --- 4. 主程序：改为递归搜索，并保持你的核心识别逻辑 ---
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

        # (你写的核心诊断和对比逻辑完全保留，因为它非常有效)
        overall_best_result = {
            "results": [],
            "line_count": -1,
            "language_expert": "None",
            "preprocess_strategy": "None",
        }
        image_height, image_width, _ = original_image.shape
        total_area = image_width * image_height

        for lang_name, ocr_engine in ocr_experts.items():
            print(f"\n  --- 语言专家 '{lang_name}' 开始诊断 ---")

            pipelines = {
                "Original": original_image,
                "adaptive_threshold": preprocess_adaptive_threshold(original_image),
                "clahe_only": preprocess_clahe_only(original_image),
                "low_quality_rescue": preprocess_low_quality_rescue(original_image),
            }
            best_result_for_this_lang = {
                "results": [],
                "line_count": -1,
                "strategy": "None",
            }

            for strategy_name, img in pipelines.items():
                print(f"    > 尝试预处理策略: '{strategy_name}'...")
                ocr_results = ocr_engine.predict(img)

                # (你写的这个结果解析逻辑是正确的，完全保留)
                high_quality_lines = []
                if ocr_results and ocr_results[0]:
                    results_dict = ocr_results[0]
                    boxes = results_dict.get("dt_polys", [])
                    texts = results_dict.get("rec_texts", [])
                    scores = results_dict.get("rec_scores", [])
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

                line_count = len(high_quality_lines)
                if line_count > best_result_for_this_lang["line_count"]:
                    best_result_for_this_lang = {
                        "results": high_quality_lines,
                        "line_count": line_count,
                        "strategy": strategy_name,
                    }

            print(
                f"    > '{lang_name}' 专家诊断完毕，最佳策略 '{best_result_for_this_lang['strategy']}' 找到 {best_result_for_this_lang['line_count']} 行。"
            )

            if (
                best_result_for_this_lang["line_count"]
                > overall_best_result["line_count"]
            ):
                print("  *** 新的全局最佳结果! ***")
                overall_best_result = {
                    "results": best_result_for_this_lang["results"],
                    "line_count": best_result_for_this_lang["line_count"],
                    "language_expert": lang_name,
                    "preprocess_strategy": best_result_for_this_lang["strategy"],
                }

        # (结果汇总和文件保存部分，完全保留你的逻辑)
        print(f"\n--- 图片 '{filename}' 最终诊断 ---")
        print(f"  > 胜出语言专家: '{overall_best_result['language_expert']}'")
        print(f"  > 胜出预处理策略: '{overall_best_result['preprocess_strategy']}'")
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

        # 为了支持子目录，保存路径做一点小小的调整
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


# --- 5. 新增：标准的Python程序入口 ---
if __name__ == "__main__":
    main()
