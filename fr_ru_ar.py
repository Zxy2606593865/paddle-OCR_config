import os
import cv2
import json
import numpy as np
from paddleocr import PaddleOCR
import time

# --- 核心配置 ---
INPUT_DIR = 'ar_jpg'
OUTPUT_VISUAL = 'result_ar'
OUTPUT_JSON = 'result_ar'
for d in [OUTPUT_VISUAL, OUTPUT_JSON]:
    if not os.path.exists(d): os.makedirs(d)

# --- “品控中心” ---
LINE_MIN_CONFIDENCE = 0.4
LINE_MIN_TEXT_LENGTH = 1
LINE_MIN_AREA_RATIO = 0.0001
# --- 加载“多语言专家委员会” ---
print("正在加载 PaddleOCR 模型 (精简多语言模式)...")
common_params = {
    # 关闭文档图像方向整体分类
    'use_doc_orientation_classify':False,
    # 关闭文档图像的扭曲校正
    'use_doc_unwarping':False,
}

# PaddleOCR() 的用法本身是完全正确的，我们保持原样
ocr_experts = {
    'russian': PaddleOCR(lang='ru', **common_params),
    'arabic': PaddleOCR(lang='ar', **common_params),
    'english': PaddleOCR(lang='en', **common_params),
    'japan': PaddleOCR(lang='japan', **common_params),
    'korean': PaddleOCR(lang='korean', **common_params),
    'Spanish': PaddleOCR(lang='es', **common_params),
    'french': PaddleOCR(lang='french', **common_params),

}
print("所有语言模型加载完毕。")

# --- 预处理专家团队 ---
def preprocess_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)


def preprocess_sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def preprocess_closing(image):
    """
    使用形态学闭运算来连接断裂的字符笔画，并使字体更清晰。
    """
    # 首先，为了让形态学操作更有效，我们通常在灰度图上进行
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 定义一个“结构元素”或“内核”。它定义了膨胀/腐蚀操作的邻域范围。
    # 2x2 或 3x3 的矩形内核对于常规大小的文本效果最好。
    kernel = np.ones((2, 2), np.uint8)

    # 执行闭运算
    # cv2.morphologyEx 是一个多功能的形态学函数
    # cv2.MORPH_CLOSE 指定我们要做的是闭运算
    closed_image = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 将处理后的灰度图转换回BGR格式，以匹配其他管道的输出
    return cv2.cvtColor(closed_image, cv2.COLOR_GRAY2BGR)

# --- 主程序 ---
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(INPUT_DIR, filename)
        print(f"\n--- 正在处理图片: {filename} ---")

        original_image = cv2.imread(image_path)
        if original_image is None: continue

        overall_best_result = {"results": [], "line_count": -1, "language_expert": "None",
                               "preprocess_strategy": "None"}
        image_height, image_width, _ = original_image.shape
        total_area = image_width * image_height

        for lang_name, ocr_engine in ocr_experts.items():
            print(f"\n  --- 语言专家 '{lang_name}' 开始诊断 ---")

            pipelines = {
                "Original": original_image,
                "CLAHE": preprocess_clahe(original_image),
                "Sharpen": preprocess_sharpen(original_image),
                "Closing": preprocess_closing(original_image)
            }
            best_result_for_this_lang = {"results": [], "line_count": -1, "strategy": "None"}

            for strategy_name, img in pipelines.items():
                print(f"    > 尝试预处理策略: '{strategy_name}'...")
                # 【最终语法修正】使用推荐的 predict 方法，并移除已废弃的 cls 参数
                ocr_results = ocr_engine.predict(img)

                lines_this_image = ocr_results[0] if ocr_results and ocr_results[0] is not None else []

                high_quality_lines = []
                if ocr_results and ocr_results[0]:
                    results_dict = ocr_results[0]
                    boxes = results_dict.get('dt_polys', [])
                    texts = results_dict.get('rec_texts', [])
                    scores = results_dict.get('rec_scores', [])

                    for bbox, text, confidence in zip(boxes, texts, scores):
                        if confidence < LINE_MIN_CONFIDENCE or len(text.strip()) < LINE_MIN_TEXT_LENGTH: continue
                        box_np = np.array(bbox).astype(np.int32)
                        if (cv2.contourArea(box_np) / total_area) < LINE_MIN_AREA_RATIO: continue

                        # 重新组合成 [bbox, (text, confidence)] 格式
                        high_quality_lines.append([bbox.tolist(), (text, confidence)])

                line_count = len(high_quality_lines)
                if line_count > best_result_for_this_lang["line_count"]:
                    best_result_for_this_lang = {"results": high_quality_lines, "line_count": line_count,
                                                 "strategy": strategy_name}

            print(
                f"    > '{lang_name}' 专家诊断完毕，最佳策略 '{best_result_for_this_lang['strategy']}' 找到 {best_result_for_this_lang['line_count']} 行。")

            if best_result_for_this_lang['line_count'] > overall_best_result['line_count']:
                print("  *** 新的全局最佳结果! ***")
                overall_best_result = {
                    "results": best_result_for_this_lang['results'],
                    "line_count": best_result_for_this_lang['line_count'],
                    "language_expert": lang_name,
                    "preprocess_strategy": best_result_for_this_lang['strategy']
                }

        print(f"\n--- 图片 '{filename}' 最终诊断 ---")
        print(f"  > 胜出语言专家: '{overall_best_result['language_expert']}'")
        print(f"  > 胜出预处理策略: '{overall_best_result['preprocess_strategy']}'")

        visual_image = original_image.copy()
        output_data = []

        if overall_best_result['line_count'] > 0:
            for line_data in overall_best_result['results']:
                bbox, (text, confidence) = line_data
                box = np.array(bbox).astype(np.int32)
                cv2.polylines(visual_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)

                min_x, min_y = int(min(p[0] for p in bbox)), int(min(p[1] for p in bbox))
                max_x, max_y = int(max(p[0] for p in bbox)), int(max(p[1] for p in bbox))
                simple_bbox = [[min_x, min_y], [max_x, max_y]]
                output_data.append({'bbox': simple_bbox, 'text': text})
                print(f"  > 识别结果: '{text}' (置信度: {confidence:.2f})")

        cv2.imwrite(os.path.join(OUTPUT_VISUAL, f"vis_{filename}"), visual_image)
        json_path = os.path.join(OUTPUT_JSON, f"{os.path.splitext(filename)[0]}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"  > 已生成 {len(output_data)} 个高质量文本行。")

print("\n\n所有图片处理完毕！")