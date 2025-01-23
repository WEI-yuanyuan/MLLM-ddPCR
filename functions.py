import cv2
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt
import pandas as pd
import math
from config import *
from concurrent.futures import ThreadPoolExecutor


def has_multiple_components(mask):
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return num_labels > 2

def show_bbox(image, bbox, classes, thickness=2):
    # 确保输入图像是RGB格式
    if len(image.shape) == 2:  # 灰度图
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        if image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # BGR格式的颜色
    if classes == '1':
        color = (0, 0, 255)  # 红色
    elif classes == '0':
        color = (255, 0, 0)  # 蓝色
    elif classes == 'TD':
        color = (26, 221, 26)  # 绿色
    
    x1, y1, width, height = bbox
    x2 = x1 + width
    y2 = y1 + height
    
    # 绘制边框
    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    # 转回RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def findFalsyAnnotationsIndex(sorted_anns, removeArrayList):
    for index_to_remove, ann in enumerate(sorted_anns):
        uint8_mask = (ann['segmentation'] * 255).astype(np.uint8)
        if uint8_mask is not None and has_multiple_components(uint8_mask):
            removeArrayList.add(index_to_remove)
    return removeArrayList

def removeFalsyAnnotations(sorted_anns, removeArrayList):
    for i in sorted(removeArrayList, reverse=True):
        del sorted_anns[i]
    return sorted_anns

def convert_bbox_to_yolo(image_size, bbox):
    img_h, img_w = image_size
    x1, y1, width, height = bbox
    x_center = (x1 + width / 2) / img_w
    y_center = (y1 + height / 2) / img_h
    width /= img_w
    height /= img_h
    return x_center, y_center, width, height

def removeOutOfSizeLimitAnnotations(sorted_anns):
    wholewidth = [d['width'] for d in sorted_anns]
    wholeheight = [d['height'] for d in sorted_anns]
    
    median_width = np.median(wholewidth)
    upperlimit = median_width*1.5
    lowerlimit = median_width*0.5
    
    for index, i in enumerate(sorted_anns):
        if i['width'] > upperlimit or i['width'] < lowerlimit:
            del sorted_anns[index]
            
    median_height = np.median(wholeheight)
    upperlimit = median_height*1.5
    lowerlimit = median_height*0.5
    
    for index, i in enumerate(sorted_anns):
        if i['height'] > upperlimit or i['height'] < lowerlimit:
            del sorted_anns[index]
    
    return sorted_anns

def create_quant_prompt(positive_num, negative_num):
    total_droplets = positive_num + negative_num
    positive_ratio = (positive_num / total_droplets) * 100 if total_droplets > 0 else 0
    lambdaE = - math.log(negative_num / total_droplets) if total_droplets > 0 else 0
    return f"""
Based on the image analysis results:
- Total number of droplets: {total_droplets}
- Number of positive droplets: {positive_num}
- Number of negative droplets: {negative_num}
- Positive ratio: {positive_ratio:.2f}%
- λ: {lambdaE:.2f}

Please incorporate these results into your analysis and recommendations.
"""

def analyze_ddpcr_image(url, prompt):
    client = OpenAI(
        api_key = OPENAI_API_KEY,
        base_url = OPENAI_BASE_URL
    )
    
    messages = [
        {"role": "system", "content": "You are an expert in reviewing and analyzing digital droplet PCR (ddPCR) images, specializing in checking and interpreting the quality and accuracy of ddPCR results."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Request failed: {str(e)}"

def analyze_ddpcr_image_LLM(prompt):
    client = OpenAI(
        api_key = OPENAI_API_KEY,
        base_url = OPENAI_BASE_URL
    )
    
    messages = [
        {"role": "system", "content": "You are an expert in reviewing and analyzing digital droplet PCR (ddPCR) images, specializing in checking and interpreting the quality and accuracy of ddPCR results."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Request failed: {str(e)}" 
    


    














def create_quant_prompt(positive_num, negative_num):
    total_droplets = positive_num + negative_num
    positive_ratio = (positive_num / total_droplets) * 100 if total_droplets > 0 else 0
    lambdaE = - math.log(negative_num / total_droplets) if total_droplets > 0 else 0
    return f"""
Based on the image analysis results:
- Total number of droplets: {total_droplets}
- Number of positive droplets: {positive_num}
- Number of negative droplets: {negative_num}
- Positive ratio: {positive_ratio:.2f}%
- λ: {lambdaE:.2f}

Please incorporate these results into your analysis and recommendations.
"""

def overall_evaluation(url, positive_num, negative_num):
    prompt = create_quant_prompt(positive_num, negative_num) + """
Perform an overall evaluation of the ddPCR experiment based on the provided image and data:

1. Experiment Success: Determine if the experiment was successful overall. Consider factors such as droplet formation, separation, and signal clarity.

2. Quantification Accuracy: Evaluate the reliability of the quantification based on the λ value and total droplet count. Assess whether the current concentration is optimal for accurate quantification. If λ deviates significantly higher than 0.2, suggest potential adjustments to the sample dilution or experimental setup to achieve a more ideal concentration (preferably with λ lower than 0.2) for improved accuracQy and precision in digital PCR quantification.

3. General Quality: Evaluate the overall quality of the ddPCR run, including background noise levels and signal-to-noise ratio.

Please provide a concise yet comprehensive evaluation addressing these points.No summary is needed at the end of the evaluation.
"""
    return analyze_ddpcr_image(url, prompt)

def morphological_evaluation(url):
    prompt =  """
Conduct a morphological evaluation of the ddPCR droplets based on the provided image:

1. Droplet Size Uniformity: Assess the consistency of droplet sizes across the image.

2. Droplet Shape: Evaluate the regularity of droplet shapes (e.g., circular vs. irregular).

3. Droplet Distribution: Analyze the spatial distribution of droplets in the image.

4. Droplet Integrity: Look for signs of droplet merging, splitting, or other irregularities.

5. Emulsion Stability: Assess the overall stability of the emulsion based on droplet morphology.

Please provide a detailed morphological analysis addressing these aspects.No summary is needed at the end of the evaluation.
"""
    return analyze_ddpcr_image(url, prompt)

def fluorescence_evaluation(url):
    prompt = """
Perform a fluorescence evaluation of the ddPCR results based on the provided image:

1. Signal Intensity: Assess the overall fluorescence signal intensity for positive and negative droplets.

2. Signal Separation: Evaluate the clarity of separation between positive and negative droplet signals.

3. Background Fluorescence: Analyze the level of background fluorescence and its potential impact on results.

4. Signal Consistency: Check for consistency in fluorescence intensity among positive droplets.

5. Dynamic Range: Evaluate the dynamic range of fluorescence signals observed.

6. Potential Artifacts: Identify any fluorescence artifacts or anomalies in the image.

Please provide a comprehensive fluorescence evaluation addressing these points.No summary is needed at the end of the evaluation.
"""
    return analyze_ddpcr_image(url, prompt)

def analyze_ddpcr_image(url, prompt):
    client = OpenAI(
        api_key = "sk-GlGjjAByHVYxAwON11A703D222184f4984830a680eE5B11b",
        base_url = "https://chatapi.onechats.top/v1"
    )
    
    messages = [
        {"role": "system", "content": "You are an expert in reviewing and analyzing digital droplet PCR (ddPCR) images, specializing in checking and interpreting the quality and accuracy of ddPCR results."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Request failed: {str(e)}"
    
def analyze_ddpcr_image_LLM(prompt):
    client = OpenAI(
        api_key = "sk-GlGjjAByHVYxAwON11A703D222184f4984830a680eE5B11b",
        base_url = "https://chatapi.onechats.top/v1"
    )
    
    messages = [
        {"role": "system", "content": "You are an expert in reviewing and analyzing digital droplet PCR (ddPCR) images, specializing in checking and interpreting the quality and accuracy of ddPCR results."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Request failed: {str(e)}"



def comprehensive_analysis(url, positive_num, negative_num):

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_overall = executor.submit(overall_evaluation, url, positive_num, negative_num)
        future_morphological = executor.submit(morphological_evaluation, url)
        future_fluorescence = executor.submit(fluorescence_evaluation, url)
        future_quant = executor.submit(create_quant_prompt, positive_num, negative_num)
        
        overall_result = future_overall.result()
        morphological_result = future_morphological.result()
        fluorescence_result = future_fluorescence.result()
        quant_result = future_quant.result()
    
    combined_results = f"""
Quantitative Evaluation:
{quant_result}

Overall Evaluation:
{overall_result}

Morphological Evaluation:
{morphological_result}

Fluorescence Evaluation:
{fluorescence_result}
"""
    final_analysis_prompt = f"""
Based on the following ddPCR analysis results:

{combined_results}

Please provide a comprehensive analysis and recommendations. Include:
1. A summary of key findings from each evaluation, including quantitative, overall, morphological, and fluorescence assessments.
2. An overall assessment of the ddPCR experiment quality and reliability.
3. Specific recommendations for protocol optimization or experimental improvements.
4. Identification of any potential issues or anomalies observed in the analysis.
5. Suggestions for further validation or follow-up experiments if necessary.

Ensure that your analysis is concise, direct, and based solely on the provided evaluation results and the image.
"""
    
    final_analysis = analyze_ddpcr_image(url, final_analysis_prompt)

    simplify_prompt = f"""
    Based on the following comprehensive ddPCR analysis:

    {final_analysis}

    Please provide a simplified version of the analysis as new comprehensive analysis and recommendations, focusing on:
    1. Keep the "Summary of Key Findings" section as is.
    2. Condense and simplify the content after the "Overall Assessment" section.
    3. Maintain the essential information and recommendations, but present them in a more concise manner.

    Ensure the comprehensive analysis your simplified remains informative and actionable, while being more concise and easier to read.
    Please provide a comprehensive analysis and recommendations, maintaining the same level of detail and thoroughness as the original analysis.
    The title is Comprehensive analysis and recommendations, Leave "simplified" out of the results.

    """

    simplified_analysis = analyze_ddpcr_image_LLM(simplify_prompt)

    return f"{simplified_analysis}"