import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLabel, 
                           QLineEdit, QPushButton, QTextEdit, QProgressBar)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import requests
import io
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import markdown
from concurrent.futures import ThreadPoolExecutor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ultralytics import YOLO
import logging
from functions import *
from config import *

class AnalysisThread(QThread):
    finished = pyqtSignal(tuple)
    progress = pyqtSignal(int)
    
    def __init__(self, images, parent=None):
        super().__init__(parent)
        self.images = images
        
    def run(self):
        try:
            self.progress.emit(10)
            
            FITC_image = np.array(self.images[0])
            TD_image = np.array(self.images[1])
            RGB_image = np.array(self.images[2])
            
            model = YOLO(YOLO_MODEL)
            sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
            sam.to(device=DEVICE)
            mask_generator = SamAutomaticMaskGenerator(sam)
            
            positive_num = 0
            negative_num = 0
            annotations_to_remove = set()

            TD_image_array = cv2.cvtColor(TD_image, cv2.COLOR_RGB2BGR)
            hsv_targetImage = cv2.cvtColor(TD_image_array, cv2.COLOR_BGR2HSV)

            self.progress.emit(20)

            clahe = cv2.createCLAHE(tileGridSize=(10,10))
            hsv_targetImage[:, :, 2] = clahe.apply(hsv_targetImage[:, :, 2])
            hsv_targetImage = cv2.merge((hsv_targetImage[:, :, 0], hsv_targetImage[:, :, 1], hsv_targetImage[:, :, 2]))
            img_compensated = cv2.cvtColor(hsv_targetImage, cv2.COLOR_HSV2BGR)

            annotations = mask_generator.generate(img_compensated)
            self.progress.emit(30)
            
            if len(annotations) == 0:
                self.finished.emit((TD_image, RGB_image, pd.DataFrame(), 0, 0))
                return

            imgSatu = hsv_targetImage[:, :, 2]
            
            sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)
            annotations_to_remove = findFalsyAnnotationsIndex(sorted_anns=sorted_anns, removeArrayList=annotations_to_remove)
            sorted_anns = removeFalsyAnnotations(removeArrayList=annotations_to_remove, sorted_anns=sorted_anns)

            self.progress.emit(40)

            batch_images = []
            batch_info = []
            df = pd.DataFrame(columns=['class', 'width', 'height', 'confidence'])
            
            for annYolo in sorted_anns:
                uint8_mask = (annYolo['segmentation'] * 255).astype(np.uint8)
                _, uint8_mask = cv2.threshold(uint8_mask, 0, 255, cv2.THRESH_BINARY)
                mean_value = cv2.mean(imgSatu, mask=uint8_mask)[0]

                bbox = annYolo['bbox']
                x, y, w, h = bbox
                annYolo['width'] = w
                annYolo['height'] = h
                annYolo['mean_value'] = mean_value

            sorted_anns = removeOutOfSizeLimitAnnotations(sorted_anns=sorted_anns)

            for annYolo in sorted_anns:
                x, y, w, h = annYolo['bbox']
                image_bbox = FITC_image[int(y):int(y+h), int(x):int(x+w)]
                
                batch_images.append(image_bbox)
                batch_info.append({
                    'annYolo': annYolo,
                    'bbox': annYolo['bbox'],
                    'mean_value': annYolo['mean_value']
                })

            results = model(batch_images)
            self.progress.emit(50)

            for result, info in zip(results, batch_info):
                probs = result.probs
                classes = str(probs.top1)

                if classes == '1':
                    positive_num += 1
                    class_label = 'Positive'
                else:
                    negative_num += 1
                    class_label = 'Negative'

                TD_image = show_bbox(TD_image, info['bbox'], "TD")
                RGB_image = show_bbox(RGB_image, info['bbox'], classes)

                new_row = pd.DataFrame({
                    'class': [class_label],
                    'width': [info['annYolo']['width']],
                    'height': [info['annYolo']['height']],
                    'confidence': [probs.top1conf.item()]
                })
                df = pd.concat([df, new_row], ignore_index=True)

            self.progress.emit(60)
            self.finished.emit((TD_image, RGB_image, df, positive_num, negative_num))
            
        except Exception as e:
            print(f"Error in analysis thread: {str(e)}")

class ComprehensiveAnalysisThread(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, url, positive_num, negative_num, parent=None):
        super().__init__(parent)
        self.url = url
        self.positive_num = positive_num
        self.negative_num = negative_num
        
    def run(self):
        try:
            result = comprehensive_analysis(self.url, self.positive_num, self.negative_num)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(f"Analysis failed: {str(e)}")

class LLMThread(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, url, prompt, parent=None):
        super().__init__(parent)
        self.url = url
        self.prompt = prompt
        
    def run(self):
        try:
            response = analyze_ddpcr_image(self.url, self.prompt)
            self.finished.emit(response)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'LLM_dPCR'
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.negative_num = 0
        self.positive_num = 0
        self.analysis_thread = None
        self.llm_thread = None
        self.comprehensive_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
            }
            QLabel {
                color: #333333;
                font-size: 20px;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-size: 20px;
            }
            QPushButton:!focus {
                background-color: #4CAF50;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 3px; 
            }
            QProgressBar {
                border: 1px solid #cccccc;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0.5px;
            }
        """)

        grid_layout = QGridLayout()

        self.label1 = QLabel('Fluorescence Image URL:')
        self.text1 = QLineEdit('https://my.microsoftpersonalcontent.com/personal/bdeebb56c930dee1/_layouts/15/download.aspx?UniqueId=55c93317-4aad-4271-b201-130d1d856a87&Translate=false&tempauth=v1e.eyJzaXRlaWQiOiJlYTFkNDUzYi1hYTg5LTQ1M2QtOTdiZS02NzExZGM2NTUyYjkiLCJhcHBpZCI6IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDA0ODE3MTBhNCIsImF1ZCI6IjAwMDAwMDAzLTAwMDAtMGZmMS1jZTAwLTAwMDAwMDAwMDAwMC9teS5taWNyb3NvZnRwZXJzb25hbGNvbnRlbnQuY29tQDkxODgwNDBkLTZjNjctNGM1Yi1iMTEyLTM2YTMwNGI2NmRhZCIsImV4cCI6IjE3Mzc2NDA4MjUifQ.-TXVFpufOr20gU2exXgibZnT9PmX5pPgjIjXM65vWGLp9TgX1ZFcFlF6175CAMhQqKP5bXe2eRaq3zepfbth7JPqrWtOOwt4bNo8EaBDKE7nDp0kS8MVzRgxwnKNjGdkOE0AnbWNGBXaAW9K79tvtYQsJhmEdWBEr6nULeCc9e2egFnCPmwzqzsfYKShrIiRugj3ihooDroPPjL59QSeASLb9oMsIPMCweJfeTnP1kY30r-xzq41H-uhTQxC4TFPTcy8ZKxubvJYc3te35R4HbSJV2TRytBYnP8prk4TdEpI2-EZn4K8FATohkaaXRcqNgJx6o6mpbWSRj3-SlGp8wAVWDNhnarPtVjjoNJKgSDcHFqRh2AgOeZknshKdl7hglhM9_rFyeMbg6KT7CfFu5Vs_rpdeKNasdjx5oJeC28.TApR-qLwTtSw-EDnD1VEcTOGxs74FFfhFkowCf78E10&ApiVersion=2.0&AVOverride=1')
        self.label2 = QLabel('Bright Field Image URL:')
        self.text2 = QLineEdit('https://my.microsoftpersonalcontent.com/personal/bdeebb56c930dee1/_layouts/15/download.aspx?UniqueId=11f892fc-0703-4ea2-9fa7-f9c718cddc20&Translate=false&tempauth=v1e.eyJzaXRlaWQiOiJlYTFkNDUzYi1hYTg5LTQ1M2QtOTdiZS02NzExZGM2NTUyYjkiLCJhcHBpZCI6IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDA0ODE3MTBhNCIsImF1ZCI6IjAwMDAwMDAzLTAwMDAtMGZmMS1jZTAwLTAwMDAwMDAwMDAwMC9teS5taWNyb3NvZnRwZXJzb25hbGNvbnRlbnQuY29tQDkxODgwNDBkLTZjNjctNGM1Yi1iMTEyLTM2YTMwNGI2NmRhZCIsImV4cCI6IjE3Mzc2NDExMDEifQ.UEG8XGv9tFnQHPh1_ZkRg7oKFzavCqsBr6fNLYts6iy4ef4ji8HjPIXE3sFnex8si2X3FOWPNSFHYA60JItSotutIU4ddXCx06gkqWBxD5UcWIsEHqnk5kPCs3GyOCkhw6dykaXM486vclzxexZdCG6_ltPKjqcKCG7hYqbmS0hWmNO2UA4-RwN7kUyxfdmr5QhAd3nNyx_3eRkV3RqJBXbwTN1_0ZpYv5JqdNBEl0xQBKClpnlwmAfSrHQlKBvuSqyS_nKJr1Clhzb8u8aOrV-M2A7yc1BZ8CBBNEQpr67tx8-7vVwuJ1wI1eksWfr8muXKyovSpUeCmY960PkB6WCsKFY6pXVxySDeWIhR9XZYIGr6nJoNijzRqbgSxFNoLi8vJlcY07MdgvUiS7R4W4ogThmDrtyTCFwhElpqIOE.sHgBTEQc-IGyxmvzcakD8uBbZIGoJym7w_RpLcNnre8&ApiVersion=2.0&AVOverride=1')
        self.label3 = QLabel('Merged Image URL:')
        self.text3 = QLineEdit('https://my.microsoftpersonalcontent.com/personal/bdeebb56c930dee1/_layouts/15/download.aspx?UniqueId=4f96f77a-9b18-4004-b195-beeef41b732a&Translate=false&tempauth=v1e.eyJzaXRlaWQiOiJlYTFkNDUzYi1hYTg5LTQ1M2QtOTdiZS02NzExZGM2NTUyYjkiLCJhcHBpZCI6IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDA0ODE3MTBhNCIsImF1ZCI6IjAwMDAwMDAzLTAwMDAtMGZmMS1jZTAwLTAwMDAwMDAwMDAwMC9teS5taWNyb3NvZnRwZXJzb25hbGNvbnRlbnQuY29tQDkxODgwNDBkLTZjNjctNGM1Yi1iMTEyLTM2YTMwNGI2NmRhZCIsImV4cCI6IjE3Mzc2NDEwNzEifQ.v84Wkz42Rx4R6AOcgvd0T_bxqmEgSJm3og1Fro7S-mnlY5PcQexq3QDP9Er4rInYmVJEUDIy7q3MeeH7Ug1m_lZr1nKyesWMfxkXDuY1utXUUJQEVC4rXCOQk79RNwGWv09Mr28USfNQ6Rk2Op2sXdGlLn5P5C3K0r9XXPTcp5BPtgu93up53KWqi3BkxOUEp3uDh7u3blJRFjjvmC-JradLBLL1Eew6f5mI1lbl84MkNJnoGohoq0ts6XpIvmfV3VH0dO3zpvmOyAkkEy6LsFiksv3D3xnMLFb4SDke5H7PeWng0oILNkCvq6VFRRyislU_OvrW94FFsiUnruEjsT0UDGbgoMtH0uIK3N-KVqarwtc1_I0s-LMDAOUJtoSfvSWIoS4CzsJDXQBE2XEfOweMyAVb_4Vnbe9f1rV-eHY.mTQawAs-1K1olPaO5CO95kxsCI8pC3lTAyBSHhmBI54&ApiVersion=2.0&AVOverride=1')

        grid_layout.addWidget(self.label1, 0, 0)
        grid_layout.addWidget(self.text1, 0, 1)
        grid_layout.addWidget(self.label2, 1, 0)
        grid_layout.addWidget(self.text2, 1, 1)
        grid_layout.addWidget(self.label3, 2, 0)
        grid_layout.addWidget(self.text3, 2, 1)

        self.button = QPushButton('Analyze')
        self.button.clicked.connect(self.on_click)
        grid_layout.addWidget(self.button, 3, 0, 1, 2)

        self.image_label1 = QLabel()
        self.image_label2 = QLabel()
        self.image_label3 = QLabel()
        self.label_image1 = QLabel('Bright field label:')
        self.label_image2 = QLabel('Merged field label:')
        self.label_image3 = QLabel('Scatter plot:')
        self.set_image_size(self.image_label1, 300, 300)
        self.set_image_size(self.image_label2, 300, 300)
        self.set_image_size(self.image_label3, 300, 300)

        grid_layout.addWidget(self.label_image1, 4, 0)
        grid_layout.addWidget(self.image_label1, 4, 1)
        grid_layout.addWidget(self.label_image2, 5, 0)
        grid_layout.addWidget(self.image_label2, 5, 1)
        grid_layout.addWidget(self.label_image3, 6, 0)
        grid_layout.addWidget(self.image_label3, 6, 1)

        self.progress_bar = QProgressBar()
        grid_layout.addWidget(self.progress_bar, 7, 0, 1, 2)

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        grid_layout.addWidget(self.text_output, 0, 2, 8, 1)

        grid_layout.setColumnMinimumWidth(2, 1000)

        self.llm_prompt_label = QLabel('Ask LLM:')
        self.llm_prompt_button = QPushButton('Ask')
        self.llm_prompt_button.clicked.connect(self.on_llm_prompt)
        grid_layout.addWidget(self.llm_prompt_label, 0, 3)
        grid_layout.addWidget(self.llm_prompt_button, 0, 4)

        self.llm_input = QLineEdit()
        grid_layout.addWidget(self.llm_input, 1, 3, 1, 2)
        self.llm_input.setMinimumWidth(400)

        self.llm_output = QTextEdit()
        self.llm_output.setReadOnly(True)
        grid_layout.addWidget(self.llm_output, 2, 3, 6, 2)

        self.setLayout(grid_layout)
        self.setGeometry(300, 300, 1600, 800)
        self.show()

    def set_image_size(self, image_label, width, height):
        image_label.setFixedSize(width, height)
        image_label.setScaledContents(True)

    def download_single_image(self, url):
        attempts = 0
        while attempts < 5:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                image_data = io.BytesIO(response.content)
                image = Image.open(image_data)
                return image
            except RequestException as e:
                attempts += 1
                if attempts == 5:
                    print(f"Failed to download image: {url}, Maximum attempts reached")
                    return None
                time.sleep(1)

    def download_image(self):
        self.url1 = self.text1.text()
        self.url2 = self.text2.text()
        self.url3 = self.text3.text()

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self.download_single_image, url) 
                for url in [self.url1, self.url2, self.url3]
            ]
        
        images = [future.result() for future in futures]
        return images

    def on_click(self):
        self.button.setEnabled(False)
        self.images = self.download_image()
        self.text_output.setHtml("Performing image analysis, please wait...")
        
        if all(image_data is not None for image_data in self.images):
            self.analysis_thread = AnalysisThread(self.images)
            self.analysis_thread.progress.connect(self.progress_bar.setValue)
            self.analysis_thread.finished.connect(self.handle_analysis_complete)
            self.analysis_thread.start()
        else:
            self.text_output.setText("Failed to download images")
            self.button.setEnabled(True)

    def handle_analysis_complete(self, results):
        TD_image, RGB_image, df, positive_num, negative_num = results
        self.display_results(TD_image, RGB_image, df)
        
        if positive_num > 0 and negative_num > 0:
            self.text_output.setHtml("Performing comprehensive analysis, please wait...")
            self.comprehensive_thread = ComprehensiveAnalysisThread(self.url3, positive_num, negative_num)
            self.comprehensive_thread.progress.connect(self.progress_bar.setValue)
            self.comprehensive_thread.finished.connect(self.handle_comprehensive_complete)
            self.comprehensive_thread.start()
        else:
            self.button.setEnabled(True)

    def handle_comprehensive_complete(self, result):
        try:
            markdown_text = result
            html_text = markdown.markdown(markdown_text)
            self.text_output.setHtml(html_text)
        except Exception as e:
            self.text_output.setHtml(f"Error processing analysis results: {str(e)}")
        finally:
            self.button.setEnabled(True)

    def display_results(self, TD_image, RGB_image, df):
        try:
            qimage1 = QImage(TD_image.data, TD_image.shape[1], TD_image.shape[0], TD_image.strides[0], QImage.Format_RGB888)
            pixmap1 = QPixmap.fromImage(qimage1)
            self.image_label1.setPixmap(pixmap1.scaled(300, 300, Qt.KeepAspectRatio))

            height, width = RGB_image.shape[:2]
            bytes_per_line = 3 * width
            qimage2 = QImage(RGB_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap2 = QPixmap.fromImage(qimage2)
            self.image_label2.setPixmap(pixmap2.scaled(300, 300, Qt.KeepAspectRatio))

            self.create_scatter_plot(df)
            
        except Exception as e:
            print(f"Error in display_results: {str(e)}")

    def create_scatter_plot(self, df):
        plt.figure(figsize=(6, 6), facecolor='white')
        df['diameter'] = (df['width'] + df['height']) / 2 / 81 * 200
        df['x_value'] = np.where(df['class'] == 'Positive', -df['diameter'], df['diameter'])

        plt.scatter(df[df['class'] == 'Positive']['x_value'], 
                    df[df['class'] == 'Positive']['confidence'], 
                    color='red', alpha=0.5, label='Positive', s=70)
        plt.scatter(df[df['class'] == 'Negative']['x_value'], 
                    df[df['class'] == 'Negative']['confidence'], 
                    color='blue', alpha=0.5, label='Negative', s=70)

        plt.xlabel('Droplet diameter(a.u.)', fontsize=26)
        plt.ylabel('Confidence of classification', fontsize=26)

        max_diameter = math.ceil(max(df['diameter']) / 10) * 10 + 10
        plt.xticks([-max_diameter, -max_diameter/2, 0, max_diameter/2, max_diameter],
                   [f'{max_diameter}', f'{max_diameter//2}', '0', f'{max_diameter//2}', f'{max_diameter}'],
                   fontsize=18)
        plt.yticks(fontsize=18)

        plt.axvline(x=0, color='k', linestyle='--')
        plt.xlim(-max_diameter, max_diameter)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=14)
        plt.tight_layout()
        plt.grid(False)

        plt.gca().spines['top'].set_color('black')
        plt.gca().spines['right'].set_color('black')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['left'].set_color('black')

        plt.savefig("scatter_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        scatter_pixmap = QPixmap("scatter_plot.png")
        self.image_label3.setPixmap(scatter_pixmap.scaled(300, 300, Qt.KeepAspectRatio))

    def on_llm_prompt(self):
        user_input = self.llm_input.text()
        combined_prompt = f"""
        Analysis Results:
        {self.text_output.toPlainText()}
        
        User Question:
        {user_input}
        
        Please provide insights and answers based on the analysis results and the user's question.
        """
        
        self.llm_prompt_button.setEnabled(False)
        self.llm_thread = LLMThread(self.url3, combined_prompt)
        self.llm_thread.finished.connect(self.handle_llm_complete)
        self.llm_thread.start()

    def handle_llm_complete(self, response):
        markdown_text = response
        html_text = markdown.markdown(markdown_text)
        self.llm_output.setText(html_text)
        self.llm_prompt_button.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 