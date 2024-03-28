from docx import Document
import os
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from googletrans import Translator
import csv

ori_path = 'C:\\Users\\DESKTOP\\Documents\\연구\\MultiModal\\raw_data'
new_path = 'C:\\Users\\DESKTOP\\Documents\\연구\\MultiModal\\new_data'

ori_data = os.listdir(ori_path)

def read_doc(data_path):
    doc = Document(data_path)
    full_text = []
    first_empty_text_found = False
    for para in doc.paragraphs:
        if first_empty_text_found or not para.text.strip():
            first_empty_text_found = True
            if para.text.strip():  # 첫 번째 비어있는 텍스트 이후에 대해서만 처리 --> 이름, creator, dictator 등 익명화
                full_text.append(para.text)
    return full_text

def translate(text, dest_lan="en"):
    translator = Translator()
    translated = translator.translate(text, dest=dest_lan)
    return translated.text.replace("=","").strip()

def process():
    with open('C:\\Users\\DESKTOP\Documents\\연구\\MultiModal\\data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['inputs', 'response'])

        for data in ori_data:
            text = read_doc(os.path.join(ori_path, data))

            full_text = ''
            for para in text:
                full_text += para + ' '  # 모든 문장을 하나의 문자열로 결합하여 저장

            # Conclusion을 기준으로 입력과 응답을 나눔
            conclusion_index = full_text.find("[Conclusion]")
            input_text = full_text[:conclusion_index].strip()
            response_text = full_text[conclusion_index + len("[Conclusion]"):].strip()

            # 번역
            translated_input = translate(input_text)
            translated_response = translate(response_text)

            writer.writerow([translated_input, translated_response])

            print(f"Processed and saved: {data}")

process()