
import huggingface_hub
huggingface_hub.login()

dataPath = "C:\\Users\\DESKTOP\\Documents\\연구\\MultiModal\\dataset\\"
print('hello')

import pandas as pd
import json
import jsonlines
from datasets import Dataset

# 데이터 경로 설정 및 불러오기
datasetName = "data.csv"
jsonFileName = "data.jsonl"

def csv_to_json(csv_file_path, json_file_path):
    # CSV 파일을 DataFrame으로 읽기
    df = pd.read_csv(csv_file_path)

    # JSON 파일로 저장
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        # 각 행을 JSON으로 변환하여 바로 파일에 쓰기
        for index, row in df.iterrows():
            data = {'inputs': row['inputs'], 'response': row['response']}
            json.dump(data, json_file, ensure_ascii=False)
            json_file.write('\n')  # 각 행마다 줄바꿈

# CSV 파일 경로와 JSON 파일 경로 설정
csv_file_path = dataPath + datasetName
json_file_path = dataPath + jsonFileName

# 함수 호출
csv_to_json(csv_file_path, json_file_path)

#json 파일 데이터를 허깅페이스에 올리기 전 포맷 변환
indataset = []
with jsonlines.open(json_file_path) as f:
    for line in f.iter():
      #허깅페이스는 INST를 읽게 되어있음
      indataset.append(f'<s>[INST] {line["inputs"]} [/INST] {line["response"]} </s>')
    #   indataset.append(f'<s>### Instruction: \n{line["inputs"]} \n\n### Response: \n{line["response"]}</s>')

# 데이터셋 확인
print('데이터셋 확인')
print(indataset[:5])

# 데이터셋 생성 및 저장
indataset = Dataset.from_dict({"text": indataset})
indataset.save_to_disk(dataPath)

# 데이터셋 info 확인
print('데이터셋 info 확인')
print(indataset)

indataset.push_to_hub("heeseongE/customllm") #허깅페이스의 내 데이터 디렉토리로 푸시

