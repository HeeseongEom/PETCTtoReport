# PETCTtoReport 프로젝트 소개

PETCTtoReport 프로젝트는 PET CT 이미지로부터 Report를 생성하는 작업의 시작 단계에서, Report 데이터의 정제 과정을 기록한 것입니다.

## 주요 파일 설명

- **finetuning.py**: `LLAMA-2-7b` 및 `gemma-2b` 모델을 파인튜닝하는 코드입니다.
- **hub_preprocess.py**: 데이터를 허깅페이스(Hugging Face)에 업로드하기 전, 포맷 변환을 통해 데이터를 가공하는 코드입니다.
- **inference.py**: 실제로 추론을 진행하는 코드입니다.
