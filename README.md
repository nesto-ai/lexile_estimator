# TO Lexile Calculator

순수 긴 영어 지문을 입력받아 현재 모델 Lexile과 TO Content Dev 보정 Lexile을 계산하는 모듈입니다.
LLM을 사용하지 않고 `textstat`, `spaCy`, AoA/CEFR/SUBTLEX/wordfreq feature, sklearn 모델로만 동작합니다.

## 목적

입력:

- 순수 long text 문자열

출력:

- 모델 원본 Lexile
- TO Content Dev 보정 Lexile
- `T/H/E/O/P/N` 레벨 후보
- 각 후보의 `low/mid/top` 세부 위치
- 후보 rank와 band까지의 거리
- 주요 feature 요약과 경고

## 설치

프로젝트 루트에서 실행합니다.

```bash
conda activate dev
python -m pip install -r shared/lexile_calc/requirements.txt
python -m spacy download en_core_web_sm
```

필수 리소스는 프로젝트의 `data/` 폴더에 있어야 합니다.

- `AoA_51715_words.xlsx`
- `CEFR,CEFR_J 데이터.xlsx`
- `families.xlsx`
- `SUBTLEXusExcel2007.xlsx`

학습 feature 파일은 기본적으로 다음 경로를 사용합니다.

```text
lexile_test/deep_results/deep_features.csv
```

## 사용 방식

```python
from shared.lexile_calc import LexileCalculator

text = """
The invention of the camera changed the art world. Before cameras became common,
painters were expected to create realistic images. After photography could capture
reality with great accuracy, many artists began to explore color, light, and feeling
instead of simply copying what they saw.
"""

calc = LexileCalculator.from_project("/Users/cyyoon/dev/TO")
result = calc.analyze(text)

print(result.model_lexile)
print(result.to_calibrated_lexile)
print(result.top_level, result.top_segment)
print(result.to_dict())
```

레벨을 이미 알고 있는 경우에는 `known_level`을 전달할 수 있습니다.

```python
result = calc.analyze(text, known_level="H")
```

## 스크립트 예시

이 모듈은 별도 실행기가 아니라 다른 코드에서 import해서 쓰는 계산 class입니다.
예를 들어 배치 스크립트에서는 다음처럼 사용할 수 있습니다.

```python
from pathlib import Path
from shared.lexile_calc import LexileCalculator

calc = LexileCalculator.from_project("/Users/cyyoon/dev/TO")

rows = []
for path in Path("inputs").glob("*.txt"):
    result = calc.analyze(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "file": path.name,
            "model_lexile": result.model_lexile,
            "to_calibrated_lexile": result.to_calibrated_lexile,
            "top_level": result.top_level,
            "top_segment": result.top_segment,
        }
    )
```

이미 TO 레벨을 알고 있는 데이터라면 다음처럼 route 보정을 적용합니다.

```python
result = calc.analyze(text, known_level="H")
```

## 출력 구조

```json
{
  "model_lexile": 885.3,
  "to_calibrated_lexile": 665.3,
  "model_population": "all",
  "model_name": "all_hgb",
  "top_level": "T",
  "top_segment": "mid",
  "calibration_mode": "global_offset:-220.0",
  "candidates": [
    {
      "level": "T",
      "range_low": 600.0,
      "range_high": 700.0,
      "lexile": 665.3,
      "segment": "mid",
      "position": "inside",
      "distance_to_range": 0.0,
      "midpoint_distance": 15.3,
      "rank": 1
    }
  ],
  "feature_summary": {},
  "warnings": []
}
```

## 레벨 후보 계산 방식

`known_level`이 없는 경우:

1. 전체 학습셋 기반 global model로 `model_lexile`을 계산합니다.
2. TO 교재 검증에서 관측된 global offset `-220L`을 적용해 `to_calibrated_lexile`을 만듭니다.
3. 이 보정 Lexile 하나를 `T/H/E/O/P/N` Content Dev band에 거리 기준으로 매핑합니다.
4. 별도 classifier는 사용하지 않습니다.

`known_level`이 있는 경우:

1. 해당 level의 route model을 사용합니다.
2. level별 TO Content Dev offset을 적용합니다.
3. 보정 Lexile을 전체 `T/H/E/O/P/N` 후보 band에 다시 매핑합니다.

현재 level별 offset:

| Level | Offset |
| --- | ---: |
| T | +11.0L |
| H | -102.5L |
| E | 0.0L |
| O | -173.0L |
| P | 0.0L |
| N | 0.0L |

P/N은 TO 교재 검증 데이터가 없어 아직 보정하지 않습니다.

## low / mid / top

각 레벨 band 내부를 3등분합니다.

- `low`: band 하단 1/3
- `mid`: band 중간 1/3
- `top`: band 상단 1/3

Lexile이 해당 band 밖이면 `position`은 `below` 또는 `above`로 표시됩니다. 그래도 가까운 후보 판단을 위해 `segment`는 해당 band 기준 위치로 계산합니다.

## 주의사항

- 이 값은 공식 Lexile 인증값이 아닙니다.
- TO 보정값은 TO 교재 샘플에서 경험적으로 맞춘 scale 보정입니다.
- `known_level`이 없을 때는 Lexile 보정만 사용하므로, 후보가 실제 기대 레벨보다 낮거나 높게 나올 수 있습니다.
- 짧은 글, Spanish-body text, 특수 도메인 용어가 많은 글은 경고를 확인해야 합니다.
- P/N 레벨은 학습/검증 샘플이 부족하므로 결과를 보수적으로 해석해야 합니다.
