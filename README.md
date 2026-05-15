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
- 각 후보의 `low/core/top` 세부 위치
- 후보 rank와 band까지의 거리
- 주요 feature 요약과 경고

## 설치

프로젝트 루트에서 실행합니다.

```bash
conda activate dev
python -m pip install -r shared/lexile_calc/requirements.txt
python -m spacy download en_core_web_sm
```

런타임 필수 리소스는 모듈 내부의 `ref_data/`에 CSV로 포함되어 있습니다.
따라서 계산 시점에는 프로젝트 `data/*.xlsx` 원본이 필요하지 않습니다.

- `shared/lexile_calc/ref_data/aoa_lookup.csv`
- `shared/lexile_calc/ref_data/cefr_lookup.csv`
- `shared/lexile_calc/ref_data/academic_lemmas.csv`
- `shared/lexile_calc/ref_data/subtlex_lookup.csv`
- `shared/lexile_calc/ref_data/training_features.csv`

각 CSV는 기존 Excel/실험 파일에서 실제 계산에 쓰는 컬럼만 추출한 것입니다.
원본 Excel은 재생성/검증용 자료이며, 모듈 실행 경로의 기본 의존성은 아닙니다.

필요하면 `ref_data_dir` 또는 `training_features`를 넘겨 다른 lookup/model feature 파일로 교체할 수 있습니다.

## 사용 방식

```python
from shared.lexile_calc import LexileCalculator

text = """
The invention of the camera changed the art world. Before cameras became common,
painters were expected to create realistic images. After photography could capture
reality with great accuracy, many artists began to explore color, light, and feeling
instead of simply copying what they saw.
"""

calc = LexileCalculator.from_project()
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

calc = LexileCalculator.from_project()

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
  "top_level": "T",
  "top_segment": "core",
  "candidates": [
    {
      "level": "T",
      "range_low": 600.0,
      "range_high": 700.0,
      "lexile": 665.3,
      "segment": "core",
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

## low / core / top

각 레벨 band를 단순 3등분하지 않습니다. 현재 supported 구간의 MAE가 대략 29-33L이고 P90는 54-72L입니다.
P90를 boundary로 쓰면 100L band 전체가 boundary가 되므로, 인접 레벨로 흔들릴 수 있는 구간은 1 MAE 수준인 `30L`로 둡니다.

- `low`: 해당 band에 들어왔지만 하단 boundary에서 30L 이내라 아래 레벨 가능성이 있는 구간
- `core`: 양쪽 boundary에서 모두 30L보다 멀어 해당 레벨 중심부로 보는 구간
- `top`: 해당 band에 들어왔지만 상단 boundary에서 30L 이내라 위 레벨 가능성이 있는 구간

Lexile이 해당 band 밖이면 `position`은 `below` 또는 `above`로 표시됩니다. 이 경우 `segment`는 후보 해석을 위해 `below -> low`, `above -> top`으로 표시합니다.

## 주의사항

- 이 값은 공식 Lexile 인증값이 아닙니다.
- TO 보정값은 TO 교재 샘플에서 경험적으로 맞춘 scale 보정입니다.
- `known_level`이 없을 때는 Lexile 보정만 사용하므로, 후보가 실제 기대 레벨보다 낮거나 높게 나올 수 있습니다.
- 짧은 글, Spanish-body text, 특수 도메인 용어가 많은 글은 경고를 확인해야 합니다.
- P/N 레벨은 학습/검증 샘플이 부족하므로 결과를 보수적으로 해석해야 합니다.
