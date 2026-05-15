# Lexile Calc Ref Data

이 폴더는 `LexileCalculator` 런타임에 필요한 lookup/model feature CSV만 포함합니다.
프로젝트 `data/*.xlsx` 원본은 재생성용 자료이며, 계산 시점에는 읽지 않습니다.

## Files

- `aoa_lookup.csv`: `lemma`, `aoa`, `freq_pm_sum`, `known`
- `cefr_lookup.csv`: `lemma`, `level`
- `academic_lemmas.csv`: `lemma`, `is_domain_academic`
- `subtlex_lookup.csv`: `word`, `lg10wf`
- `training_features.csv`: `lexile` target + numeric model feature columns

## Source

- AoA: `data/AoA_51715_words.xlsx`, `Sheet1`
- CEFR: `data/CEFR,CEFR_J 데이터.xlsx`, `CEFR` and `CEFR_J`
- Academic/domain lemmas: `data/families.xlsx`, `data`
- SUBTLEX: `data/SUBTLEXusExcel2007.xlsx`, `out1g`
- Training features: `lexile_test/deep_results/deep_features.csv`
