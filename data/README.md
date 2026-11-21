# Data

## SentiMix Hinglish (SemEval-2020 Task 9)

| Field | Value |
|---|---|
| Paper | Patwa et al. (2020), *SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets*, SemEval-2020 (pp. 774–790). arXiv: https://arxiv.org/abs/2008.04277 |
| Source used | Zenodo record 3974927, https://zenodo.org/record/3974927 (DOI 10.5281/zenodo.3974927, v1, 2020-08-06). The overview paper cites this as the official location. |
| File | `Semeval_2020_task9_data.zip`, 2,807,728 bytes, md5 `4d93553c213b3a966b6d55b7a7c03be7` (verified on download) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the Zenodo record. The README in the zip asks users to cite the overview paper. |
| Other locations | CodaLab competition 20654, which needs a login. The task site `ritual-uh.github.io/sentimix2020` returns 404 as of 2026-11. HF mirror `RTT1/SentiMix` is **not used**: its card is empty, its license claim has no stated basis, and its files differ in size from the official release. |

**Not committed.** The files contain full tweet text. CC BY 4.0 allows redistribution, but X/Twitter developer terms generally restrict sharing tweet text to tweet IDs (not verified against the current terms). Keep `data/raw/` out of git and re-download it with the steps below.

### Download

```bash
curl -L -o data/raw/Semeval_2020_task9_data.zip "https://zenodo.org/records/3974927/files/Semeval_2020_task9_data.zip?download=1"
md5sum data/raw/Semeval_2020_task9_data.zip        # expect 4d93553c213b3a966b6d55b7a7c03be7
cd data/raw && unzip Semeval_2020_task9_data.zip
```

`src/hinglishkey/data.py` reads from `data/raw/Semeval_2020_task9_data/Hinglish/`. The files are:

| Split | File | Labels |
|---|---|---|
| train | `Hinglish_train_14k_split_conll.txt` | in the `meta` line |
| dev | `Hinglish_dev_3k_split_conll.txt` | in the `meta` line |
| test | `Hinglish_test_unalbelled_conll_updated.txt` ("unalbelled" is the official spelling) | `Hinglish_test_labels.txt` (CSV `Uid,Sentiment`, joined by uid) |

Format: one token per line (`token<TAB>lang_tag`, tags `Hin`/`Eng`/`O`/`EMT`). Each tweet starts with `meta<TAB>uid<TAB>label` and tweets are separated by blank lines. `load_split()` returns `(tweet_id, text, label)` with `text = " ".join(tokens)`. The language tags are dropped, because the core pipeline uses no language ID.

### Split sizes and label distribution

Computed by `python -m hinglishkey.data`. They match Table 1 of the overview paper exactly, and `tests/test_data.py` asserts this.

| Split | Total | negative | neutral | positive |
|---|---|---|---|---|
| train | 14,000 | 4,102 (29.3%) | 5,264 (37.6%) | 4,634 (33.1%) |
| dev | 3,000 | 890 (29.7%) | 1,128 (37.6%) | 982 (32.7%) |
| test | 3,000 | 900 (30.0%) | 1,100 (36.7%) | 1,000 (33.3%) |

The tweet IDs are unique within each split and the three splits are disjoint (tested).

### Known data issues

1. **Test file is double-encoded (mojibake).** The official test file stores UTF-8 text that was decoded as cp1252 and saved again, so 😅 appears as `ðŸ˜…`. Train and dev are clean: 2,639 and 558 tweets contain real emoji, against 0 in raw test. `data.fix_mojibake` repairs the test split on load and fixes 2,802 tokens. One triple-encoded token (`Ã¢Â€Â¢`) is left as it is. The repair changes no token in train or dev.
2. **Tokenisation splits URLs.** URLs come out as separate tokens, e.g. `https // t . co / hBg7zS0viy`. A URL regex run on the joined text will not match them, so `tokenize.py` has to handle this form.
3. **Noise.** Some tweets are not Hinglish; for example, the first train tweet is in Czech. Nothing is filtered, because the official splits are used as released.

## Gold files (`data/gold/`)

- `candidates.tsv`, `variants_gold.tsv`: E1 variant groups. Code proposes candidates; only **humans** assign gold labels.
- `e3_queries.tsv`: E3 known-item queries. The respelled queries are written by a human. Code may only create the empty template.
