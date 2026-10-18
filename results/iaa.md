# Inter-annotator agreement — diacritization subset

Items scored: **150** of 150 in the subset  
Slots compared: **7290**  
Bootstrap: 5000 cluster resamples over utterances; permutation test: 2000 shuffles; seed 20260727

## Overall

| Stratum | Utt. | Slots | Agree | κ [95% CI] | α | DER | DER −final | Word agree |
|---|---|---|---|---|---|---|---|---|
| All | 150 | 7290 | 93.2% | 0.898 [0.886, 0.909] | 0.898 | 6.8% | 7.7% | 77.6% |

- Krippendorff's α 95% CI: [0.886, 0.909]
- Permutation test against κ = 0: **p < 0.001**
- Words compared: 1739; words diverging orthographically (excluded from the slot statistics): 142

## By subdialect

| Stratum | Utt. | Slots | Agree | κ [95% CI] | α | DER | DER −final | Word agree |
|---|---|---|---|---|---|---|---|---|
| Pal./Urban | 30 | 2237 | 93.1% | 0.898 [0.881, 0.917] | 0.898 | 6.9% | 7.2% | 76.6% |
| Pal./Rural | 30 | 1484 | 95.4% | 0.930 [0.913, 0.944] | 0.930 | 4.6% | 5.6% | 83.7% |
| Jordanian | 30 | 1008 | 91.3% | 0.867 [0.823, 0.901] | 0.867 | 8.7% | 10.1% | 72.8% |
| Lebanese | 30 | 1078 | 93.2% | 0.896 [0.863, 0.927] | 0.896 | 6.8% | 8.3% | 77.4% |
| Syrian | 30 | 1483 | 92.4% | 0.886 [0.861, 0.912] | 0.886 | 7.6% | 8.3% | 76.3% |

## By gender

| Stratum | Utt. | Slots | Agree | κ [95% CI] | α | DER | DER −final | Word agree |
|---|---|---|---|---|---|---|---|---|
| F | 59 | 2983 | 92.9% | 0.894 [0.876, 0.912] | 0.894 | 7.1% | 7.8% | 75.8% |
| M | 91 | 4307 | 93.4% | 0.900 [0.886, 0.914] | 0.900 | 6.6% | 7.6% | 78.8% |

## By source

| Stratum | Utt. | Slots | Agree | κ [95% CI] | α | DER | DER −final | Word agree |
|---|---|---|---|---|---|---|---|---|
| casablanca | 70 | 2937 | 93.8% | 0.907 [0.888, 0.923] | 0.907 | 6.2% | 7.1% | 79.4% |
| langmedia | 80 | 4353 | 92.7% | 0.891 [0.877, 0.906] | 0.891 | 7.3% | 8.0% | 76.3% |

## By first-pass provenance

| Stratum | Utt. | Slots | Agree | κ [95% CI] | α | DER | DER −final | Word agree |
|---|---|---|---|---|---|---|---|---|
| silver approved in round 1 | 150 | 7290 | 93.2% | 0.898 [0.886, 0.909] | 0.898 | 6.8% | 7.7% | 77.6% |

## Annotator variety vs utterance variety

The annotators are speakers of Palestinian Arabic. If a Palestinian annotator cannot reliably diacritize another variety's phonology, agreement should be measurably lower off Palestinian material.

| Stratum | Utt. | Slots | Agree | κ [95% CI] | α | DER | DER −final | Word agree |
|---|---|---|---|---|---|---|---|---|
| Palestinian (Urban + Rural) | 60 | 3721 | 94.0% | 0.911 [0.898, 0.924] | 0.911 | 6.0% | 6.6% | 79.5% |
| Non-Palestinian (Jor/Leb/Syr) | 90 | 3569 | 92.4% | 0.883 [0.865, 0.901] | 0.883 | 7.6% | 8.8% | 75.6% |

Difference in κ (Palestinian − non-Palestinian): **0.027**, 95% CI [0.006, 0.050]

An interval covering zero would mean the data show no variety-dependent drop in reliability at this sample size. It is not evidence of equivalence either way; state the interval, not just the sign.

## Most frequent disagreements

| Label pair | Count |
|---|---|
| i ↔ ∅ | 146 |
| a ↔ ∅ | 105 |
| a ↔ i | 62 |
| ~ ↔ ∅ | 59 |
| u ↔ ∅ | 22 |
| i ↔ u | 21 |
| a ↔ ~a | 14 |
| i ↔ ~i | 11 |
| ~i ↔ ∅ | 11 |
| ~a ↔ ∅ | 9 |
| ~a ↔ ~i | 6 |
| a ↔ u | 5 |

