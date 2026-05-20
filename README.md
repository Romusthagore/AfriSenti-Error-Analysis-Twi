# AfriSenti Error Analysis on Twi Language

**Author:** Romuald Ahomagnon  
**Affiliation:** AIMS-Cameroon / IVADO (upcoming stay)  
**Date:** May 2026

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Methodology](#methodology)
3. [Results](#results)
4. [Error Analysis](#error-analysis)
5. [Pattern Analysis](#pattern-analysis)
6. [Conclusion](#conclusion)
7. [Files](#files)

---

## Project Overview

This project performs a comprehensive error analysis of the **AfroXLM-R** model fine-tuned for sentiment classification on the **Twi language** using the AfriSenti dataset.

### Key Objectives
- Establish a baseline for Twi sentiment analysis
- Identify systematic error patterns
- Understand why the model fails on certain classes
- Derive lessons for future work on Fon/Goun

---

## Methodology

### Model
- **Architecture:** AfroXLM-R base (Davlan/afro-xlmr-base)
- **Task:** 3-class sentiment classification (positive, neutral, negative)
- **Training epochs:** 10
- **Batch size:** 32
- **Learning rate:** 1e-5
- **Seed:** 2

### Dataset (Twi - AfriSenti)
| Split | Size |
|-------|------|
| Train | 3,481 |
| Validation | 388 |
| Test | 949 |

---

## Results

### Overall Performance
Test Accuracy: 64.6%

text

### Per-Class Metrics (Test Set)
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Positive | 0.64 | 0.83 | 0.72 | 450 |
| Neutral | 0.67 | **0.01** | **0.03** | 146 |
| Negative | 0.65 | 0.68 | 0.66 | 353 |

### Key Finding
**The neutral class is practically invisible to the model** (recall = 1%).

---

## Error Analysis

### Validation Set Errors
- **Total errors:** 135 / 388 (34.8%)

### Confusion Summary
| Confusion | Count |
|-----------|-------|
| negative → positive | 44 |
| neutral → positive | 39 |
| positive → negative | 33 |
| neutral → negative | 19 |

### Interpretation
Neutral tweets are absorbed by both positive and negative classes. The model never predicts "neutral".

---

## Pattern Analysis

### Error Patterns (135 errors)
| Pattern | Frequency | Percentage |
|---------|-----------|------------|
| Emojis | 58 / 135 | **43.0%** |
| Repeated characters | 42 / 135 | **31.1%** |
| Exclamation (!) | 0 / 135 | 0.0% |
| Question (?) | 0 / 135 | 0.0% |

### Sample Errors

**Error 1:**
True: negative | Pred: positive
Text: amanfuo girls na omo bl3 😂😂😂

text

**Error 2:**
True: negative | Pred: positive
Text: 5mins biaa na scam nam mu 😁

text

**Error 3:**
True: negative | Pred: positive
Text: saaa mo president no wabɔn paaa apeetor

text

### Key Observations
1. **Smiling emojis** (😂, 😁) dominate the model's prediction, pushing it toward "positive" even when the text is negative.
2. **Repeated characters** ("ooooo", "paaa") are ignored, causing the model to miss emphasis and sarcasm.
3. **Exclamation and question marks** cause 0% of errors → these patterns work well.

---

## Conclusion

### Main Finding
The model's limitation is **data-related, not architectural**.

### Identified Issues
| Issue | Impact |
|-------|--------|
| Underrepresented neutral class | Model never predicts neutral |
| Emojis override text sentiment | 43% of errors contain emojis |
| Repeated characters ignored | 31% of errors contain emphasis markers |

### What Works Well
- Exclamation marks (!)
- Question marks (?)

### Lessons for Future Work (Fon/Goun)
| Action | Priority |
|--------|----------|
| Collect 30-40% neutral examples | High |
| Convert emojis to text carefully | High |
| Normalize repeated characters | Medium |
| Keep ! and ? (they work well) | Low |

---

## Files

| File | Description |
|------|-------------|
| `analysis.ipynb` | Complete analysis notebook |
| `error_analysis.py` | Reusable analysis functions |

---

## Author

**Romuald Ahomagnon**  
Data Science Student at AIMS-Cameroon  
Upcoming research stay at IVADO (Université de Montréal)

**Contact:** romuald.ahomagnon@aims-cameroon.org

---

## License

MIT
