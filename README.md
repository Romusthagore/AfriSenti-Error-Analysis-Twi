# AfriSenti Error Analysis on Twi

**Author:** Romuald Ahomagnon | AIMS-Cameroon / IVADO

---

## Results

- **Accuracy:** 64.6% on test set
- **Main issue:** Neutral class ignored (recall = 1%)

### Per-Class Metrics

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Positive | 0.64 | 0.83 | 0.72 |
| Neutral | 0.67 | **0.01** | 0.03 |
| Negative | 0.65 | 0.68 | 0.66 |

---

## Error Analysis (135 errors / 388)

| Confusion | Count |
|-----------|-------|
| negative → positive | 44 |
| neutral → positive | 39 |
| positive → negative | 33 |
| neutral → negative | 19 |

---

## Pattern Analysis

| Pattern | Frequency |
|---------|-----------|
| Emojis (😂, 😁) | 43.0% |
| Repeated chars (ooooo, paaa) | 31.1% |
| ! and ? | 0.0% |

### Sample Error
> *"5mins biaa na scam nam mu 😁"* (negative → positive)

The 😁 emoji overrides the negative text.

---

## Conclusion

The model's limitation is **data-related** (class imbalance + misleading emojis), not architectural.

### For Future Work 
- Collect 30-40% neutral examples
- Convert emojis to text carefully
- Keep ! and ? (they work well)

---

## Files

- `analysis.ipynb` - Main notebook
- `error_analysis.py` - Analysis functions

---

**License:** MIT
