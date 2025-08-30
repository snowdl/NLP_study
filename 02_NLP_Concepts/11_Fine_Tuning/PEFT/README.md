# PEFT Experiments (LoRA)

This README collects minimal LoRA fine-tuning experiments on BANKING77.

---

## 📘 RoBERTa + LoRA
- **Model:** `roberta-base`
- **Epochs:** 3  
- **Config:** r=8, α=32, dropout=0.05, targets=["query","key","value","dense"]  
- **Result:**

eval_accuracy ≈ 0.93
eval_macro_f1 ≈ 0.93

- Much stronger accuracy and F1.

👉 Detailed notes: [lora_roberta_min.md](./lora_roberta_min.md)

---

## 📗 DistilBERT + LoRA
- **Model:** `distilbert-base-uncased`
- **Epochs:** 1 (test run)  
- **Config:** r=8, α=32, dropout=0.05, targets=["q_lin","k_lin","v_lin","out_lin"]  
- **Result:**

eval_accuracy ≈ 0.75
eval_macro_f1 ≈ 0.73

- Much faster, but lower accuracy compared to RoBERTa.  
- With 3 epochs, accuracy expected ~0.82–0.85.

👉 Detailed notes: [LoRA_DistilBERT_min.md](./LoRA_DistilBERT_min.md)

---

## 🔍 Comparison

| Model            | Epochs | Accuracy | Macro F1 | Notes                |
|------------------|--------|----------|----------|----------------------|
| RoBERTa + LoRA   | 3      | ~0.93    | ~0.93    | Strong baseline      |
| DistilBERT + LoRA| 1      | ~0.75    | ~0.73    | Very fast, underfits |

---

## 📝 Notes
- **RoBERTa** is recommended when accuracy is priority.  
- **DistilBERT** is good for speed / resource-limited setups.  
- Both benefit from LoRA, but base model capacity dominates results.
