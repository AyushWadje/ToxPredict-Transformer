MODEL CONFIGURATION:
- Base Model: ChemBERTa-77M-MTR (pre-trained)
- Fine-tuning: Frozen base + trained classifier head
- Dataset: Tox21 (12 toxicity endpoints)
- Training samples: 6264
- Validation samples: 783
- Test samples: 784

PERFORMANCE METRICS:
- Train ROC-AUC: 0.8822
- Valid ROC-AUC: 0.8127
- Test ROC-AUC: 0.8278

TOXICITY ENDPOINTS:
NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
