
import gradio as gr
import torch
import torch.nn as nn
import deepchem as dc
import numpy as np
from scipy.special import expit
from transformers import AutoTokenizer, AutoModel

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

class ChemBERTaFineTuned(nn.Module):
    def __init__(self, n_tasks=12, freeze_base=True):
        super(ChemBERTaFineTuned, self).__init__()
        self.chemberta = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        if freeze_base:
            for param in self.chemberta.parameters():
                param.requires_grad = False
        hidden_size = self.chemberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_tasks)
        )
    
    def forward(self, x):
        attention_mask = (x != tokenizer.pad_token_id).long()
        with torch.no_grad():
            outputs = self.chemberta(input_ids=x.long(), attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

pytorch_model = ChemBERTaFineTuned(n_tasks=12, freeze_base=True)
model = dc.models.TorchModel(
    pytorch_model,
    loss=dc.models.losses.SigmoidCrossEntropy(),
    model_dir="./model"
)
model.restore()

task_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
              'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
              'SR-HSE', 'SR-MMP', 'SR-p53']

def predict_toxicity(smiles_input, progress=gr.Progress()):
    try:
        smiles_input = smiles_input.replace('\r\n', '\n').replace('\r', '\n')
        smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
        
        if not smiles_list:
            return "⚠️ Please enter at least one SMILES string", []
        
        progress(0, desc="Tokenizing...")
        tokens = tokenizer(smiles_list, padding='max_length', truncation=True, 
                          max_length=128, return_tensors="np")
        test_dataset = dc.data.NumpyDataset(
            X=tokens['input_ids'],
            y=np.zeros((len(smiles_list), 12)),
            w=np.ones((len(smiles_list), 12)),
            ids=smiles_list
        )
        
        progress(0.2, desc="Running Model Inference...")
        predictions = model.predict(test_dataset)
        predictions = expit(predictions)
        
        results_data = []
        high_risk_count = 0
        
        for i, smiles in enumerate(smiles_list):
            progress((i + 1) / len(smiles_list), desc=f"Processing molecule {i+1}/{len(smiles_list)}")
            max_risk = predictions[i].max()
            
            if max_risk > 0.8:
                risk_level = "🔴 HIGH RISK"
                high_risk_count += 1
            elif max_risk > 0.5:
                risk_level = "🟡 MEDIUM RISK"
            else:
                risk_level = "🟢 LOW RISK"
            
            # Formatting probabilities
            row = [smiles, risk_level, f"{max_risk:.1%}"]
            for prob in predictions[i]:
                row.append(f"{prob:.1%}")
            
            results_data.append(row)
            
        summary = f"""
### Analysis Complete
- **Total Molecules:** {len(smiles_list)}
- **High Risk:** {high_risk_count}
- **Medium/Low Risk:** {len(smiles_list) - high_risk_count}
"""
        return summary, results_data
        
    except Exception as e:
        return f"❌ Error: {str(e)}", []

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧪 ChemBERTa Toxicity Predictor")
    gr.Markdown("Predicts toxicity across 12 endpoints using fine-tuned ChemBERTa-77M-MTR model. **Test ROC-AUC: 82.8%**")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter SMILES (one per line)",
                placeholder="CCO\nCC(=O)O\nc1ccccc1",
                lines=8
            )
            analyze_btn = gr.Button("Analyze Toxicity", variant="primary")

        with gr.Column():
            summary_output = gr.Markdown(label="Analysis Summary")

    results_output = gr.DataFrame(
        label="Detailed Predictions",
        headers=["SMILES", "Overall Risk", "Max Probability"] + task_names,
        interactive=False
    )

    analyze_btn.click(
        fn=predict_toxicity,
        inputs=input_text,
        outputs=[summary_output, results_output]
    )

    gr.Examples(
        examples=[
            ["CCO"],
            ["CCO\nCC(=O)O\nc1ccccc1"],
            ["CC(C)(c1ccc(cc1)O)c2ccc(cc2)O"],
        ],
        inputs=input_text
    )

if __name__ == "__main__":
    demo.launch()
