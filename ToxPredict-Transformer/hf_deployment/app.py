
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

def predict_toxicity(smiles_input):
    try:
        smiles_input = smiles_input.replace('\r\n', '\n').replace('\r', '\n')
        smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
        
        if not smiles_list:
            return "⚠️ Please enter at least one SMILES string"
        
        tokens = tokenizer(smiles_list, padding='max_length', truncation=True, 
                          max_length=128, return_tensors="np")
        test_dataset = dc.data.NumpyDataset(
            X=tokens['input_ids'],
            y=np.zeros((len(smiles_list), 12)),
            w=np.ones((len(smiles_list), 12)),
            ids=smiles_list
        )
        
        predictions = model.predict(test_dataset)
        predictions = expit(predictions)
        
        results_text = f"📊 Analyzed {len(smiles_list)} molecule(s)\n\n"
        
        for i, smiles in enumerate(smiles_list):
            max_risk = predictions[i].max()
            
            if max_risk > 0.8:
                risk_level = "🔴 HIGH RISK"
            elif max_risk > 0.5:
                risk_level = "🟡 MEDIUM RISK"
            else:
                risk_level = "🟢 LOW RISK"
            
            results_text += f"\n{'='*70}\n"
            results_text += f"MOLECULE #{i+1}\n"
            results_text += f"SMILES: {smiles}\n"
            results_text += f"Overall Risk: {risk_level} (Max: {max_risk:.1%})\n"
            results_text += f"{'='*70}\n\n"
            results_text += "Toxicity Predictions by Endpoint:\n"
            results_text += "-" * 70 + "\n"
            
            for j, task in enumerate(task_names):
                prob = predictions[i, j]
                bar_length = int(prob * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                warning = " ⚠️" if prob > 0.7 else ""
                results_text += f"{task:15s}: {prob:6.1%} |{bar}|{warning}\n"
            
            high_risk_tasks = [task_names[j] for j in range(12) if predictions[i, j] > 0.7]
            if high_risk_tasks:
                results_text += f"\n⚠️  HIGH RISK ENDPOINTS: {', '.join(high_risk_tasks)}\n"
            results_text += "\n"
        
        return results_text
    except Exception as e:
        return f"❌ Error: {str(e)}"

demo = gr.Interface(
    fn=predict_toxicity,
    inputs=gr.Textbox(label="Enter SMILES (one per line)", 
                     placeholder="CCO\nCC(=O)O\nc1ccccc1", lines=8),
    outputs=gr.Textbox(label="Toxicity Predictions", lines=30, show_copy_button=True),
    title="🧪 ChemBERTa Toxicity Predictor",
    description="Predicts toxicity across 12 endpoints using fine-tuned ChemBERTa-77M-MTR model. **Test ROC-AUC: 82.8%**",
    examples=[
        ["CCO"],
        ["CCO\nCC(=O)O\nc1ccccc1"],
        ["CC(C)(c1ccc(cc1)O)c2ccc(cc2)O"],
    ],
    theme=gr.themes.Soft()
)

demo.launch()
