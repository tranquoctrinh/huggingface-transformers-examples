
import pandas as pd
import joblib
import torch
from tqdm import tqdm
import wandb
import fire

from transformers import (
    BertConfig,
    BertModel,
    RobertaConfig,
    RobertaModel,
    DebertaConfig,
    DebertaModel,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import Normalizer

import torch.nn as nn
import math
import os
from torch.autograd import Variable



class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim=768, max_seq_length=128, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_length, embedding_dim)
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/embedding_dim)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/embedding_dim)))
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x*math.sqrt(self.embedding_dim)
        seq_length = x.size(1)
        pe = Variable(self.pe[:, :seq_length], requires_grad=False).to(x.device)
        # Add the positional encoding vector to the embedding vector
        x = x + pe
        x = self.dropout(x)
        return x
    

class TabNet(nn.Module):
    def __init__(self, model_base, input_size=128):
        super().__init__()
        self.model_base = model_base
        self.input_size = input_size
        
        if model_base == "bert":
            self.config = BertConfig.from_pretrained("bert-base-uncased")
            model = BertModel(self.config)
        elif model_base == "roberta":
            self.config = RobertaConfig.from_pretrained("roberta-base")
            model = RobertaModel(self.config)
        elif model_base == "deberta":
            self.config = DebertaConfig.from_pretrained("microsoft/deberta-base")
            model = DebertaModel(self.config)
        
        self.encoder = model.encoder
        self.embedding = self.embedding = nn.Linear(1, self.config.hidden_size)
        self.pos_encoder = PositionalEncoder(embedding_dim=self.config.hidden_size, max_seq_length=self.input_size)
        
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        
    def forward(self, x, labels=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        if self.model_base == "deberta":
            x = self.encoder(x, attention_mask=torch.ones(x.shape[0], x.shape[1]).to(x.device))[0]
        else:
            x = self.encoder(x)[0]
        x = self.dropout(x).mean(dim=1)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        
        return loss, logits
    
    
def load_data(file_name):
    print("Loading data from: ", file_name)
    df = pd.read_excel(file_name, sheet_name="Sheet1")
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df


def preprocess_data(df):
    label2idx = {label: i for i, label in enumerate(df["label"].unique())}
    idx2label = {v: k for k, v in label2idx.items()}
    df["label_idx"] = df["label"].map(label2idx)
    X, y = df.values[:, :23].astype("float64"), df.values[:, -1].astype("int64")
    # nomarlize the data
    nor = Normalizer()
    nor.fit(X)
    X = nor.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Train shape: ", X_train.shape, y_train.shape)
    print("Test shape: ", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test, idx2label, nor


def train(model, train_loader, optimizer, device, global_step, wandb_key):
    model.train()
    total_loss = 0
    bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training...")
    for i, batch in bar:
        optimizer.zero_grad()
        loss, _ = model(batch[0].to(device), batch[1].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=total_loss/(i+1), step=global_step)
        global_step += 1
        if wandb_key:
            wandb.log({"train_loss": loss.item()}, step=global_step)
    
    return total_loss/(i+1), global_step


def evaluate(model, val_loader, device, idx2label, threshold=0.5):
    model.eval()
    predicts, labels = [], []
    total_loss = 0
    bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating...")
    with torch.no_grad():
        for i, batch in bar:
            loss, logits = model(batch[0].to(device), batch[1].to(device))
            total_loss += loss.item()
            predicts += (logits.sigmoid().view(-1).cpu() > threshold).int().tolist()
            labels += batch[1].view(-1).tolist()
            accuracy = accuracy_score(labels, predicts)
            f1 = f1_score(labels, predicts)
            bar.set_postfix(accuracy=accuracy, f1_score=f1)

    print(classification_report(labels, predicts, target_names=[idx2label[0], idx2label[1]]))
    return accuracy, f1, total_loss/(i+1)


def main(
    model_base="roberta", 
    data_path="./data.xlsx", 
    output_dir="./models/",
    num_epochs=100,
    batch_size=128,
    lr=1e-5,
    early_stop=10,
    wandb_key=None
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, idx2label, nor = preprocess_data(df)
    joblib.dump(nor, os.path.join(output_dir, "normalizer.joblib"))
    
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(project="Tabular Classification", name=model_base)


    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_train).float().unsqueeze(-1), torch.tensor(y_train).float().unsqueeze(-1)
        ), batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test).float().unsqueeze(-1), torch.tensor(y_test).float().unsqueeze(-1)
        ), batch_size=batch_size, shuffle=False)

    model = TabNet(model_base=model_base, input_size=X_train.shape[1]).to(device)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step, count_no_improve = 0, 0
    best_accuracy, best_epoch = 0, 0
    for epoch in range(num_epochs):
        print("--------------------Epoch: {}--------------------".format(epoch+1))
        train_loss_epoch, global_step = train(model, train_loader, optimizer, device, global_step, wandb_key)
        accuracy, f1, valid_loss_epoch = evaluate(model, test_loader, device, idx2label)
        if wandb_key:
            wandb.log({
                "train_loss_epoch": train_loss_epoch, 
                "valid_loss_epoch": valid_loss_epoch,
                "valid_accuracy": accuracy, 
                "valid_f1_score": f1
            }, step=global_step)
        
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch+1
            torch.save(model.state_dict(), os.path.join(output_dir, f"stress_{model.model_base}_model.pt"))
            count_no_improve = 0
        else:
            count_no_improve += 1
        
        print("Best accuracy: {:.4f} at epoch {}".format(best_accuracy, best_epoch))
        print("------------------------------------------------")
        if count_no_improve == early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("Training finished!")



if __name__ == "__main__":
    fire.Fire(main)





"""
python stress_classification.py \
    --model_base roberta \
    --data_path "./data.xlsx" \
    --output_dir models \
    --num_epochs 100 \
    --batch_size 128 \
    --lr 1e-5 \
    --early_stop 6 \
    --wandb_key "<your wandb key>"
"""
