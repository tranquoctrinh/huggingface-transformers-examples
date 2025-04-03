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
    
    
class DataManager:
    def __init__(self, data_path, batch_size=128, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.normalizer = None
        self.idx2label = None
        
    def load_data(self):
        print("Loading data from: ", self.data_path)
        df = pd.read_excel(self.data_path, sheet_name="Sheet1")
        df.drop(columns=["Unnamed: 0"], inplace=True)
        return df
    
    def preprocess_data(self, df):
        label2idx = {label: i for i, label in enumerate(df["label"].unique())}
        self.idx2label = {v: k for k, v in label2idx.items()}
        df["label_idx"] = df["label"].map(label2idx)
        X, y = df.values[:, :23].astype("float64"), df.values[:, -1].astype("int64")
        
        self.normalizer = Normalizer()
        self.normalizer.fit(X)
        X = self.normalizer.transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        print("Train shape: ", X_train.shape, y_train.shape)
        print("Test shape: ", X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
    
    def create_data_loaders(self, X_train, y_train, X_test, y_test):
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_train).float().unsqueeze(-1), 
                torch.tensor(y_train).float().unsqueeze(-1)
            ), 
            batch_size=self.batch_size, 
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_test).float().unsqueeze(-1), 
                torch.tensor(y_test).float().unsqueeze(-1)
            ), 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def save_normalizer(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.normalizer, os.path.join(output_dir, "normalizer.joblib"))


class Trainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        device, 
        idx2label,
        output_dir="./models/",
        wandb_key=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.idx2label = idx2label
        self.output_dir = output_dir
        self.wandb_key = wandb_key
        self.global_step = 0
        
        if self.wandb_key:
            wandb.login(key=self.wandb_key)
            wandb.init(project="Tabular Classification", name=self.model.model_base)
            
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training...")
        for i, batch in bar:
            self.optimizer.zero_grad()
            loss, _ = self.model(batch[0].to(self.device), batch[1].to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            bar.set_postfix(loss=total_loss/(i+1), step=self.global_step)
            self.global_step += 1
            if self.wandb_key:
                wandb.log({"train_loss": loss.item()}, step=self.global_step)
        
        return total_loss/(i+1)
    
    def evaluate(self, threshold=0.5):
        self.model.eval()
        predicts, labels = [], []
        total_loss = 0
        bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Evaluating...")
        with torch.no_grad():
            for i, batch in bar:
                loss, logits = self.model(batch[0].to(self.device), batch[1].to(self.device))
                total_loss += loss.item()
                predicts += (logits.sigmoid().view(-1).cpu() > threshold).int().tolist()
                labels += batch[1].view(-1).tolist()
                accuracy = accuracy_score(labels, predicts)
                f1 = f1_score(labels, predicts)
                bar.set_postfix(accuracy=accuracy, f1_score=f1)

        print(classification_report(labels, predicts, target_names=[self.idx2label[0], self.idx2label[1]]))
        return accuracy, f1, total_loss/(i+1)
    
    def train(self, num_epochs=100, early_stop=10):
        os.makedirs(self.output_dir, exist_ok=True)
        count_no_improve = 0
        best_accuracy, best_epoch = 0, 0
        
        for epoch in range(num_epochs):
            print("--------------------Epoch: {}--------------------".format(epoch+1))
            train_loss_epoch = self.train_epoch()
            accuracy, f1, valid_loss_epoch = self.evaluate()
            
            if self.wandb_key:
                wandb.log({
                    "train_loss_epoch": train_loss_epoch, 
                    "valid_loss_epoch": valid_loss_epoch,
                    "valid_accuracy": accuracy, 
                    "valid_f1_score": f1
                }, step=self.global_step)
            
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch+1
                torch.save(
                    self.model.state_dict(), 
                    os.path.join(self.output_dir, f"stress_{self.model.model_base}_model.pt")
                )
                count_no_improve = 0
            else:
                count_no_improve += 1
            
            print("Best accuracy: {:.4f} at epoch {}".format(best_accuracy, best_epoch))
            print("------------------------------------------------")
            if count_no_improve == early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("Training finished!")
        return best_accuracy, best_epoch


class TabularClassifier:
    def __init__(
        self,
        model_base="roberta",
        data_path="./data.xlsx",
        output_dir="./models/",
        batch_size=128,
        lr=1e-5,
        wandb_key=None
    ):
        self.model_base = model_base
        self.data_path = data_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.lr = lr
        self.wandb_key = wandb_key
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.data_manager = None
        self.model = None
        self.trainer = None
        
    def setup(self):
        # Setup data
        self.data_manager = DataManager(self.data_path, batch_size=self.batch_size)
        df = self.data_manager.load_data()
        X_train, X_test, y_train, y_test = self.data_manager.preprocess_data(df)
        train_loader, test_loader = self.data_manager.create_data_loaders(X_train, y_train, X_test, y_test)
        self.data_manager.save_normalizer(self.output_dir)
        
        # Setup model
        self.model = TabNet(model_base=self.model_base, input_size=X_train.shape[1]).to(self.device)
        print("Number of parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Setup trainer
        self.trainer = Trainer(
            model=self.model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=self.device,
            idx2label=self.data_manager.idx2label,
            output_dir=self.output_dir,
            wandb_key=self.wandb_key
        )
        
        return self
    
    def train(self, num_epochs=100, early_stop=10):
        if self.trainer is None:
            self.setup()
        return self.trainer.train(num_epochs=num_epochs, early_stop=early_stop)


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
    classifier = TabularClassifier(
        model_base=model_base,
        data_path=data_path,
        output_dir=output_dir,
        batch_size=batch_size,
        lr=lr,
        wandb_key=wandb_key
    )
    
    classifier.setup()
    classifier.train(num_epochs=num_epochs, early_stop=early_stop)


if __name__ == "__main__":
    fire.Fire(main)