# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from transformers import T5Tokenizer,T5ForConditionalGeneration, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
import csv

# Load and preprocess the dataset
df = pd.read_excel('../ArithOpsTrain.xlsx')
header = df.iloc[0][1:]
df = df.drop('Table 1',axis=1)
df = df[1:]
df.columns = header

# Display the first few rows of the dataset
df.head()

# Analyze and visualize the length distribution of descriptions, questions, and their combined lengths
length_desc = df.Description.apply(lambda x:len(x))
length_q = df.Question.apply(lambda x:len(x))
length_overall = np.asarray(length_desc) + np.asarray(length_q)
perc_90 = np.quantile(length_overall,0.9)
print('percentile 90th is of length', perc_90)

# Plot histograms
plt.figure(figsize=(10,8))
plt.subplot(1,3,1)
plt.hist(length_desc)

plt.subplot(1,3,2)
plt.hist(length_q)

plt.subplot(1,3,3)
plt.hist(length_overall)
plt.show()

# Split the dataset into training and testing sets
# X, val_df = train_test_split(df, test_size=0.15, random_state=123)
train_df, test_df = train_test_split(df, test_size=0.05, random_state=123)

train_df.shape, test_df.shape#, val_df.shape

# Define a custom dataset class for preprocessing and encoding
class ArithmeticDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer: T5Tokenizer, 
        text_max_token_len: int = 512,
        eqn_max_token_len: int = 10
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.eqn_max_token_len = eqn_max_token_len

    def __len__(self):
         # Get the length of the dataset
        return len(self.data)
    
    def __getitem__(self, idx: int):
         # Retrieve a specific item from the dataset
        data_row = self.data.iloc[idx]

        text = [data_row['Question'], data_row['Description']]
        eqn = data_row['Equation']

        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt')

        eqn_encoding = self.tokenizer(
            eqn,
            max_length=self.eqn_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt')
        
        labels = eqn_encoding['input_ids']
        labels[labels == 0] = -100
        
        return dict(
            text=text,
            eqn=eqn,
            text_input_ids=text_encoding['input_ids'].flatten(),
            text_attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=eqn_encoding['attention_mask'].flatten()
        )



# Define a PyTorch Lightning DataModule class
class ArithmeticDataModule(pl.LightningDataModule):
     # Initialization method
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        text_max_token_len: int = 512,
        eqn_max_token_len: int = 10
        ):

        super().__init__()

        self.train_df = train_df
        self.test_df = test_df

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.eqn_max_token_len = eqn_max_token_len
    
    def setup(self, stage=None):
        # Set up the training and testing datasets
        self.train_dataset = ArithmeticDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.eqn_max_token_len
        )

        self.test_dataset = ArithmeticDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.eqn_max_token_len
        )
    # Create a DataLoader for training
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
    # Create a DataLoader for validation
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
     # Create a DataLoader for testing
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )


# Model setup
MODEL_NAME = 't5-base'

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

additional_tkns = ['number0','number1','number2']

tokenizer.add_tokens(additional_tkns,special_tokens=True)

# Model training parameters
N_EPOCHS = 10
BATCH_SIZE = 8

# Instantiate the DataModule
data_module = ArithmeticDataModule(train_df, test_df, tokenizer, batch_size=BATCH_SIZE)


# # MODEL

# Model definition
class ArithmeticModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        self.accuracy = torchmetrics.Accuracy()
    
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):

        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, output = self(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
            )
        
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, output = self(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
            )
        
        # actual_op = labels
        # pred_op = output
        # acc = (labels == output).sum()/labels.shape[0]
        # print(output)
        # self.accuracy(output, labels)
        
        self.log('val_loss', loss, prog_bar=True, logger=True)
        # self.log('val_acc', output, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, output = self(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
            )
        
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-4)

# Instantiate the model
model = ArithmeticModel()

# Set up TensorBoard logging
%load_ext tensorboard
%tensorboard  --logdir ./lightning_logs

# Model training setup with checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best_checkpoint",
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

logger = TensorBoardLogger('lightning_logs', name='Arithmetic-eqn')

trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=N_EPOCHS,
    accelerator='gpu',
    devices=[2],
    enable_progress_bar=True
)

# Train the model
trainer.fit(model, data_module)

# Load the trained model
trained_model = ArithmeticModel.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path
)

# Freeze the model parameters
trained_model.freeze()



# Function to compute predictions on test data
def compute_eqn(text_question, text_description):
    text_encoding = tokenizer(
        text_question,
        text_description,
        max_length=396,
        padding='max_length',
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    
    generated_ids = trained_model.model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=10,
        num_beams=1,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )
    
    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]
    
    return "".join(preds)


# sample_row = test_df.iloc[0]

# eqn = sample_row['Equation']

eqn_preds = compute_eqn('how many apples are in the basket ?', 'number0 red apples and number1 green apples are in the basket .')


# eqn_preds


dict_num = {'number0':0, 'number1':1, 'number2':2,'number3':3,'number4':4,'number5':5,'number6':6,'number7':7,'number8':8,
            'number9':9}

# Function to replace tokens with numbers
def replace_with_numbers(eqn, nums):
    nums = nums.split()
    # new_eqn = eqn
    for tok in eqn.split():
        if tok in dict_num.keys():
            # print(tok, nums[dict_num[tok]])
            try:
                eqn = eqn.replace(tok, nums[dict_num[tok]])
            except:
                print(dict_num, tok, nums)
                continue
    eqn = eqn.strip()
    return eqn
    


replace_with_numbers('+ number0 number1','7 2')

# Python3 program to evaluate a prefix expression.
def isdigit(ch):
    for i in ch:
        if(ord(i) < 48 or ord(i) > 57) and i !='.':
            return False
    return True

 # Function to evaluate a prefix expression 
def evaluatePrefix(expr):
    try:
        S = []
        expr = expr.split()[::-1]
        # print(expr)
        for i in expr:
            # print(i)
            if isdigit(i):
                S.append(float(i))
            else:
                a = S.pop()
                b = S.pop()
                if i == '+':
                    S.append(a+b)
                elif i == '-':
                    S.append(a-b)
                elif i == '*':
                    S.append(a*b)
                elif i == '/':
                    S.append(a/b)
            # print(S, i)
        return S[-1]
    
    except :
        return np.nan
            

# 
evaluatePrefix('+ 7 2')

# Function to compute predictions on the entire test dataset
def compute_on_test_data(x):
    
    pred_eqns = compute_eqn(x['Question'], x['Description'])
    
    final_eqn = replace_with_numbers(pred_eqns, x['Input Numbers'])
    
    ans = evaluatePrefix(final_eqn)
    
    if ans == int(ans):
        return int(ans)
    else:
        return np.around(ans, decimals=2)
    
    # test_df['pred_eqns'] = test_df.apply(lambda x: compute_eqn(x['Question'], x['Description']), axis=1)
    
    # test_df['final_eqn'] = test_df.apply(lambda x: replace_with_numbers(x['pred_eqns'], x['Input Numbers']), axis=1)
    
    # test_df['output_final'] = test_df.final_eqn.apply(lambda x: evaluatePrefix(x))
    
    # return ans if ans - int(ans) else int(ans)

# Example usage on a sample DataFrame
# # FINAL CODE

# %%
def predict(test_df, save=False):
    test_df = test_df.apply(lambda x:compute_on_test_data(x), axis=1).to_numpy()
    test_df = [[int(i)] if  i == int(i) else [np.around(i,decimals=2)] for i in test_df]
    if save:
        with open("abcd.csv","a") as f:
            writer = csv.writer(f)
            for row in test_df:
                writer.writerow(row)
    return test_df
    # test_df['output_pred'].to_csv('abc.csv',index=False)


sample_df = test_df.copy()
pred_y = predict(sample_df, save=True)
print(pred_y)

