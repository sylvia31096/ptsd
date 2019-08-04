import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms
import time

from PIL import Image
from random import randrange



import mlflow.sklearn


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller
            than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, num_labels=23):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

from pytorch_pretrained_bert import BertConfig
​
config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
​
num_labels = 23
model = BertForSequenceClassification(num_labels)
​
# Convert inputs to PyTorch tensors
#tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(zz)])
​
#logits = model(tokens_tensor)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            sentiment_corrects = 0

            # Iterate over data.
            for inputs, sentiment in dataloaders_dict[phase]:
                # print("ok till here")
                # inputs = inputs
                # print(len(inputs),type(inputs),inputs)
                # inputs = torch.from_numpy(np.array(inputs)).to(device)
                inputs = inputs.to(device)

                sentiment = sentiment.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print(inputs)
                    outputs = model(inputs)

                    outputs = F.softmax(outputs, dim=1)

                    loss = criterion(outputs, torch.max(sentiment.float(), 1)[1])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])

            epoch_loss = running_loss / dataset_sizes[phase]

            sentiment_acc = sentiment_corrects.double() / dataset_sizes[phase]

            print('{} total loss: {:.4f} '.format(phase, epoch_loss))
            print('{} sentiment_acc: {:.4f}'.format(
                phase, sentiment_acc))

            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'bert_model_test.pth')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_loss)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def get_criterions():
    return ["A1", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "E1", "E2", "E3",
     "E4", "E5", "E6", "F1", "G1"]

def prepare_data(data):
    """
        Return a prepared dataframe
        input : Dataframe with expected schema

    """
    dataframe_targets = data.groupby("transcript_id").sum()[get_criterions()]
    data_frame_text_fields = data.groupby("transcript_id")["text"].agg(lambda col: ''.join(col))
    data_frame_text_fields = data_frame_text_fields.to_frame()
    data_frame_text_fields.reset_index(level=0, inplace=True)
    dataframe_targets.reset_index(level=0, inplace=True)
    data_frame_merged = pd.merge(dataframe_targets, data_frame_text_fields, on="transcript_id")
    criterions = ["A1", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "E1", "E2",
                  "E3", "E4", "E5", "E6", "F1", "G1"]

    processed_df = data_frame_merged
    for criterion in criterions:
        processed_df[criterion] = processed_df[criterion].apply(lambda x: 1 if x >= 0.5 else 0)

    return processed_df

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/'PTSD_data_Colton_Alexis.csv'")

    data = pd.read_csv(wine_path, nrows=865)  # limiting the number of rows to only the text that has already been annotated by Alexis
    # ,delimiter='\t',encoding='utf-8', nrows=10000)#, encoding ="ISO-8859-1" , names=DATASET_COLUMNS, nrows=10000)

    # filter on CLIENT character - this line might not be needed
    data = data[data['character'] == 'CLIENT']

    prepared_data = prepare_data(data)


    train, test = train_test_split(prepared_data, random_state=42, test_size=0.10, shuffle=True)
    X_train = train.text
    X_test = test.text


    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():

        lrlast = .001
        lrmain = .00001
        optim1 = optim.Adam(
            [
                {"params": model.bert.parameters(), "lr": lrmain},
                {"params": model.classifier.parameters(), "lr": lrlast},

            ])

        # optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)
        # Observe that all parameters are being optimized
        optimizer_ft = optim1
        criterion = nn.CrossEntropyLoss()

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

        model.to(device)
        model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                                num_epochs=10)
#Define the model and the configuration
        from pytorch_pretrained_bert import BertConfig

        config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                            num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        num_labels = 23
        model = BertForSequenceClassification(num_labels)
#Load the previously saved model.
        model.load_state_dict(torch.load('bert_model_test.pth'))
        device = torch.device('cuda')
        model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        test_preds = np.zeros((2, 23))
        tests = torch.utils.data.TensorDataset(torch.tensor(X_final_test))
        test_loader = torch.utils.data.DataLoader(tests, batch_size=2, shuffle=False)

        for i, (x_batch,) in enumerate(test_loader):
            t_preds = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
            test_preds[i:i + 2] = t_preds[:, 0:23].detach().cpu().squeeze().numpy()
        # test_preds = t_preds[:,0:23].detach().cpu().squeeze.numpy()
        t_p = pd.DataFrame(test_preds)

# Run our test list through the model and write the results to the mlflow log metric
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x) / np.sum(np.exp(x), axis=0)


        test_preds1 = t_p.apply(lambda x: softmax(x), axis=1)
        test_preds1

        mlflow.log_metric("test_preds", test_preds1)