# Run by typing python3 main.py

## **IMPORTANT:** only collaborators on the project where you run
## this can access this web server!

"""
    Bonus points if you want to have internship at AI Camp
    1. How can we save what user built? And if we can save them, like allow them to publish, can we load the saved results back on the home page? 
    2. Can you add a button for each generated item at the frontend to just allow that item to be added to the story that the user is building? 
    3. What other features you'd like to develop to help AI write better with a user? 
    4. How to speed up the model run? Quantize the model? Using a GPU to run the model? 
"""

# import basics
import os

# import stuff for our web server
from flask import Flask, flash, request, redirect, url_for, render_template, Markup
from flask import send_from_directory
from flask import jsonify
from utils import get_base_url, allowed_file, and_syntax

# import stuff for our models
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertPreTrainedModel
from torch.nn.utils.rnn import pad_sequence

'''
Coding center code - comment out the following 4 lines of code when ready for production
'''
# load up the model into memory
# you will need to have all your trained model in the app/ directory.
# ai = aitextgen(to_gpu=False, model=r"EleutherAI/gpt-neo-125M")

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
# port = 12339
# base_url = get_base_url(port)
app = Flask(__name__)


class BertClassifier(nn.Module):

    def __init__(self, bert: BertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,

            labels=None):
        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        cls_output = outputs[1] # batch, hidden
        cls_output = self.classifier(cls_output) # batch, 6
        cls_output = torch.sigmoid(cls_output)
        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            loss = criterion(cls_output, labels)
        return loss, cls_output
    

device = torch.device('cpu')
bert_model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

#PATH TO BE CHANGED
output_model = './saved_model_rachel'
model = BertClassifier(BertModel.from_pretrained(bert_model_name), 6).to(device)
checkpoint = torch.load(output_model, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
model.eval()
texts = []


'''
Deployment code - uncomment the following line of code when ready for production
'''
app = Flask(__name__)

@app.route('/', methods = ['GET'])
# @app.route(base_url, methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/result', methods = ['POST'])
# @app.route(base_url + '/result', methods = ['POST'])
def result(message=""):
    message = request.form['message']
    text = tokenizer.encode(message, add_special_tokens=True)
    if len(text) > 120:
        text = text[:119] + [tokenizer.sep_token_id]
    texts.append(torch.LongTensor(text))
    x = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    mask = (x != tokenizer.pad_token_id).float().to(device)
    with torch.no_grad():
        _, outputs = model(x, attention_mask=mask)
    outputs = outputs.cpu().numpy()

    results = {}
    for ind, col in enumerate(columns):
        results[f"{col}"] = f"{outputs[0][ind]}"
    output = dict(sorted(results.items(), key = lambda item: item[1], reverse = True))
    result = ""
    for key, value in output.items():
        val = float(value)
        result += key + ': ' + str(round(val, 2)) + '\n'
    return render_template('index.html', result=result)


if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
    website_url = 'cocalc1.ai-camp.org'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    app.run(host = '0.0.0.0', port=port, debug=True)
    import sys; sys.exit(0)

    '''
    scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)
