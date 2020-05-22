# Summ abstractive and extractive

## Install instructions

Insatall on your Ubuntu server and serve API with gunicorn:

####  Step 1. Clone repo
```
git clone https://github.com/GraphGrailAi/summ-abs-dev.git
```
####  Step 2. Install Anaconda 5.2.0
Skip step if you have it already
```
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
bash Anaconda3-5.2.0-Linux-x86_64.sh
```
####  Step 3. Get punkt tokenizer for NLTK
In your terminal launch python interactive shell and invoke:
```
viktor@chrono-ml:~$ python
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import nltk
>>> nltk.download('punkt')
```
####  Step 4. Download summarization model to models folder
cd to summ-abs-dev/models/ and download from Dropbox, about 2GB space. Rename file to model_step_154000.pt (no ?dl=0)
```
cd summ-abs-dev/models/
wget https://dl.dropbox.com/s/54blwnq3vnfgpzv/model_step_154000.pt?dl=0
mv model_step_154000.pt?dl=0 model_step_154000.pt
```
####  Step 5. Install gunicorn
Warning: be careful you install it for Anaconda python. Allow port for inerence: 5005
```
pip install gunicorn
sudo ufw allow 5005
```
####  Step 6. Configure supervisor to run summ service as daemon.
You can test service without supervisor directly from terminal:
```
gunicorn --log-level debug --workers 1 --timeout 730 --pythonpath /home/viktor/anaconda3/bin --access-logfile=access.log --error-logfile=error.log --capture-output --bind 0.0.0.0:5005 wsgi:app
```
But better to use supervisor:

Warning: be careful you install it for Anaconda python. Allow port for inference: 5005
```
cd /etc/supervisor/conf.d
sudo wget https://raw.githubusercontent.com/GraphGrailAi/summ-abs-dev/master/supervisor/supervisord_ze.conf
sudo supervisorctl reread
sudo supervisorctl update
sudo /etc/init.d/supervisor restart
```
supervisor config is the following:
```
[program:summgunguy]
directory=/home/viktor/summ-abs-dev/src/
user=viktor
command=/usr/bin/gunicorn3 --log-level debug --workers 1 --timeout 1030 --bind 0.0.0.0:5005 wsgi:app
PYTHONPATH='/home/viktor/anaconda3/bin'
autostart=true
autorestart=true
stopwaitsecs=1
startsecs=5
priority=99
stderr_logfile=/var/log/test_summgunguy.err.log
stdout_logfile=/var/log/test_summgunguy.out.log
```

Now go to supervisorctl and start service:
```
sudo supervisorctl
summgunguy start
```
####  Step 7. Inference: send your text to service and get summarization back
In you browser navigate to following link with API. 
```
http://35.202.164.44:5005/get_summary?raw_text=”Ai text to rewrite”

```
Wait about 15 sec. Paste your text in raw_text= between quotes

For example:
```
http://35.202.164.44:5005/get_summary?raw_text=%22%E2%80%98The%20Robot,%20The%20Dentist%20and%20The%20Pyramid%E2%80%99%20follows%20a%20group%20of%20research%20engineers%20and%20scientists,%20some%20from%20the%20University%20of%20Leeds,%20who%20accepted%20a%20challenge%20to%20build%20a%20robot%20capable%20of%20exploring%20the%20pyramid.%22
```

You will get a result in JSON format:
```
{
  "text_full": "\"\u2018The Robot, The Dentist and The Pyramid\u2019 follows a group of research engineers and scientists, some from the University of Leeds, who accepted a challenge to build a robot capable of exploring the pyramid.\"", 
  "text_summary": "A group of research engineers and scientists from the university of leeds accepted a challenge to build a robot capable of exploring the pyramid. The dentist and the pyramid are two of the most successful robotic vehicles in the world"
}
```
Get your result in text_summary field.

That's it.




**This code is for EMNLP 2019 paper [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)**
Results on CNN/DailyMail (20/8/2019):


<table class="tg">
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">ROUGE-1</th>
    <th class="tg-0pky">ROUGE-2</th>
    <th class="tg-0pky">ROUGE-L</th>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="4">Extractive</td>
  </tr>
  <tr>
    <td class="tg-0pky">TransformerExt</td>
    <td class="tg-0pky">40.90</td>
    <td class="tg-0pky">18.02</td>
    <td class="tg-0pky">37.17</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt</td>
    <td class="tg-0pky">43.23</td>
    <td class="tg-0pky">20.24</td>
    <td class="tg-0pky">39.63</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt (large)</td>
    <td class="tg-0pky">43.85</td>
    <td class="tg-0pky">20.34</td>
    <td class="tg-0pky">39.90</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="4">Abstractive</td>
  </tr>
  <tr>
    <td class="tg-0lax">TransformerAbs</td>
    <td class="tg-0lax">40.21</td>
    <td class="tg-0lax">17.76</td>
    <td class="tg-0lax">37.09</td>
  </tr>
  <tr>
    <td class="tg-0lax">BertSumAbs</td>
    <td class="tg-0lax">41.72</td>
    <td class="tg-0lax">19.39</td>
    <td class="tg-0lax">38.76</td>
  </tr>
  <tr>
    <td class="tg-0lax">BertSumExtAbs</td>
    <td class="tg-0lax">42.13</td>
    <td class="tg-0lax">19.60</td>
    <td class="tg-0lax">39.18</td>
  </tr>
</table>

**Python version**: This code is in Python3.6

**Package Requirements**: torch==1.1.0 pytorch_transformers tensorboardX multiprocess pyrouge

**Updates**: For encoding a text longer than 512 tokens, for example 800. Set max_pos to 800 during both preprocessing and training.


Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)

## Trained Models
[CNN/DM Extractive](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)

[CNN/DM Abstractive](https://drive.google.com/open?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr)

[XSum](https://drive.google.com/open?id=1H50fClyTkNprWJNh10HWdGEdDdQIkzsI)

## Data Preparation For XSum
[Pre-processed data](https://drive.google.com/open?id=1BWBN1coTWGBqrWoOfRc5dhojPHhatbYs)


## Data Preparation For CNN/Dailymail
### Option 1: download the processed data

[Pre-processed data](https://drive.google.com/open?id=1DN7ClZCCXsk2KegmC6t4ClBwtAf5galI)

unzip the zipfile and put all `.pt` files into `bert_data`

### Option 2: process the data yourself

#### Step 1 Download Stories
Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all  `.story` files in one directory (e.g. `../raw_stories`)

####  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2017-06-09` directory. 

####  Step 3. Sentence Splitting and Tokenization

```
python preprocess.py -mode tokenize -raw_path RAW_PATH -save_path TOKENIZED_PATH
```

* `RAW_PATH` is the directory containing story files (`../raw_stories`), `JSON_PATH` is the target directory to save the generated json files (`../merged_stories_tokenized`)


####  Step 4. Format to Simpler Json Files
 
```
python preprocess.py -mode format_to_lines -raw_path RAW_PATH -save_path JSON_PATH -n_cpus 1 -use_bert_basic_tokenizer false -map_path MAP_PATH
```

* `RAW_PATH` is the directory containing tokenized files (`../merged_stories_tokenized`), `JSON_PATH` is the target directory to save the generated json files (`../json_data/cnndm`), `MAP_PATH` is the  directory containing the urls files (`../urls`)

####  Step 5. Format to PyTorch Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH  -lower -n_cpus 1 -log_file ../logs/preprocess.log
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)

## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

### Extractive Setting

```
python train.py -task ext -mode train -bert_data_path BERT_DATA_PATH -ext_dropout 0.1 -model_path MODEL_PATH -lr 2e-3 -visible_gpus 0,1,2 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert_cnndm -use_interval true -warmup_steps 10000 -max_pos 512
```

### Abstractive Setting

#### TransformerAbs (baseline)
```
python train.py -mode train -accum_count 5 -batch_size 300 -bert_data_path BERT_DATA_PATH -dec_dropout 0.1 -log_file ../../logs/cnndm_baseline -lr 0.1 -model_path MODEL_PATH -save_checkpoint_steps 2000 -seed 777 -sep_optim false -train_steps 200000 -use_bert_emb true -use_interval true -warmup_steps 8000  -visible_gpus 0,1,2,3 -max_pos 512 -report_every 50 -enc_hidden_size 512  -enc_layers 6 -enc_ff_size 2048 -enc_dropout 0.1 -dec_layers 6 -dec_hidden_size 512 -dec_ff_size 2048 -encoder baseline -task abs
```
#### BertAbs
```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3  -log_file ../logs/abs_bert_cnndm
```
#### BertExtAbs
```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file ../logs/abs_bert_cnndm  -load_from_extractive EXT_CKPT   
```
* `EXT_CKPT` is the saved `.pt` checkpoint of the extractive model.




## Model Evaluation
```
 python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -log_file ../logs/val_abs_bert_cnndm -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_cnndm 
```
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries (this will take a while)


