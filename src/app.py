# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, "..")))

import argparse
import glob
import os
import random
import signal
import time

import torch
from pytorch_transformers import BertTokenizer

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.loss import abs_loss
from models.model_builder import AbsSummarizer
from models.predictor import build_predictor
from models.trainer import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
# logger.info(pformat(args))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_abs_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """

    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_abs_single(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def validate_abs(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            if (args.test_start_from != -1 and step < args.test_start_from):
                xent_lst.append((1e6, cp))
                continue
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test_abs(args, device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_abs(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    valid_loss = abs_loss(model.generator, symbols, model.vocab_size, train=False, device=device)

    trainer = build_trainer(args, device_id, model, None, valid_loss)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()


def test_abs(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)



def baseline(args, cal_lead=False, cal_oracle=False):
    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.batch_size, 'cpu',
                                       shuffle=False, is_test=True)

    trainer = build_trainer(args, '-1', None, None, None)
    #
    if (cal_lead):
        trainer.test(test_iter, 0, cal_lead=True)
    elif (cal_oracle):
        trainer.test(test_iter, 0, cal_oracle=True)


def train_abs(args, device_id):
    if (args.world_size > 1):
        train_abs_multi(args)
    else:
        train_abs_single(args, device_id)


def train_abs_single(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    if (args.load_from_extractive != ''):
        logger.info('Loading bert from extractive model %s' % args.load_from_extractive)
        bert_from_extractive = torch.load(args.load_from_extractive, map_location=lambda storage, loc: storage)
        bert_from_extractive = bert_from_extractive['model']
    else:
        bert_from_extractive = None
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False)

    model = AbsSummarizer(args, device, checkpoint, bert_from_extractive)
    if (args.sep_optim):
        optim_bert = model_builder.build_optim_bert(args, model, checkpoint)
        optim_dec = model_builder.build_optim_dec(args, model, checkpoint)
        optim = [optim_bert, optim_dec]
    else:
        optim = [model_builder.build_optim(args, model, checkpoint)]

    logger.info(model)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    train_loss = abs_loss(model.generator, symbols, model.vocab_size, device, train=True,
                          label_smoothing=args.label_smoothing)

    trainer = build_trainer(args, device_id, model, optim, train_loss)

    trainer.train(train_iter_fct, args.train_steps)




def test_text_abs(args):

    logger.info('Loading checkpoint from %s' % args.test_from)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = data_loader.load_text(args, args.text_src, args.text_tgt, device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, -1)




class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]




@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        # скопировано из функции def run():, заменен словарь args на фактически распарсенный набор параметров #########
        args = {'dataset_path': './persona_2ver_1500pers_en.json',
                'dataset_cache': './dataset_cache',
                'model': 'openai-gpt',
                'model_checkpoint': './runs/Nov23_16-25-39_joo-tf_openai-gpt/',
                'max_history': 2,
                'device': 'cpu', # cuda
                'max_length': 30,
                'min_length': 2,
                'seed': 0,
                'temperature': 0.7,
                'top_k': 0,
                'top_p': 0.9,
                }

        args2 = Map(args)

        # logging.basicConfig(level=logging.INFO)
        # logger = logging.getLogger(__file__)
        # logger.info(pformat(args))

        if args['model_checkpoint'] == "":
            if args['model'] == 'gpt2':
                raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
            else:
                args['model_checkpoint'] = download_pretrained_model()

        if args['seed'] != 0:
            random.seed(args['seed'])
            torch.random.manual_seed(args['seed'])
            torch.cuda.manual_seed(args['seed'])

        logger.info("Get pretrained model and tokenizer")
        tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args['model'] == 'gpt2' else (
            OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
        tokenizer = tokenizer_class.from_pretrained(args['model_checkpoint'])
        model = model_class.from_pretrained(args['model_checkpoint'])
        model.to(args['device'])
        add_special_tokens_(model, tokenizer)

        logger.info("Sample a personality")
        dataset = get_dataset(tokenizer, args['dataset_path'], args['dataset_cache'])
        personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
        # personality = random.choice(personalities)
        personality = personalities[1]  # the first is about ze
        logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
        ###############################################################################################################

        # фактический вызов модели путорч на предсказание, из файла interact.py
        history = []
        raw_text = request.get_json(force=True)['raw_text'] # {"raw_text":"some text to pass in pytorch gpt", "username": "fizz bizz"}
        print('########### raw_text: ', raw_text)
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = interact.sample_sequence(personality, history, tokenizer, model, args2)
        history.append(out_ids)
        history = history[-(2 * args2.max_history + 1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)

        return jsonify({'q': raw_text, 'a': out_text})

    if request.method == 'GET':

        # скопировано из функции def run():, заменен словарь args на фактически распарсенный набор параметров #########
        args = {'dataset_path': './persona_2ver_1500pers_en.json',
                'dataset_cache': './dataset_cache',
                'model': 'openai-gpt',
                'model_checkpoint': './runs/Nov23_16-25-39_joo-tf_openai-gpt/',
                'max_history': 2,
                'device': 'cpu', # cuda
                'max_length': 30,
                'min_length': 2,
                'seed': 0,
                'temperature': 0.7,
                'top_k': 0,
                'top_p': 0.9,
                }

        args2 = Map(args)

        # logging.basicConfig(level=logging.INFO)
        # logger = logging.getLogger(__file__)
        # logger.info(pformat(args))

        if args['model_checkpoint'] == "":
            if args['model'] == 'gpt2':
                raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
            else:
                args['model_checkpoint'] = download_pretrained_model()

        if args['seed'] != 0:
            random.seed(args['seed'])
            torch.random.manual_seed(args['seed'])
            torch.cuda.manual_seed(args['seed'])

        logger.info("Get pretrained model and tokenizer")
        tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args['model'] == 'gpt2' else (
            OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
        tokenizer = tokenizer_class.from_pretrained(args['model_checkpoint'])
        model = model_class.from_pretrained(args['model_checkpoint'])
        model.to(args['device'])
        add_special_tokens_(model, tokenizer)

        logger.info("Sample a personality")
        dataset = get_dataset(tokenizer, args['dataset_path'], args['dataset_cache'])
        personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
        # personality = random.choice(personalities)
        personality = personalities[1]  # the first is about ze
        logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
        ###############################################################################################################

        # фактический вызов модели путорч на предсказание, из файла interact.py
        history = []
        # raw_text = request.get_json(force=True)['raw_text'] # {"raw_text":"some text to pass in pytorch gpt", "username": "fizz bizz"}
        raw_text = request.args.get('raw_text')  # http://213.159.215.173:5000/predict?raw_text=how
        print('########### raw_text: ', raw_text)
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = interact.sample_sequence(personality, history, tokenizer, model, args2)
        history.append(out_ids)
        history = history[-(2 * args2.max_history + 1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)

        return jsonify({'q': raw_text, 'a': out_text})



@app.route('/get_summary', methods=['GET', 'POST'])
def get_summary():
    if request.method == 'POST':

        raw_text = request.args.get('raw_text')  # http://213.159.215.173:5000/get_summary?raw_text=how
        # print('########### raw_text: ', raw_text)
        while not raw_text:
            print('Prompt should not be empty!')
        file1 = open("../raw_data/raw_text.txt", "w+")
        file1.write(str(raw_text.encode('utf-8'))) # Hello \n
        file1.close()

        #############
        # некоторые аргс изменены в соответствии с парамтерами запуска на инференс:
        # python train.py -task abs -mode test_text -text_src '../raw_data/naked_photos_petapixel.txt' -bert_data_path '../bert_data/' -ext_dropout 0.1 -model_path '../models/' -test_from '../models/model_step_154000.pt' -lr 2e-3 -visible_gpus -1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 140 -train_steps 50000 -accum_count 2 -log_file ../logs/abs_bert -use_interval true -warmup_steps 10000 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path '../results/cnndm' -test_all True
        #############
        # parser = argparse.ArgumentParser()
        # parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
        # parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
        # parser.add_argument("-mode", default='test_text', type=str, choices=['train', 'validate', 'test', 'test_text'])
        # parser.add_argument("-bert_data_path", default='../bert_data/')
        # parser.add_argument("-model_path", default='../models/')
        # parser.add_argument("-result_path", default='../results/cnndm')
        # parser.add_argument("-temp_dir", default='../temp')
        # parser.add_argument("-text_src", default='../raw_data/raw_text.txt')
        # parser.add_argument("-text_tgt", default='')
        #
        # parser.add_argument("-batch_size", default=140, type=int)
        # parser.add_argument("-test_batch_size", default=200, type=int)
        # parser.add_argument("-max_ndocs_in_batch", default=6, type=int)
        #
        # parser.add_argument("-max_pos", default=512, type=int)
        # parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
        # parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
        # parser.add_argument("-load_from_extractive", default='', type=str)
        #
        # parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=False)
        # parser.add_argument("-lr_bert", default=2e-3, type=float)
        # parser.add_argument("-lr_dec", default=2e-3, type=float)
        # parser.add_argument("-use_bert_emb", type=str2bool, nargs='?', const=True, default=False)
        #
        # parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
        # parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
        # parser.add_argument("-dec_dropout", default=0.2, type=float)
        # parser.add_argument("-dec_layers", default=6, type=int)
        # parser.add_argument("-dec_hidden_size", default=768, type=int)
        # parser.add_argument("-dec_heads", default=8, type=int)
        # parser.add_argument("-dec_ff_size", default=2048, type=int)
        # parser.add_argument("-enc_hidden_size", default=512, type=int)
        # parser.add_argument("-enc_ff_size", default=512, type=int)
        # parser.add_argument("-enc_dropout", default=0.2, type=float)
        # parser.add_argument("-enc_layers", default=6, type=int)
        #
        # # params for EXT
        # parser.add_argument("-ext_dropout", default=0.1, type=float)
        # parser.add_argument("-ext_layers", default=2, type=int)
        # parser.add_argument("-ext_hidden_size", default=768, type=int)
        # parser.add_argument("-ext_heads", default=8, type=int)
        # parser.add_argument("-ext_ff_size", default=2048, type=int)
        #
        # parser.add_argument("-label_smoothing", default=0.1, type=float)
        # parser.add_argument("-generator_shard_size", default=32, type=int)
        # parser.add_argument("-alpha", default=0.6, type=float)
        # parser.add_argument("-beam_size", default=5, type=int)
        # parser.add_argument("-min_length", default=40, type=int)
        # parser.add_argument("-max_length", default=200, type=int)
        # parser.add_argument("-max_tgt_len", default=140, type=int)
        #
        # parser.add_argument("-param_init", default=0, type=float)
        # parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
        # parser.add_argument("-optim", default='adam', type=str)
        # parser.add_argument("-lr", default=1, type=float)
        # parser.add_argument("-beta1", default=0.9, type=float)
        # parser.add_argument("-beta2", default=0.999, type=float)
        # parser.add_argument("-warmup_steps", default=8000, type=int)
        # parser.add_argument("-warmup_steps_bert", default=8000, type=int)
        # parser.add_argument("-warmup_steps_dec", default=8000, type=int)
        # parser.add_argument("-max_grad_norm", default=0, type=float)
        #
        # parser.add_argument("-save_checkpoint_steps", default=5, type=int)
        # parser.add_argument("-accum_count", default=1, type=int)
        # parser.add_argument("-report_every", default=1, type=int)
        # parser.add_argument("-train_steps", default=1000, type=int)
        # parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)
        #
        # parser.add_argument('-visible_gpus', default='-1', type=str)
        # parser.add_argument('-gpu_ranks', default='0', type=str)
        # parser.add_argument('-log_file', default='../logs/abs_bert')
        # parser.add_argument('-seed', default=666, type=int)
        #
        # parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=True)
        # parser.add_argument("-test_from", default='../models/model_step_154000.pt')
        # parser.add_argument("-test_start_from", default=-1, type=int)
        #
        # parser.add_argument("-train_from", default='')
        # parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
        # parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

        # самые важные аргументы для суммаризации:
        args = {'task': 'abs',
                'mode': 'test_text',
                'model_path': '../models/',
                'result_path': '../results/cnndm',
                'text_src': '../raw_data/raw_text.txt',
                'device': 'cpu', # cuda
                'test_from': '../models/model_step_154000.pt',
                'visible_gpus': '-1',
                'gpu_ranks': '0',
                'log_file': '../logs/abs_bert',
                'top_k': 0,
                'top_p': 0.9,
                }

        args = Map(args)
        # Внимание, закомментил args = parser.parse_args() тк выдает ошибку при запуске с gunicorn https://github.com/benoitc/gunicorn/issues/1867
        #args = parser.parse_args()
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
        args.world_size = len(args.gpu_ranks)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

        init_logger(args.log_file)
        device = "cpu" if args.visible_gpus == '-1' else "cuda"
        device_id = 0 if device == "cuda" else -1

        if (args.task == 'abs'):
            if (args.mode == 'train'):
                train_abs(args, device_id)
            elif (args.mode == 'validate'):
                validate_abs(args, device_id)
            elif (args.mode == 'lead'):
                baseline(args, cal_lead=True)
            elif (args.mode == 'oracle'):
                baseline(args, cal_oracle=True)
            if (args.mode == 'test'):
                cp = args.test_from
                try:
                    step = int(cp.split('.')[-2].split('_')[-1])
                except:
                    step = 0
                test_abs(args, device_id, cp, step)
            elif (args.mode == 'test_text'):
                test_text_abs(args)  # вызываем на инференс именно test_text_abs

        elif (args.task == 'ext'):
            if (args.mode == 'train'):
                train_ext(args, device_id)
            elif (args.mode == 'validate'):
                validate_ext(args, device_id)
            if (args.mode == 'test'):
                cp = args.test_from
                try:
                    step = int(cp.split('.')[-2].split('_')[-1])
                except:
                    step = 0
                test_ext(args, device_id, cp, step)
            elif (args.mode == 'test_text'):
                test_text_ext(args)

        # текст саммари находится в results/cnndm.-1.candidate
        f = open("../results/cnndm.-1.candidate", "r")
        if f.mode == 'r':
            out_text = f.read()
        from nltk.tokenize import sent_tokenize
        out_text = out_text.replace('<q>', '. ')
        input_sen = out_text # 'hello! how are you? please remember capitalization. EVERY time.'
        sentences = sent_tokenize(input_sen)
        sentences = [sent.capitalize() for sent in sentences]
        print(sentences)
        text_summary = ' '.join([str(elem) for elem in sentences])
        return jsonify({'text_full': raw_text, 'text_summary': text_summary})

    if request.method == 'GET':

        raw_text = request.args.get('raw_text')  # http://213.159.215.173:5000/get_summary?raw_text=how
        # print('########### raw_text: ', raw_text)
        while not raw_text:
            print('Prompt should not be empty!')
        file1 = open("../raw_data/raw_text.txt", "w+")
        file1.write(str(raw_text.encode('utf-8'))) # Hello \n
        file1.close()

        #############
        # некоторые аргс изменены в соответствии с парамтерами запуска на инференс:
        # python train.py -task abs -mode test_text -text_src '../raw_data/naked_photos_petapixel.txt' -bert_data_path '../bert_data/' -ext_dropout 0.1 -model_path '../models/' -test_from '../models/model_step_154000.pt' -lr 2e-3 -visible_gpus -1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 140 -train_steps 50000 -accum_count 2 -log_file ../logs/abs_bert -use_interval true -warmup_steps 10000 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path '../results/cnndm' -test_all True
        #############
        # parser = argparse.ArgumentParser()
        # parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
        # parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
        # parser.add_argument("-mode", default='test_text', type=str, choices=['train', 'validate', 'test', 'test_text'])
        # parser.add_argument("-bert_data_path", default='../bert_data/')
        # parser.add_argument("-model_path", default='../models/')
        # parser.add_argument("-result_path", default='../results/cnndm')
        # parser.add_argument("-temp_dir", default='../temp')
        # parser.add_argument("-text_src", default='../raw_data/raw_text.txt')
        # parser.add_argument("-text_tgt", default='')
        #
        # parser.add_argument("-batch_size", default=140, type=int)
        # parser.add_argument("-test_batch_size", default=200, type=int)
        # parser.add_argument("-max_ndocs_in_batch", default=6, type=int)
        #
        # parser.add_argument("-max_pos", default=512, type=int)
        # parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
        # parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
        # parser.add_argument("-load_from_extractive", default='', type=str)
        #
        # parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=False)
        # parser.add_argument("-lr_bert", default=2e-3, type=float)
        # parser.add_argument("-lr_dec", default=2e-3, type=float)
        # parser.add_argument("-use_bert_emb", type=str2bool, nargs='?', const=True, default=False)
        #
        # parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
        # parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
        # parser.add_argument("-dec_dropout", default=0.2, type=float)
        # parser.add_argument("-dec_layers", default=6, type=int)
        # parser.add_argument("-dec_hidden_size", default=768, type=int)
        # parser.add_argument("-dec_heads", default=8, type=int)
        # parser.add_argument("-dec_ff_size", default=2048, type=int)
        # parser.add_argument("-enc_hidden_size", default=512, type=int)
        # parser.add_argument("-enc_ff_size", default=512, type=int)
        # parser.add_argument("-enc_dropout", default=0.2, type=float)
        # parser.add_argument("-enc_layers", default=6, type=int)
        #
        # # params for EXT
        # parser.add_argument("-ext_dropout", default=0.1, type=float)
        # parser.add_argument("-ext_layers", default=2, type=int)
        # parser.add_argument("-ext_hidden_size", default=768, type=int)
        # parser.add_argument("-ext_heads", default=8, type=int)
        # parser.add_argument("-ext_ff_size", default=2048, type=int)
        #
        # parser.add_argument("-label_smoothing", default=0.1, type=float)
        # parser.add_argument("-generator_shard_size", default=32, type=int)
        # parser.add_argument("-alpha", default=0.6, type=float)
        # parser.add_argument("-beam_size", default=5, type=int)
        # parser.add_argument("-min_length", default=40, type=int)
        # parser.add_argument("-max_length", default=200, type=int)
        # parser.add_argument("-max_tgt_len", default=140, type=int)
        #
        # parser.add_argument("-param_init", default=0, type=float)
        # parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
        # parser.add_argument("-optim", default='adam', type=str)
        # parser.add_argument("-lr", default=1, type=float)
        # parser.add_argument("-beta1", default=0.9, type=float)
        # parser.add_argument("-beta2", default=0.999, type=float)
        # parser.add_argument("-warmup_steps", default=8000, type=int)
        # parser.add_argument("-warmup_steps_bert", default=8000, type=int)
        # parser.add_argument("-warmup_steps_dec", default=8000, type=int)
        # parser.add_argument("-max_grad_norm", default=0, type=float)
        #
        # parser.add_argument("-save_checkpoint_steps", default=5, type=int)
        # parser.add_argument("-accum_count", default=1, type=int)
        # parser.add_argument("-report_every", default=1, type=int)
        # parser.add_argument("-train_steps", default=1000, type=int)
        # parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)
        #
        # parser.add_argument('-visible_gpus', default='-1', type=str)
        # parser.add_argument('-gpu_ranks', default='0', type=str)
        # parser.add_argument('-log_file', default='../logs/abs_bert')
        # parser.add_argument('-seed', default=666, type=int)
        #
        # parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=True)
        # parser.add_argument("-test_from", default='../models/model_step_154000.pt')
        # parser.add_argument("-test_start_from", default=-1, type=int)
        #
        # parser.add_argument("-train_from", default='')
        # parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
        # parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

        # самые важные аргументы для суммаризации:
        args = {'task': 'abs',
                'mode': 'test_text',
                'model_path': '../models/',
                'result_path': '../results/cnndm',
                'text_src': '../raw_data/raw_text.txt',
                'device': 'cpu', # cuda
                'test_from': '../models/model_step_154000.pt',
                'visible_gpus': '-1',
                'gpu_ranks': '0',
                'log_file': '../logs/abs_bert',
                'top_k': 0,
                'top_p': 0.9,
                }

        args = Map(args)
        # Внимание, закомментил args = parser.parse_args() тк выдает ошибку при запуске с gunicorn https://github.com/benoitc/gunicorn/issues/1867
        #args = parser.parse_args()
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
        args.world_size = len(args.gpu_ranks)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

        init_logger(args.log_file)
        device = "cpu" if args.visible_gpus == '-1' else "cuda"
        device_id = 0 if device == "cuda" else -1

        if (args.task == 'abs'):
            if (args.mode == 'train'):
                train_abs(args, device_id)
            elif (args.mode == 'validate'):
                validate_abs(args, device_id)
            elif (args.mode == 'lead'):
                baseline(args, cal_lead=True)
            elif (args.mode == 'oracle'):
                baseline(args, cal_oracle=True)
            if (args.mode == 'test'):
                cp = args.test_from
                try:
                    step = int(cp.split('.')[-2].split('_')[-1])
                except:
                    step = 0
                test_abs(args, device_id, cp, step)
            elif (args.mode == 'test_text'):
                test_text_abs(args)  # вызываем на инференс именно test_text_abs

        elif (args.task == 'ext'):
            if (args.mode == 'train'):
                train_ext(args, device_id)
            elif (args.mode == 'validate'):
                validate_ext(args, device_id)
            if (args.mode == 'test'):
                cp = args.test_from
                try:
                    step = int(cp.split('.')[-2].split('_')[-1])
                except:
                    step = 0
                test_ext(args, device_id, cp, step)
            elif (args.mode == 'test_text'):
                test_text_ext(args)

        # текст саммари находится в results/cnndm.-1.candidate
        f = open("../results/cnndm.-1.candidate", "r")
        if f.mode == 'r':
            out_text = f.read()
        from nltk.tokenize import sent_tokenize
        out_text = out_text.replace('<q>', '. ')
        input_sen = out_text # 'hello! how are you? please remember capitalization. EVERY time.'
        sentences = sent_tokenize(input_sen)
        sentences = [sent.capitalize() for sent in sentences]
        print(sentences)
        text_summary = ' '.join([str(elem) for elem in sentences])
        return jsonify({'text_full': raw_text, 'text_summary': text_summary})


@socketio.on('my event', namespace='/my_namespace')
# this method is invoked when an event called
# 'my event' is is triggered
def test_message(message):
    # this triggers new event called 'i said'
	emit('i said ', {'data': message['data']})

if __name__ == '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    # на локальном сервере работает
    #app.run(debug=True)#socketio.run(app, debug=True) #app.run()

    # на моем сервере firstvds jason
    # Внимание, запускать python app.py а не FLASK_APP=app.py flask run тк запустится как локалхост и будет ошибка 111 requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionRefusedError(111, 'Connection refused'))

    # dev server
    # app.run(host = '0.0.0.0', port = 5000, debug = True)

    # gunicorn
    app.run(host = '0.0.0.0', port = 5005, debug = True) # 0.0.0.0  213.159.215.173 # 35.202.164.44