# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup

from data_utils import ABSADataset
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores

logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='rest15', type=str,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", default=True,action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true',
                        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=128,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    output_dir = f"outputs/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    if args.do_train:
        print("train")

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                       data_type=type_path, max_len=args.max_seq_length)


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    # 初始化
    def __init__(self, hparams1, tfm_model, tokenizer):
        super(T5FineTuner, self).__init__()
        self.hparams1 = hparams1
        self.model = tfm_model
        self.tokenizer = tokenizer

    def is_logger(self):
        return True

    # 定义前向传播过程，返回一个模型
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    # 自定义的函数，pl框架的train_step会调用它
    def _step(self, batch):
        lm_labels = batch["target_ids"]

        # 在 tokenizer 中，可以通过 tokenizer.pad_token_id 确认 padding id
        '''举个例子：
            sequence1_ids = [[200, 200, 200]]
            sequence2_ids = [[200, 200]]
            batched_ids = [[200, 200, 200], [200, 200, tokenizer.pad_token_id]]'''
        # 实际上，lm_labels的值并没有发生变化
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        # return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
        print("avg_train_loss:{}    ".format(avg_train_loss))

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams1.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams1.learning_rate, eps=self.hparams1.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, **kwargs):
        if self.trainer.use_tpu:
            print("111")
            # xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    # 加载dataloader，用来给训练加载数据
    def train_dataloader(self):
        # 获得数据集
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams1)
        # 获得dataloader
        dataloader = DataLoader(train_dataset, batch_size=self.hparams1.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        # print(len(dataloader.dataset))
        # print(self.hparams1.train_batch_size)
        # # print(len(self.hparams1.n_gpu))
        # print(self.hparams1.gradient_accumulation_steps)
        #
        # print(self.hparams1.num_train_epochs)

        # 获得总训练步数
        # dataset的长度/batch_size/梯度累计*epoch
        '''如果显存不足，我们可以通过gradient_accumulation_steps梯度累计来解决。
            假设原来的batch size=10,数据总量为1000，那么一共需要100train steps，同时一共进行100次梯度更新。
            若是显存不够，我们需要减小batch size，我们设置gradient_accumulation_steps=2，那么我们新的batch size=10/2=5，
            我们需要运行两次，才能在内存中放入10条数据，梯度更新的次数不变为100次，那么我们的train steps=200'''
        t_total = (
                (len(dataloader.dataset) // self.hparams1.train_batch_size)
                // self.hparams1.gradient_accumulation_steps
                * float(self.hparams1.num_train_epochs)
        )

        # 使用学习率预热。
        '''由于刚开始训练时,模型的权重(weights)是随机初始化的，此时若选择⼀个较⼤的学习率,可能带来模型的不稳定(振荡)，选择Warmup预热
        学习率的⽅式，可以使得开始训练的⼏个epoches或者⼀些steps内学习率较⼩,在预热的⼩学习率下，模型可以慢慢趋于稳定,等模型相对
        稳定后再选择预先设置的学习率进⾏训练,使得模型收敛速度变得更快，模型效果更佳'''
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams1.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams1)
        return DataLoader(val_dataset, batch_size=self.hparams1.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams1.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(data_loader, model, sents):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{args.n_gpu}')
    model.model.to(device)

    model.model.eval()

    outputs, targets = [], []

    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.model.generate(input_ids=batch['source_ids'].to(device),
                                    attention_mask=batch['source_mask'].to(device),
                                    max_length=128)  # num_beams=8, early_stopping=True)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)

    '''
    print("\nPrint some results to check the sanity of generation method:", '\n', '-'*30)
    for i in [1, 5, 25, 42, 50]:
        try:
            print(f'>>Target    : {targets[i]}')
            print(f'>>Generation: {outputs[i]}')
        except UnicodeEncodeError:
            print('Unable to print due to the coding error')
    print()
    '''

    scores, all_labels, all_preds = compute_scores(outputs, targets, sents)
    results = {'scores': scores, 'labels': all_labels, 'preds': all_preds}
    # pickle.dump(results, open(f"{args.output_dir}/results-{args.dataset}.pickle", 'wb'))

    return scores


if __name__ == '__main__':
    # 初始化参数
    args = init_args()
    print("\n", "=" * 30, f"NEW EXP: ASQP on {args.dataset}", "=" * 30, "\n")

    # sanity check
    # show one sample to check the code and the expected output
    # 加载预训练模型
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    #输出一个测试数据
    print(f"Here is an example (from the dev set):")

    # 使用自定义的数据类，得到数据集dataset
    dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                          data_type='train', max_len=args.max_seq_length)
    data_sample = dataset[1]  # 得到一个例子
    # print(data_sample['source_ids'])
    print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))

    # training process
    if args.do_train:
        print("\n****** Conduct Training ******")

        # 加载T5模型
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        # 通过自定义的T5微调类，获得初始化的模型
        model = T5FineTuner(args, tfm_model, tokenizer)

        # checkpoint_callback = pl.callbacks.ModelCheckpoint(
        #     filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=3
        # )

        # 训练器的一些参数
        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            callbacks=[LoggingCallback()],
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # save the final model
        # model.model.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)

        print("Finish training and saving the model!")

    # evaluation
    if args.do_direct_eval:
        print("\n****** Conduct Evaluating with the last state ******")

        # model = T5FineTuner(args)

        # print("Reload the model")
        # model.model.from_pretrained(args.output_dir)

        sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

        print()
        test_dataset = ABSADataset(tokenizer, data_dir=args.dataset,
                                   data_type='test', max_len=args.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
        # print(test_loader.device)

        # compute the performance scores
        scores = evaluate(test_loader, model, sents)

        # write to file
        log_file_path = f"results_log/{args.dataset}.txt"
        local_time = time.asctime(time.localtime(time.time()))

        exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
        exp_results = f"F1 = {scores['f1']:.4f}"

        log_str = f'============================================================\n'
        log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

        if not os.path.exists('./results_log'):
            os.mkdir('./results_log')

        with open(log_file_path, "a+") as f:
            f.write(log_str)

    if args.do_inference:
        print("\n****** Conduct inference on trained checkpoint ******")

        # initialize the T5 model from previous checkpoint
        print(f"Load trained model from {args.output_dir}")
        print('Note that a pretrained model is required and `do_true` should be False')
        tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

        model = T5FineTuner(args, tfm_model, tokenizer)

        sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

        print()
        test_dataset = ABSADataset(tokenizer, data_dir=args.dataset,
                                   data_type='test', max_len=args.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
        # print(test_loader.device)

        # compute the performance scores
        scores = evaluate(test_loader, model, sents)

        # write to file
        log_file_path = f"results_log/{args.dataset}.txt"
        local_time = time.asctime(time.localtime(time.time()))

        exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
        exp_results = f"F1 = {scores['f1']:.4f}"

        log_str = f'============================================================\n'
        log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

        if not os.path.exists('./results_log'):
            os.mkdir('./results_log')

        with open(log_file_path, "a+") as f:
            f.write(log_str)
