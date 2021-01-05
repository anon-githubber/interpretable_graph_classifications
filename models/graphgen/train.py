import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter

from utils import save_model, load_model, get_model_attribute
from graphgen.train import evaluate_loss as eval_loss_dfscode_rnn
from baselines.graph_rnn.train import evaluate_loss as eval_loss_graph_rnn
from baselines.dgmg.train import evaluate_loss as eval_loss_dgmg
from graphgen_cls.train import evaluate_loss as eval_loss_dfscode_rnn_cls
from sklearn.metrics import accuracy_score

import sys


def evaluate_loss(args, model, data, feature_map):
    if args.note == 'GraphRNN':
        loss = eval_loss_graph_rnn(args, model, data, feature_map)
    elif args.note == 'DFScodeRNN':
        loss = eval_loss_dfscode_rnn(args, model, data, feature_map)
    elif args.note == 'DFScodeRNN_cls':
        loss = eval_loss_dfscode_rnn_cls(args, model, data, feature_map)
    elif args.note == 'DGMG':
        loss = eval_loss_dgmg(model, data)

    return loss


def train_epoch(
        epoch, args, model, dataloader_train, optimizer,
        scheduler, feature_map, summary_writer=None):
    # Set training mode for modules
    for _, net in model.items():
        net.train()

    batch_count = len(dataloader_train) # number of batches
    # print('number of batches: ', batch_count)
    # print('batch_count: ', batch_count)
    # print('len of train dataset: ', len(dataloader_train.dataset))

    y_preds=[]
    y_trues=[]

    total_loss = 0.0
    for batch_id, data in enumerate(dataloader_train):

        # print('train.py: batch_id: ', batch_id)
        # print('train.py: data: ', data)

        for _, net in model.items():
            net.zero_grad()

        loss, y_pred, y_true = evaluate_loss(args, model, data, feature_map)
        y_preds.extend(y_pred)
        y_trues.extend(y_true)

        loss.backward()
        total_loss += loss.data.item()

        # Clipping gradients
        if args.gradient_clipping:
            for _, net in model.items():
                clip_grad_value_(net.parameters(), 1.0)

        # Update params of rnn and mlp
        for _, opt in optimizer.items():
            opt.step()

        for _, sched in scheduler.items():
            sched.step()

        if args.log_tensorboard:
            summary_writer.add_scalar('{} {} Loss/train batch'.format(
                args.note, args.graph_type), loss, batch_id + batch_count * epoch)

    y_preds = [1. if n >= 0.5 else 0. for n in y_preds]
    # print('y_trues: ', y_trues)
    # print('y_preds: ', y_preds)

    return total_loss / batch_count, accuracy_score(y_trues, y_preds)


def test_data(args, model, dataloader, feature_map):
    for _, net in model.items():
        net.eval()

    batch_count = len(dataloader)
    with torch.no_grad():
        total_loss = 0.0
        for _, data in enumerate(dataloader):
            loss = evaluate_loss(args, model, data, feature_map)
            total_loss += loss.data.item()

    return total_loss / batch_count


# Main training function
def train(args, dataloader_train, model, feature_map, dataloader_validate=None):
    # initialize optimizer
    optimizer = {}
    for name, net in model.items():
        optimizer['optimizer_' + name] = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
            weight_decay=5e-5)

    scheduler = {}
    for name, net in model.items():
        scheduler['scheduler_' + name] = MultiStepLR(
            optimizer['optimizer_' + name], milestones=args.milestones,
            gamma=args.gamma)

    if args.load_model:
        load_model(args.load_model_path, args.device,
                   model, optimizer, scheduler)
        print('Model loaded')

        epoch = get_model_attribute('epoch', args.load_model_path, args.device)
    else:
        epoch = 0

    if args.log_tensorboard:
        writer = SummaryWriter(
            log_dir=args.tensorboard_path + args.fname + ' ' + args.time, flush_secs=5)
    else:
        writer = None

    while epoch < args.epochs:
        loss, acc = train_epoch(
            epoch, args, model, dataloader_train, optimizer, scheduler, feature_map, writer)
        epoch += 1
        print('Epoch: {}/{}, train loss: {:.3f}, accuray: {:.3f}'.format(epoch, args.epochs, loss, acc))

        # logging
        if args.log_tensorboard:
            writer.add_scalar('{} {} Loss/train'.format(
                args.note, args.graph_type), loss, epoch)

        # save model checkpoint
        if args.save_model and epoch != 0 and epoch % args.epochs_save == 0:
            save_model(
                epoch, args, model, optimizer, scheduler, feature_map=feature_map)
            print(
                'Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

        if dataloader_validate is not None and epoch % args.epochs_validate == 0:
            loss_validate = test_data(
                args, model, dataloader_validate, feature_map)
            if args.log_tensorboard:
                writer.add_scalar('{} {} Loss/validate'.format(
                    args.note, args.graph_type), loss_validate, epoch)
            else:
                print('Epoch: {}/{}, validation loss: {:.6f}'.format(
                    epoch, args.epochs, loss_validate))

    save_model(epoch, args, model, optimizer,
               scheduler, feature_map=feature_map)
    print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))
