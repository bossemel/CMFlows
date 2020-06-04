import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter

import datasets
import RealNVP_modules.flows as fnn
import RealNVP_modules.utils as utils
from datasets.copulas import Copula_sampler


def train(epoch):
    global global_step, writer
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))

        writer.add_scalar('training/loss', loss.item(), global_step)
        global_step += 1

    pbar.close()

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    if args.cond:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device), train_loader.dataset.tensors[1].to(data.device).float())
    else:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1


def validate(epoch, model, loader, prefix='Validation'):
    global global_step, writer

    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            val_loss += -model.log_probs(data, cond_data).sum().item()  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    writer.add_scalar('validation/LL', val_loss / len(loader.dataset), epoch)

    pbar.close()
    return val_loss / len(loader.dataset)


def jsd_eval(epoch, model, loader, prefix='Validation'):
    global global_step, writer

    model.eval()
    js_divergence = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval_JSD')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            current_jsd = model.jsd(data).sum().item()
            #if not math.isnan(current_jsd) and not math.isinf(current_jsd):
            js_divergence += current_jsd  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description('JSD: {:.6f}'.format(
            js_divergence / pbar.n))
    writer.add_scalar('js_divergence/LL', js_divergence / len(loader.dataset), epoch)

    pbar.close()
    return js_divergence / len(loader.dataset)


def margin_uniformity(epoch, model, loader, prefix='Validation'):
    global global_step, writer

    model.eval()
    t_metric_x1 = 0
    t_metric_x2 = 0
    m_metric_x1 = 0
    m_metric_x2 = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval_Marginals')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            current_t_metric_x1, current_m_metric_x1, current_t_metric_x2, current_m_metric_x2 = model.t_metric_eval(data)
            t_metric_x1 += current_t_metric_x1  # sum up batch loss
            t_metric_x2 += current_t_metric_x2  # sum up batch loss
            m_metric_x1 += current_m_metric_x1  # sum up batch loss
            m_metric_x2 += current_m_metric_x2  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description('t_metric_x1: {:.6f}'.format(
            t_metric_x1 / pbar.n))
        pbar.set_description('t_metric_x2: {:.6f}'.format(
            t_metric_x2 / pbar.n))
        pbar.set_description('m_metric_x1: {:.6f}'.format(
            m_metric_x1 / pbar.n))
        pbar.set_description('m_metric_x2: {:.6f}'.format(
            m_metric_x2 / pbar.n))
    writer.add_scalar('t_metric_x1/LL', t_metric_x1 / len(loader.dataset), epoch)
    writer.add_scalar('t_metric_x2/LL', t_metric_x2 / len(loader.dataset), epoch)
    writer.add_scalar('m_metric_x1/LL', m_metric_x1 / len(loader.dataset), epoch)
    writer.add_scalar('m_metric_x2/LL', m_metric_x2 / len(loader.dataset), epoch)

    pbar.close()
    return t_metric_x1 / len(loader.dataset), t_metric_x1 / len(loader.dataset), \
        m_metric_x1 / len(loader.dataset), m_metric_x2 / len(loader.dataset)


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Flows')
    parser.add_argument('--batch-size',
                        type=int,
                        default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        help='input batch size for testing (default: 1000)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument(
        '--dataset',
        default='MOONS',
        help='POWER | GAS | HEPMASS | MINIBONE | BSDS300 | MOONS | GAUSSIAN | TDISTR | CLAYTON | FRANK | GUMBEL')
    parser.add_argument('--flow', default='maf', help='flow to use: maf | realnvp | glow')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--cond',
        action='store_true',
        default=False,
        help='train class conditional flow (only for MNIST)')
    parser.add_argument(
        '--num-blocks',
        type=int,
        default=5,
        help='number of invertible blocks (default: 5)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1000,
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--obs',
        type=int,
        default=3000,
        help='How many data samples to generate')
    parser.add_argument(
        '--tau',
        type=float,
        default=0.5,
        help='tau to use for copula sampling')
    parser.add_argument(
        '--num_hidden',
        type=int,
        default=64,
        help='number of hidden units')
    parser.add_argument(
        '--random_seed',
        type=int,
        default=58093,
        help='random seed')
    parser.add_argument(
        '--df',
        type=int,
        default=2,
        help='degrees of freedom for student-t copula')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    assert args.dataset in [
        'POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300', 'MOONS', 'MNIST', 'GAUSSIAN', 'TDISTR', 'CLAYTON', 'FRANK', 'GUMBEL'
    ]
    if args.dataset in ['POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300', 'MOONS', 'MNIST']:
        dataset = getattr(datasets, args.dataset)()
    else:
        dataset = Copula_sampler(args.dataset, args.obs, args.tau, args.random_seed, args.df)

    train_tensor = torch.from_numpy(dataset.trn.x)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    num_cond_inputs = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    num_inputs = dataset.n_dims
    if args.dataset in ['POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300', 'MOONS', 'MNIST']:
        num_hidden = {
            'POWER': 100,
            'GAS': 100,
            'HEPMASS': 512,
            'MINIBOONE': 512,
            'BSDS300': 512,
            'MOONS': 64,
            'GAUSSIAN': 64,
            'CLAYTON': 64,
            'GUMBEL': 64,
            'FRANK': 64,
            'TDISTR': 64,
            'MNIST': 1024
        }[args.dataset]
    else:
        num_hidden = {args.dataset: args.num_hidden}[args.dataset]

    act = 'tanh' if args.dataset == 'GAS' else 'relu'

    modules = []

    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask

    model = fnn.FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    writer = SummaryWriter(comment=args.flow + "_" + args.dataset)
    global_step = 0

    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model

    for epoch in range(args.epochs):
        print('\nEpoch: {}'.format(epoch))

        train(epoch)
        validation_loss = validate(epoch, model, valid_loader)

        if epoch - best_validation_epoch >= 30:
            break

        if validation_loss < best_validation_loss:
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)

        print(
            'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
            format(best_validation_epoch, -best_validation_loss))

        if (args.dataset in ['MOONS', 'GAUSSIAN', 'TDISTR', 'CLAYTON', 'FRANK', 'GUMBEL']) and epoch % 10 == 0:
            utils.save_moons_plot(epoch, model, dataset)
        elif args.dataset == 'MNIST' and epoch % 1 == 0:
            utils.save_images(epoch, model, args.cond)

    validate(best_validation_epoch, best_model, test_loader, prefix='Test')

    jsd_eval(best_validation_epoch, best_model, test_loader, prefix='Test')

    print(margin_uniformity(best_validation_epoch, best_model, test_loader, prefix='Test'))
