import argparse
from model_utils import get_opt, opt_sparsellm, opt_eval
from datautils import get_loaders
import torch

def main():
    parser = argparse.ArgumentParser()

    # Arguments parsing
    parser.add_argument('--model', type=str, default='facebook/opt-125m', help='OPT model to load; pass `facebook/opt-X`.')
    parser.add_argument('--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], default='c4', help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=64, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Target sparsity')
    parser.add_argument('--prunen', type=int, default=0, help='N for N:M pruning.')
    parser.add_argument('--prunem', type=int, default=0, help='M for N:M pruning.')
    parser.add_argument('--blocksize', type=int, default=128, help='Blocksize to use for adaptive mask selection.')
    parser.add_argument('--gmp', action='store_true', help='Whether to run the GMP baseline.')
    parser.add_argument('--wbits', type=int, default=16, help='Whether to quantize as well.')
    parser.add_argument('--minlayer', type=int, default=-1, help='Prune all layers with id >= this.')
    parser.add_argument('--maxlayer', type=int, default=1000, help='Prune all layers with id < this.')
    parser.add_argument('--prune_only', type=str, default='', help='Prune only layers that contain this text.')
    parser.add_argument('--invert', action='store_true', help='Invert subset.')
    parser.add_argument('--save', type=str, default='', help='Path to saved model.')
    parser.add_argument('--log_wandb', action='store_true', help='Whether to log to wandb.')
    parser.add_argument(
        '--use_vacuum', 
        action='store_true', 
        help='Whether to use the vacuum pruning method for FFN layers.'
    )
    parser.add_argument(
        '--n_vac', 
        type=int, 
        default=3, 
        help='Power of the vacuum function w^(2n+1).'
    )
    parser.add_argument(
        '--n_vac_att', 
        type=int, 
        default=1, 
        help='Power of the vacuum function w^(2n+1).'
    )
    parser.add_argument(
        '--lmbda_vac', 
        type=float, 
        default=0.01, 
        help='Lambda regularization for the vacuum cooking phase.'
    )
    parser.add_argument(
        '--cooking_iters', 
        type=int, 
        default=20, 
        help='Number of optimization steps in the vacuum cooking phase.'
    )
    parser.add_argument(
        '--lr_vac', 
        type=float, 
        default=1e-3, 
        help='Learning rate for the vacuum optimizer.'
    )
    args = parser.parse_args()

    model = get_opt(args)
    model.eval()
    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)

    if (args.sparsity or args.prunen) and not args.gmp:
        opt_sparsellm(model, dataloader, torch.device('cuda'), args)

    for dataset in ['wikitext2',  'c4']:  # 'ptb',
        dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
        opt_eval(model, testloader, torch.device('cuda'), args, dataset)

    if args.save:
        model.save_pretrained(args.save)

if __name__ == '__main__':
    main()
