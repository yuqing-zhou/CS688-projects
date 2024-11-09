##############################################
#            Author: Yuqing Zhou
##############################################
from train import *


def main():
    dataset = 'MultiNLI' #'waterbirds'
    args = SimpleNamespace(
        dataset_name=dataset,
        bert_version=0,
        cpns_version=1,
        reweight_version=0,
        n_exp=0,
        model_name=DATASET_INFO[dataset]['model'],

        n_epochs=3,
        batch_size=10,
        feature_size=128,

        finetune_flg=False,
        reweight_flg=False,  # True

        lr=3e-3,
        momentum=0.9,
        weight_decay=0, #1e-4,
        reg_disentangle=0,
        reg_causal=0,
        gamma_reweight=0.5,

        dfr_reweighting_frac=0.2,
        reweighting_seed=1,
        dfr_reweighting_drop=True,
        max_grad_norm=1.0,
        load_best_model=False
    )
    if args.dataset_name == 'civilcomments':
        train_civil(args)
    elif args.dataset_name == 'MultiNLI':
        train_nli(args)
    elif args.dataset_name == 'waterbirds' or 'celeba':
        train_cv(args)


if __name__ == '__main__':
    main()

