##############################################
#            Author: Yuqing Zhou
##############################################
from train import *


def main():
    # dataset = 'MultiNLI' # 'civilcomments' #'waterbirds'
    # args = SimpleNamespace(
    #     dataset_name=dataset,
    #     bert_version=1,
    #     cpns_version=0,
    #     reweight_version=3,
    #     n_exp=0,
    #     model_name=DATASET_INFO[dataset]['model'],

    #     n_epochs=200,
    #     batch_size=128,
    #     feature_size=768,

    #     finetune_flg=True,
    #     reweight_flg=True,  # True

    #     lr=1e-2, #2e-5,
    #     momentum=0.9,
    #     weight_decay=0, #1e-4,
    #     reg_disentangle=0,
    #     reg_causal=0,
    #     gamma_reweight=1.0,

    #     dfr_reweighting_frac=0.2,
    #     reweighting_seed=1,
    #     dfr_reweighting_drop=True,
    #     max_grad_norm=1.0,
    #     load_best_model=True,
    #     warmup_steps=0
    # )

    dataset = 'civilcomments'
    args = SimpleNamespace(
        dataset_name=dataset,
        base_model_version=1,
        cpns_version=0,
        reweight_version=3,
        n_exp=2,
        model_name=DATASET_INFO[dataset]['model'],

        n_epochs=50,
        batch_size=128,
        feature_size=768,

        finetune_flg=True,
        reweight_flg=True,  # True

        lr=1e-2,  # 1e-5,
        momentum=0.9,
        weight_decay=0,  # 0.01,
        reg_disentangle=0,
        reg_causal=0,
        gamma_reweight=0,

        dfr_reweighting_frac=0.2,
        reweighting_seed=1,
        dfr_reweighting_drop=True,
        max_grad_norm=1.0,
        load_best_model=True,
        warmup_steps=0
    )

    dataset = 'waterbirds'
    args = SimpleNamespace(
        dataset_name=dataset,
        base_model_version=1,
        cpns_version=0,
        reweight_version=0,
        n_exp=0,
        model_name=DATASET_INFO[dataset]['model'],

        n_epochs=50,
        batch_size=2,
        feature_size=2048,

        finetune_flg=False,
        reweight_flg=False,  # True

        lr=3e-3,
        momentum=0.9,
        weight_decay=1e-4,
        reg_disentangle=0,
        reg_causal=0,
        gamma_reweight=0,

        dfr_reweighting_frac=0.2,
        reweighting_seed=1,
        dfr_reweighting_drop=True,
        max_grad_norm=1.0,
        load_best_model=True,
        warmup_steps=0
    )



    if args.dataset_name == 'civilcomments':
        train_civil(args)
    elif args.dataset_name == 'MultiNLI':
        train_nli(args)
    elif args.dataset_name == 'waterbirds' or 'celeba':
        train_cv(args)


if __name__ == '__main__':
    main()

