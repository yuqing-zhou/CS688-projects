##############################################
#            Author: Yuqing Zhou
##############################################
import torch
import wandb
import os
from model import *
from utils import *
import optimizers
import utils
from utils.common_utils import get_default_args, AverageMeter
from utils.supervised_utils import train_epoch
from utils.supervised_utils import eval_model
from utils.logging import TrainLogger
# from utils.general import print_time_taken
import tqdm
from transformers import BertConfig, BertForSequenceClassification
from wilds_exps.transforms import initialize_transform
from torch.optim import AdamW
from pytorch_transformers import WarmupLinearSchedule

_bar_format = '{l_bar}{bar:50}{r_bar}{bar:-10b}'

A = 1
B = 0  # 273 #205 # 25 #  125 # 225

C = 0
D = 70
E = 1

F = 0  # 250 #200

def train_civil(args):
    bert_version = args.bert_version
    cpns_version = args.cpns_version
    reweight_version = args.reweight_version
    n_exp = args.n_exp
    seed = seeds[n_exp]
    set_seed(seed)

    reg_disentangle = args.reg_disentangle
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    model_name = args.model_name
    dfr_reweighting_frac = args.dfr_reweighting_frac  # 0.2
    DATASET = args.dataset_name

    num_classes = DATASET_INFO[DATASET]['num_classes']
    feature_size = args.feature_size
    finetune_flg = args.finetune_flg  # True #
    reweight_flg = args.reweight_flg  # True
    weight_decay = args.weight_decay
    load_best_model = args.load_best_model

    root_dir = '../data/'
    data_dir = root_dir + 'datasets/'
    model_save_path = root_dir + f'models/{DATASET}/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    total_weights = None
    best_model = None
    best_loss = float('inf')
    best_acc_wg = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if finetune_flg == True:
        if load_best_model:
            load_model_path = model_save_path + f'best_model_{bert_version}.pth'
        else:
            load_model_path = model_save_path + f'final_model_{bert_version}.pth'
        best_model_path = model_save_path + f'best_model_{bert_version}_{cpns_version}_{reweight_version}.pth'
        final_model_path = model_save_path + f'final_model_{bert_version}_{cpns_version}_{reweight_version}.pth'
        load_local_model = True
        reg_causal = args.reg_causal
        disentangle_en = False
        counterfactual_en = False #True
    else:
        best_model_path = model_save_path + f'best_model_{bert_version}.pth'
        final_model_path = model_save_path + f'final_model_{bert_version}.pth'
        load_local_model = False
        reg_causal = 0
        disentangle_en = False
        counterfactual_en = False

    if reweight_flg == True:
        gamma = args.gamma_reweight
    else:
        gamma = 0

    project_name = wandb_project_name(DATASET, bert_version, cpns_version, reweight_version)
    exp_name = wandb_exp_name(DATASET, bert_version, cpns_version, reweight_version, n_exp, lr, feature_size,
                              reg_disentangle, reg_causal, gamma)
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,  # "robust-learning",
        name=exp_name,  # f"disentangle_cov_reg_{bert_version}_{cpns_version}_{reweight_version}",
        # notes="Bert with the regularization of the covariance matrix of the input of the last layer",
        # notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later",
        notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later, reweights the CE loss",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": model_name,
            "dataset": DATASET,
            "epochs": n_epochs,
            "batch_size": batch_size,
            "dfr_reweighting_frac": dfr_reweighting_frac,
            "Regularization coefficient": reg_disentangle,
            "Bert feature size": feature_size,
            "Causal Regularization coefficient": reg_causal,
            "gamma": gamma,
            "seed": seed,
            "weight_decay": weight_decay,
        }
    )
    wandb.define_metric("epoch")
    wandb.define_metric("Train Loss", step_metric='epoch')
    wandb.define_metric("Train Accuracy", step_metric='epoch')
    wandb.define_metric("Validation Loss", step_metric='epoch')
    wandb.define_metric("Validation Accuracy", step_metric='epoch')
    wandb.define_metric("Best Validation Loss", step_metric='epoch')
    wandb.define_metric("Best Validation Accuracy", step_metric='epoch')
    wandb.define_metric("Best Validation Worst Group Accuracy", step_metric='epoch')

    model_config = BertConfig.from_pretrained(model_name, num_labels=num_classes)
    model = BertClassifierWithCovReg(model_name, num_labels=num_classes, feature_size=feature_size, device=device,
                                     reg=reg_disentangle, reg_causal=reg_causal, disentangle_en=disentangle_en,
                                     counterfactual_en=counterfactual_en, config=model_config).to(device)
    if load_local_model:
        model.load_state_dict(torch.load(load_model_path, map_location=device))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset_configs.dataset_defaults[DATASET]['batch_size'] = batch_size
    task_config = SimpleNamespace(
        root_dir=data_dir,
        # batch_size=batch_size,
        dfr_reweighting_drop=args.dfr_reweighting_drop,
        dfr_reweighting_seed=args.reweighting_seed,
        dfr_reweighting_frac=args.dfr_reweighting_frac,
        algorithm='ERM',
        load_featurizer_only=False,
        pretrained_model_path=None,
        **dataset_configs.dataset_defaults[DATASET],
    )

    task_config.model_kwargs = {}
    task_config.model = args.model_name
    train_data, val_data, test_data, reweighting_data = get_datasets(task_config, DATASET)
    if finetune_flg == True:
        train_data = reweighting_data
    train_loader = get_train_loader("standard", train_data, batch_size=batch_size, uniform_over_groups=False)
    val_loader = get_eval_loader("standard", val_data, batch_size=batch_size)
    test_loader = get_eval_loader("standard", test_data, batch_size=batch_size)
    # reweighting_loader = get_eval_loader("standard", reweighting_data, batch_size=args.batch_size)

    t_total = len(train_loader) * n_epochs
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if reweight_flg == True:
        with torch.no_grad():
            all_train_logits, all_train_y_true = [], []
            model.eval()
            for batch in train_loader:
                input_ids = batch[0][:, :, 0].to(device)
                attention_mask = batch[0][:, :, 1].to(device)
                labels = batch[1].to(device)

                logits, _ = model(input_ids, attention_mask, labels)

                all_train_logits.append(logits)
                all_train_y_true.append(labels)

            all_train_logits = torch.cat(all_train_logits, axis=0)
            all_train_y_true = torch.cat(all_train_y_true, axis=0)

            total_weights = compute_weights(all_train_logits, all_train_y_true, gamma, True)

    if load_local_model:
        model = model_parameters_freeze(model)

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        total_batches = 0
        batch_start_idx = 0
        batch_end_idx = 0
        for batch in train_loader:
            # inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = batch[0][:, :, 0].to(device)
            attention_mask = batch[0][:, :, 1].to(device)
            labels = batch[1].to(device)
            # print(input_ids.device)

            batch_end_idx = batch_start_idx + len(labels)
            weights = total_weights[batch_start_idx:batch_end_idx] if total_weights is not None else None
            batch_start_idx = batch_end_idx

            logits, loss = model(input_ids, attention_mask, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()
            optimizer.step()
            accuracy = compute_accuracy(logits, labels)
            total_train_accuracy += accuracy.item()
            total_train_loss += loss.item()
            # print(f"Epoch {epoch}, Loss: {loss.item()}")
            optimizer.zero_grad()

            total_batches += 1
            if total_batches % 50 == 0:
                print(f"Epoch {epoch}, batches {total_batches} , loss = {loss.item()}")

        avg_train_loss = total_train_loss / total_batches
        avg_train_accuracy = total_train_accuracy / total_batches
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_accuracy}")
        wandb.log({"Train Loss": avg_train_loss, "Train Accuracy": avg_train_accuracy, "epoch": epoch})

        model.eval()
        val_results, avg_val_loss, avg_val_accuracy = evaluation(model, val_data, val_loader, device)
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")
        print(val_results[1])
        # wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, "Validation Accuracy": avg_val_accuracy})
        wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, 'Validation Accuracy': val_results[0]['acc_avg'],
                   'Validation Worst Group Accuracy': val_results[0]['acc_wg']})

        torch.save(model.state_dict(), final_model_path)
        if val_results[0]['acc_wg'] > best_acc_wg:
            best_accuracy = val_results[0]['acc_avg']
            best_acc_wg = val_results[0]['acc_wg']
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Epoch {epoch}, Best Validation Loss: {best_loss}, Best Validation Accuracy: {best_accuracy}, Best Validation Worst Group Accuracy: {best_acc_wg}")
            print("Saved best model")
            wandb.log({"epoch": epoch, "Best Validation Loss": best_loss, "Best Validation Accuracy": best_accuracy,
                       "Best Validation Worst Group Accuracy": best_acc_wg})

    if load_best_model:
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    test_results, avg_test_loss, avg_test_accuracy = evaluation(model, test_data, test_loader, device)
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}")
    print(test_results[1])
    wandb.log({"Test Loss": avg_test_loss, "Test Accuracy": avg_test_accuracy})
    wandb.log(
        {'Test Mean Accuracy': test_results[0]['acc_avg'], 'Test Worst Group Accuracy': test_results[0]['acc_wg']})
    wandb.log(test_results[0])
    wandb.finish()


def train_nli(args):
    bert_version = args.bert_version
    cpns_version = args.cpns_version
    reweight_version = args.reweight_version
    n_exp = args.n_exp
    seed = seeds[n_exp]
    set_seed(seed)

    reg_disentangle = args.reg_disentangle
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    model_name = args.model_name
    dfr_reweighting_frac = args.dfr_reweighting_frac  # 0.2
    DATASET = args.dataset_name

    num_classes = DATASET_INFO[DATASET]['num_classes']
    feature_size = args.feature_size
    finetune_flg = args.finetune_flg  # True #
    reweight_flg = args.reweight_flg  # True
    weight_decay = args.weight_decay
    load_best_model = args.load_best_model

    root_dir = '../data/'
    if DATASET == 'civilcomments':
        data_dir = root_dir + 'datasets/'
    elif DATASET == 'MultiNLI':
        data_dir = root_dir + 'datasets/multinli_bert_features/'
    model_save_path = root_dir + f'models/{DATASET}/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    total_weights = None
    best_model = None
    best_loss = float('inf')
    best_acc_wg = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if finetune_flg == True:
        if load_best_model:
            load_model_path = model_save_path + f'best_model_{bert_version}.pth'
        else:
            load_model_path = model_save_path + f'final_model_{bert_version}.pth'
        best_model_path = model_save_path + f'best_model_{bert_version}_{cpns_version}_{reweight_version}.pth'
        final_model_path = model_save_path + f'final_model_{bert_version}_{cpns_version}_{reweight_version}.pth'
        load_local_model = True
        reg_causal = args.reg_causal
        disentangle_en = False
        counterfactual_en =  False #True
    else:
        best_model_path = model_save_path + f'best_model_{bert_version}.pth'
        final_model_path = model_save_path + f'final_model_{bert_version}.pth'
        load_local_model = False
        reg_causal = 0
        disentangle_en = False
        counterfactual_en = False

    if reweight_flg == True:
        gamma = args.gamma_reweight
    else:
        gamma = 0

    project_name = wandb_project_name(DATASET, bert_version, cpns_version, reweight_version)
    exp_name = wandb_exp_name(DATASET, bert_version, cpns_version, reweight_version, n_exp, lr, feature_size,
                              reg_disentangle, reg_causal, gamma)
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,  # "robust-learning",
        name=exp_name,  # f"disentangle_cov_reg_{bert_version}_{cpns_version}_{reweight_version}",
        # notes="Bert with the regularization of the covariance matrix of the input of the last layer",
        # notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later",
        notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later, reweights the CE loss",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": model_name,
            "dataset": DATASET,
            "epochs": n_epochs,
            "batch_size": batch_size,
            "dfr_reweighting_frac": dfr_reweighting_frac,
            "Regularization coefficient": reg_disentangle,
            "Bert feature size": feature_size,
            "Causal Regularization coefficient": reg_causal,
            "gamma": gamma,
            "seed": seed,
            "weight_decay": weight_decay,
        }
    )
    wandb.define_metric("epoch")
    wandb.define_metric("Train Loss", step_metric='epoch')
    wandb.define_metric("Train Accuracy", step_metric='epoch')
    wandb.define_metric("Validation Loss", step_metric='epoch')
    wandb.define_metric("Validation Accuracy", step_metric='epoch')
    wandb.define_metric("Best Validation Loss", step_metric='epoch')
    wandb.define_metric("Best Validation Accuracy", step_metric='epoch')
    wandb.define_metric("Best Validation Worst Group Accuracy", step_metric='epoch')

    model_config = BertConfig.from_pretrained(model_name, num_labels=num_classes)
    model = BertClassifierWithCovReg(model_name, num_labels=num_classes, feature_size=feature_size, device=device,
                                     reg=reg_disentangle, reg_causal=reg_causal, disentangle_en=disentangle_en,
                                     counterfactual_en=counterfactual_en, config=model_config).to(device)

    if load_local_model:
        model.load_state_dict(torch.load(load_model_path, map_location=device))
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if DATASET == 'civilcomments':
        dataset_configs.dataset_defaults[DATASET]['batch_size'] = batch_size
        task_config = SimpleNamespace(
            root_dir=data_dir,
            # batch_size=batch_size,
            dfr_reweighting_drop=True,
            dfr_reweighting_seed=seed,
            dfr_reweighting_frac=dfr_reweighting_frac,
            algorithm='ERM',
            load_featurizer_only=False,
            pretrained_model_path=None,
            **dataset_configs.dataset_defaults[DATASET],
        )
        task_config.model_kwargs = {}
        task_config.model = model_name
    else:
        task_config = SimpleNamespace(
            root_dir=data_dir,
            batch_size=batch_size,
            dataset=DATASET,
            reweight_groups=False,
            target_name='gold_label_random',
            confounder_names=['sentence2_has_negation'],
            model='bert',
            augment_data=False,
            fraction=1.0,
        )
    train_data, val_data, test_data = get_datasets(task_config, DATASET, train=True)

    if args.dfr_reweighting_drop:
        idx = train_data.dataset.indices.copy()
        rng = np.random.default_rng(args.reweighting_seed)
        rng.shuffle(idx)
        n_train = int((1 - args.dfr_reweighting_frac) * len(idx))
        if finetune_flg == True:
            train_idx = idx[n_train:]
        else:
            train_idx = idx[:n_train]

        train_data.dataset = torch.utils.data.dataset.Subset(train_data.dataset.dataset, indices=train_idx)

    loader_kwargs = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True}
    train_loader = get_data_loader(DATASET, train_data, task_config, train=True, **loader_kwargs)
    val_loader = get_data_loader(DATASET, val_data, task_config, train=False, **loader_kwargs)
    test_loader = get_data_loader(DATASET, test_data, task_config, train=False, **loader_kwargs)

    t_total = len(train_loader) * n_epochs
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data

    if reweight_flg == True:
        with torch.no_grad():
            all_train_logits, all_train_y_true = [], []
            model.eval()
            for batch in train_loader:
                input_ids = batch[0][:, :, 0].to(device)
                attention_mask = batch[0][:, :, 1].to(device)
                segment_ids = batch[0][:, :, 2].to(device)
                labels = batch[1].to(device)

                logits, _ = model(input_ids, attention_mask, labels, token_type_ids=segment_ids)

                all_train_logits.append(logits)
                all_train_y_true.append(labels)

            all_train_logits = torch.cat(all_train_logits, axis=0)
            all_train_y_true = torch.cat(all_train_y_true, axis=0)

            total_weights = compute_weights(all_train_logits, all_train_y_true, gamma, True)

    if load_local_model:
        model = model_parameters_freeze(model)

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        total_batches = 0
        batch_start_idx = 0
        batch_end_idx = 0
        for batch in train_loader:
            # inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = batch[0][:, :, 0].to(device)
            attention_mask = batch[0][:, :, 1].to(device)
            segment_ids = batch[0][:, :, 2].to(device)
            labels = batch[1].to(device)
            # print(input_ids.device)

            batch_end_idx = batch_start_idx + len(labels)
            weights = total_weights[batch_start_idx:batch_end_idx] if total_weights is not None else None
            batch_start_idx = batch_end_idx

            logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids,
                                 labels=labels, weights=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()
            optimizer.step()
            accuracy = compute_accuracy(logits, labels)
            total_train_accuracy += accuracy.item()
            total_train_loss += loss.item()
            # print(f"Epoch {epoch}, Loss: {loss.item()}")
            optimizer.zero_grad()

            total_batches += 1
            if total_batches % 50 == 0:
                print(f"Epoch {epoch}, batches {total_batches} , loss = {loss.item()}")

        avg_train_loss = total_train_loss / total_batches
        avg_train_accuracy = total_train_accuracy / total_batches
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_accuracy}")
        wandb.log({"Train Loss": avg_train_loss, "Train Accuracy": avg_train_accuracy, "epoch": epoch})

        model.eval()
        val_group_acc, avg_val_loss, avg_val_accuracy, _ = evaluation_nli(model, val_data.n_groups, val_loader, device)
        acc_wg_val, _ = torch.min(val_group_acc, dim=0)
        print(
            f"Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}, Validation Worst Group Accuracy: {acc_wg_val}")
        # wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, "Validation Accuracy": avg_val_accuracy})
        wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, 'Validation Accuracy': avg_val_accuracy,
                   'Validation Worst Group Accuracy': acc_wg_val})

        torch.save(model.state_dict(), final_model_path)
        if acc_wg_val.item() >= best_acc_wg:
            best_accuracy = avg_val_accuracy
            best_acc_wg = acc_wg_val.item()
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Epoch {epoch}, Best Validation Loss: {best_loss}, Best Validation Accuracy: {best_accuracy}, Best Validation Worst Group Accuracy: {best_acc_wg}")
            print("Saved best model")
            wandb.log({"epoch": epoch, "Best Validation Loss": best_loss, "Best Validation Accuracy": best_accuracy,
                       "Best Validation Worst Group Accuracy": best_acc_wg})

    if load_best_model:
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    test_group_acc, avg_test_loss, avg_test_accuracy, _ = evaluation_nli(model, test_data.n_groups, test_loader, device)
    acc_wg_test, _ = torch.min(test_group_acc, dim=0)
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}, Test Worst Group Acc: {acc_wg_test.item()}")
    wandb.log({"Test Loss": avg_test_loss, "Test Accuracy": avg_test_accuracy})
    wandb.log({'Test Mean Accuracy': avg_test_accuracy, 'Test Worst Group Accuracy': acc_wg_test.item()})
    wandb.finish()


def train_cv(args):
    bert_version = args.bert_version  # 0 # 147 #190 #215 #
    cpns_version = args.cpns_version
    reweight_version = args.reweight_version
    n_exp = args.n_exp

    reg_disentangle = args.reg_disentangle  # 1 #0.1 # 1.0 # 0.5 #
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    model_name = args.model_name

    dfr_reweighting_frac = args.dfr_reweighting_frac  # 0.2
    DATASET = args.dataset_name
    num_classes = DATASET_INFO[DATASET]['num_classes']

    feature_size = args.feature_size
    finetune_flg = args.finetune_flg  # True #
    reweight_flg = args.reweight_flg  # True

    seed = seeds[n_exp]
    set_seed(seed)

    root_dir = '../data/'
    data_dir = os.path.join(root_dir, DATASET_INFO[DATASET]['dataset_path'])
    model_save_path = root_dir + f'models/{DATASET}/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    best_model = None
    best_loss = float('inf')
    best_acc_wg = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    total_weights = None

    if finetune_flg == True:
        output_dir = os.path.join(model_save_path, f'model_{bert_version}_{cpns_version}_{reweight_version}')
        load_model_path = os.path.join(model_save_path, f'model_{bert_version}_0_0')
        # load_model_path = model_save_path + f'best_model_{bert_version}.pth'
        # best_model_path = model_save_path + f'best_model_{bert_version}_{cpns_version}_{reweight_version}.pth'
        load_local_model = True
        reg_causal = args.reg_causal
        disentangle_en = False
        counterfactual_en = True
    else:
        output_dir = os.path.join(model_save_path, f'model_{bert_version}_{cpns_version}_{reweight_version}')
        # best_model_path = model_save_path + f'best_model_{bert_version}.pth'
        load_local_model = False
        reg_causal = 0
        disentangle_en = True
        counterfactual_en = False

    if reweight_flg == True:
        gamma = args.gamma_reweight
    else:
        gamma = 0

    project_name = wandb_project_name(DATASET, bert_version, cpns_version, reweight_version)
    exp_name = wandb_exp_name(DATASET, bert_version, cpns_version, reweight_version, n_exp, lr, feature_size,
                              reg_disentangle, reg_causal, gamma, weight_decay)

    task_config = SimpleNamespace(
        root_dir=data_dir,
        batch_size=batch_size,
        reweight_groups=False,
        augment_data=False,
        fraction=1.0,
        eval_freq=1,  # args.eval_freq,
        save_freq=10,
        data_dir=data_dir,
        dataset='SpuriousDataset',
        data_transform='NoAugWaterbirdsCelebATransform',
        model='imagenet_resnet50_pretrained',
        train_prop=(1 - args.dfr_reweighting_frac),  # -0.2, # 0.8, #
        num_epochs=n_epochs,
        optimizer='sgd_optimizer',  # 'adamw_optimizer', #
        scheduler='constant_lr_scheduler',
        init_lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        max_prop=1.0,
        val_prop=1.0,
        val_size=-1,
        pass_n=0,
        use_wandb=True,
        output_dir=output_dir,  # '../../logs/waterbirds/80_0',
        project=project_name,
        run_name=exp_name
    )

    train_loader, holdout_loaders = utils.get_data(task_config)
    model = ResNet50withCovReg(reduce_dim=feature_size, output_dim=num_classes, device=device, reg=reg_disentangle,
                               reg_causal=reg_causal, disentangle_en=disentangle_en,
                               counterfactual_en=counterfactual_en).to(device)
    if load_local_model:
        model.load_state_dict(torch.load(os.path.join(load_model_path, 'best_checkpoint.pt'), map_location=device))
        # model.load_state_dict(torch.load(os.path.join(load_model_path, 'final_checkpoint.pt'), map_location=device))
        model = model_parameters_freeze(model)

    optimizer = getattr(optimizers, task_config.optimizer)(model, task_config)
    Log = TrainLogger(task_config)
    scheduler = getattr(optimizers, task_config.scheduler)(optimizer, task_config)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    if reweight_flg == True:
        with torch.no_grad():
            all_train_logits, all_train_y_true, all_group_ids = [], [], []
            model.eval()
            for batch in train_loader:
                x, y, group, *_ = batch
                x, y = x.to(device), y.to(device)
                logits, _ = model(x, y)

                all_train_logits.append(logits)
                all_train_y_true.append(y)
                all_group_ids.append(group)

            all_train_logits = torch.cat(all_train_logits, axis=0)
            all_train_y_true = torch.cat(all_train_y_true, axis=0)
            all_group_ids = torch.cat(all_group_ids, axis=0)

            total_weights = compute_weights1(all_train_logits, all_train_y_true, gamma, True, all_group_ids)

    for epoch in range(n_epochs):

        model.train()
        loss_meter = AverageMeter()
        acc_groups = {g_idx: AverageMeter() for g_idx in train_loader.dataset.active_groups}
        batch_start_idx = 0
        batch_end_idx = 0
        for batch in (pbar := tqdm.tqdm(train_loader, bar_format=_bar_format)):
            x, y, group, *_ = batch
            x, y = x.to(device), y.to(device)

            batch_end_idx = batch_start_idx + len(y)
            weights = total_weights[batch_start_idx:batch_end_idx] if total_weights is not None else None
            batch_start_idx = batch_end_idx

            optimizer.zero_grad()
            logits, loss = model(x, y, weights)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss, x.size(0))
            preds = torch.argmax(logits, dim=1)
            utils.update_dict(acc_groups, y, group, logits)
            acc = (preds == y).float().mean()
            text = f"Loss: {loss.item():3f} ({loss_meter.avg:.3f}); Acc: {acc:3f}"
            pbar.set_description(text)
            # break
        # if epoch < 40 and lr > 1e-5:
        scheduler.step()

        Log.logger.info(f"E: {epoch} | L: {loss_meter.avg:2.5e}\n")
        Log.log_train_results_and_save_chkp(epoch, acc_groups, model, optimizer, scheduler)

        if (epoch % task_config.eval_freq == 0) or (epoch == task_config.num_epochs - 1):
            results_dict = eval_model(model, holdout_loaders, device=device)
            Log.log_results_save_chkp(model, epoch, results_dict, finetune_flg)

        # break

    Log.finalize_logging(model)