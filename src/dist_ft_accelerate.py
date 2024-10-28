import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader
import os
import sys
import json
import time
import datetime
import pandas as pd

from accelerate import Accelerator, GradScalerKwargs

from audiomae_ft import get_args_parser, modeling, get_mixup, load_model
from utils.os_utils import load_config
from utils.lars import LARS

sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.join('submodules', 'AudioMAE'))
from subtrees.AudioMAE.engine_finetune import train_one_epoch, evaluate
from subtrees.AudioMAE.util.misc import NativeScalerWithGradNormCount as NativeScaler
from subtrees.AudioMAE.util.misc import AcceleratorScalerWithGradNormCount as AcceleratorScaler
import subtrees.AudioMAE.util.lr_decay as lrd
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import timm.optim.optim_factory as optim_factory
from data.dataset import make_dataset

from models.multimodal import custom_late_fusion


def main(args):
    patience = 5  # Number of epochs to wait before stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    kwargs = GradScalerKwargs()
    accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[kwargs], log_with="wandb")
    cfg = load_config(args.config)
    config = {
        **cfg,
        **vars(args)
    }
    linprob = 'linprob' if cfg['linprob'] else 'finetuning'
    masking2d = 'masking2d' if cfg['model']['mask_2d'] else 'nomasking2d'
    mask_ratio = None

    if cfg['model']['mask_2d']:
        mask_ratio_t = args.mask_t_prob
        mask_ratio_f = args.mask_f_prob
        mask_ratio = f"t{mask_ratio_t}_f{mask_ratio_f}"
    else:
        mask_ratio = args.mask_t_prob

    if cfg['linprob']:
        args.mask_t_prob = 0.0
        args.mask_f_prob = 0.0

    tags = ['imu']
    project_name = f"imu_{linprob}"
    if cfg['model']['omnivore_included']:
        # project_name = f"imu_{linprob}_omnivore"
        project_name = f"imu_omni_{linprob}"
        tags.append('omnivore')

    if args.mixup > 0.0:
        tags.append('mixup')
    
    if args.label_balance:
        tags.append('label_balance')

    if not args.finetune:
        tags.append('from_scratch')
    
    if args.interfuse:
        cfg['model']['interfuse'] = args.interfuse
        tags.append('intermediate_fusion')
    else:
        tags.append('late_fusion')

    if cfg['model']['global_pool']:
        tags.append('global_pool')
    else:
        tags.append('cls_token')

    accelerator.init_trackers(project_name, config=config, init_kwargs={"wandb":{"name":f"{masking2d}_{mask_ratio}", "tags":tags}})

    class_labels = None
    training_priors = None
    if args.label_pkl:
        label_file = pd.read_pickle(args.label_pkl)
        label_file = label_file.sort_values(by='act_idx')
        class_labels = label_file['label_name_y'].values.tolist()
        training_priors = label_file['count_x'].apply(lambda x: label_file['count_x'].sum() / x).values.tolist()
        training_priors = torch.tensor(training_priors)

    if args.mixup > 0.0 and args.label_balance:
        train_loader, valid_loader, model, optimizer, criterion = load_train_objs(cfg, args, training_priors=training_priors)
    elif args.mixup > 0.0:
        train_loader, valid_loader, model, optimizer, criterion = load_train_objs(cfg, args)
    elif args.label_balance:
        train_loader, valid_loader, model, optimizer, criterion = load_train_objs(cfg, args, training_priors=training_priors)
    else:
        train_loader, valid_loader, model, optimizer, criterion = load_train_objs(cfg, args)

    args.nb_classes = cfg['model']['num_classes']
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = get_mixup(args)

    # Check to choose if you want to use the SummaryWriter (Tensorboard) or not.
    # Don't use it if you want to log with wandb.
    if accelerator.is_main_process and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # if accelerator.is_main_process:
    #     model = load_model(accelerator, args.finetune, args.eval, model)

    # model, optimizer = accelerator.load_state(output_dir=os.path.join(args.output_dir, "accelerator_state"))
    # accelerator.wait_for_everyone()

    # loss_scaler = AcceleratorScaler(accelerator=accelerator)
    model, optimizer, train_loader, valid_loader, criterion = accelerator.prepare(model, optimizer, train_loader, valid_loader, criterion)

    if accelerator.is_main_process:
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            epoch,
            # device,
            # loss_scaler,
            max_norm=args.clip_grad,
            mixup_fn=mixup_fn,
            log_writer=log_writer,
            accelerator=accelerator,
            args=args,
            task_name=cfg['task_name']
        )
        test_stats = evaluate(
                    data_loader=valid_loader,
                    model=model,
                    accelerator=accelerator,
                    args=args,
                    epoch=epoch,
                    task_name=cfg['task_name'],
                    class_labels=class_labels
                )
        accelerator.wait_for_everyone()
        # # Early stopping
        # val_loss = test_stats['loss']
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= patience:
        #         accelerator.save_state(output_dir=os.path.join(args.output_dir, "accelerator_state"))
        #         print("Early stopping triggered")
        #         break

        if accelerator.is_main_process:
            if args.output_dir and (epoch % args.save_every_epoch == 0 or epoch + 1 == args.epochs):
                accelerator.save_state(output_dir=os.path.join(args.output_dir, "accelerator_state"))

            if epoch >= args.first_eval_ep:
                print(f"Accuracy of the network on the {len(valid_loader) * args.batch_size} test images: {test_stats['acc1']:.1f}%")
                max_accuracy = max(max_accuracy, test_stats["acc1"])
                print(f'Max accuracy: {max_accuracy:.2f}%')
            else:
                test_stats ={
                    'acc1': 0.0,
                    'acc5': 0.0,
                    'valid_loss': 0.0,
                    'mAP': 0.0,
                    'mAUC': 0.0,
                    }
                print(f'too new to evaluate!')
            

            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}

            if args.output_dir:
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
        accelerator.wait_for_everyone()
    accelerator.end_training()


def load_train_objs(cfg, args, training_priors=None):

    if args.dataset == 'wear':
        train_set = make_dataset(
            name=args.dataset,
            is_pretrain=False,
            **cfg['dataset'],
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
        )
        val_set = make_dataset(
            name=args.dataset,
            is_pretrain=False,
            **cfg['dataset'],
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
        )
    else:

        # FIXME: Uncomment for debugging
        # preload = False
        # cfg['dataset_train']['preload'] = preload
        # cfg['dataset_valid']['preload'] = preload

        train_set = make_dataset(
            name=args.dataset,
            is_pretrain=False,
            task_name = cfg['task_name'],
            **cfg['dataset_train'],
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
        )
        print(cfg['dataset_train'])
        val_set = make_dataset(
            name=args.dataset,
            is_pretrain=False,
            task_name = cfg['task_name'],
            **cfg['dataset_valid'],
            **cfg['spectrogram_params'][f'sec_{args.seconds}'][args.matrix_type]
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    cfg['model']['dropout'] = args.drop_path

    is_multimodal = cfg['model']['omnivore_included']
    # if not cfg['linprob'] and is_multimodal:
    #     cfg['model']['classification'] = False  # Return the features of the model
    #     cfg['model']['omnivore_included'] = False

    model = modeling(
        seconds=args.seconds,
        matrix_type=args.matrix_type,
        cfg=cfg,
        finetune=args.finetune,
        eval=args.eval
    )
    model = load_model(args.finetune, args.eval, model, global_pool=cfg['model']['global_pool'], contains_omni=cfg['model']['omnivore_included'], args=args)

    if is_multimodal:
        model = custom_late_fusion(
            model=model,
            in_dim=768 + 1536,
            num_classes=cfg['model']['num_classes'],
            hidden_dims=[1024, 512, 256]
        )

    world_size = args.nodes * args.gpus_per_node
    eff_batch_size = args.batch_size * args.accum_iter * world_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    no_weight_decay_list = model.no_weight_decay()

    if not is_multimodal:
        if args.interfuse:
            # model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
            if cfg['linprob']:
                no_weight_decay_list = lrd.linprob_parse_interfusion(model, no_weight_decay_list)

        elif not cfg['model']['omnivore_included']:
            model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
            if cfg['linprob']:
                no_weight_decay_list = lrd.linprob_parse(model, no_weight_decay_list)

        elif cfg['model']['omnivore_included']:
            model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
            model.omni_classifier = torch.nn.Sequential(torch.nn.BatchNorm1d(model.omni_classifier.in_features, affine=False, eps=1e-6), model.omni_classifier)
            if  cfg['linprob']:
                no_weight_decay_list = lrd.linprob_parse_omni_late_fusion(model, no_weight_decay_list)

    optim
    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
        no_weight_decay_list=no_weight_decay_list,
        layer_decay=args.layer_decay
    )

    # FIXME: Uncomment for debugging
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    optimizer = None

    if cfg['linprob']:
        # optimizer = LARS(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # loss_scaler = NativeScaler()
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

    if args.use_soft:
        criterion = SoftTargetCrossEntropy()
    # elif mixup_active:
    #     if training_priors is not None:
    #         criterion = BCEWithLogitsLoss(weight=training_priors) # works better
    #     else:
    #         criterion = BCEWithLogitsLoss()
    else:
        if training_priors is not None:
            criterion = CrossEntropyLoss(weight=training_priors, label_smoothing=args.smoothing) # works better
        else:
            criterion = CrossEntropyLoss(label_smoothing=args.smoothing)
    # criterion = CrossEntropyLoss()
    # criterion = SoftTargetCrossEntropy()
    return train_loader, valid_loader, model, optimizer, criterion

if __name__ == "__main__":
    parent_args = get_args_parser()
    parent_args.add_argument("--nodes", type=int, required=True)
    parent_args.add_argument("--gpus_per_node", type=int, required=True)
    parent_args.add_argument("--label_balance", action="store_true")
    parent_args.add_argument("--interfuse", action="store_true", default=False)
    parent_args.set_defaults(label_balance=False)
    args = parent_args.parse_args()
    main(args)