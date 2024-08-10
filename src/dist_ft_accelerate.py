import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import os
import sys
import json
import time
import datetime

from accelerate import Accelerator, GradScalerKwargs

from audiomae_ft import get_args_parser, modeling, get_mixup, load_model
from utils.os_utils import load_config

sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.join('submodules', 'AudioMAE'))
from subtrees.AudioMAE.engine_finetune import train_one_epoch, evaluate
from subtrees.AudioMAE.util.misc import NativeScalerWithGradNormCount as NativeScaler
from subtrees.AudioMAE.util.misc import AcceleratorScalerWithGradNormCount as AcceleratorScaler
import subtrees.AudioMAE.util.lr_decay as lrd
from torch.utils.tensorboard import SummaryWriter

from data.dataset import make_dataset


def main(args):

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
    accelerator.init_trackers(f"imu_{linprob}", config=config, init_kwargs={"wandb":{"name":f"{masking2d}_{mask_ratio}"}})


    train_loader, valid_loader, model, optimizer, criterion = load_train_objs(cfg, args)
    load_model(args.finetune, args.eval, model)
    args.nb_classes = cfg['model']['num_classes']
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

    loss_scaler = AcceleratorScaler(accelerator=accelerator)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)

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
            loss_scaler,
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
                    task_name=cfg['task_name']
                )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if args.output_dir and (epoch % args.save_every_epoch == 0 or epoch + 1 == args.epochs):
                accelerator.save_state(output_dir=os.path.join(args.output_dir, "accelerator_state"))

            if epoch >= args.first_eval_ep:
                print(f"Accuracy of the network on the {len(valid_loader) * args.batch_size} test images: {test_stats['acc1']:.1f}%")
                max_accuracy = max(max_accuracy, test_stats["acc1"])
                print(f'Max accuracy: {max_accuracy:.2f}%')
            else:
                test_stats ={'acc1': 0.0, 'acc5': 0.0, 'loss': 0.0}
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


def load_train_objs(cfg, args):

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

    model = modeling(
        seconds=args.seconds,
        matrix_type=args.matrix_type,
        cfg=cfg,
        finetune=args.finetune,
        eval=args.eval
    )

    world_size = args.nodes * args.gpus_per_node
    eff_batch_size = args.batch_size * args.accum_iter * world_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    no_weight_decay_list = model.no_weight_decay()
    if cfg['linprob']:
        no_weight_decay_list = lrd.linprob_parse(model, no_weight_decay_list)

    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay
    )

    # FIXME: Uncomment for debugging
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    # loss_scaler = NativeScaler()

    criterion = BCEWithLogitsLoss()

    return train_loader, valid_loader, model, optimizer, criterion

if __name__ == "__main__":
    parent_args = get_args_parser()
    parent_args.add_argument("--nodes", type=int, required=True)
    parent_args.add_argument("--gpus_per_node", type=int, required=True)

    args = parent_args.parse_args()
    main(args)