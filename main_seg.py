"""Main script for online continual semantic segmentation experiments.

This script provides the training and evaluation loop for class-incremental
online continual semantic segmentation using SegFormer on Cityscapes.
"""
import os
import torch
import pandas as pd
import numpy as np
import sys
import logging as lg
import datetime as dt
import random as r
import ssl
import wandb

ssl._create_default_https_context = ssl._create_unverified_context

from src.utils.seg_data import get_seg_loaders
from src.utils.seg_name_match import get_seg_learner as _get_seg_learner_src
try:
    from src_baseline.utils.seg_name_match import get_seg_learner as _get_seg_learner_base
except ImportError:  # pragma: no cover
    _get_seg_learner_base = None


def get_seg_learner(name: str):
    """Prefer ``src_baseline`` registry, then fall back to ``src`` (includes AMD learners)."""

    if _get_seg_learner_base is not None:
        try:
            return _get_seg_learner_base(name)
        except ValueError:
            pass
    return _get_seg_learner_src(name)
from config.seg_parser import SegmentationParser
import warnings
warnings.filterwarnings("ignore")


def main():
    """Main training loop for semantic segmentation CL."""
    runs_mious = []
    runs_fgts = []
    
    parser = SegmentationParser()
    args = parser.parse()

    cf = lg.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch = lg.StreamHandler()
    
    for run_id in range(args.start_seed, args.start_seed + args.n_runs):
        # Re-parse for multiple runs
        args = parser.parse()
        args.run_id = run_id
        
        if args.sweep:
            wandb.init()
            for key in wandb.config.keys():
                setattr(args, key, wandb.config[key])
        
        # Seed initialization
        if args.n_runs > 1:
            args.seed = run_id
        np.random.seed(args.seed)
        r.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        if args.learner is not None:
            learner_class = get_seg_learner(args.learner)
            learner = learner_class(args)
            if args.resume:
                learner.resume(args.model_state, args.buffer_state)
        else:
            raise ValueError("Please select a learner with --learner")
        
        logfile = f'{args.tag}.log'
        if not os.path.exists(args.logs_root):
            os.makedirs(args.logs_root)
        
        ff = lg.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = lg.getLogger()
        fh = lg.FileHandler(os.path.join(args.logs_root, logfile))
        ch.setFormatter(cf)
        fh.setFormatter(ff)
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        if args.verbose:
            logger.setLevel(lg.DEBUG)
            logger.warning("Running in VERBOSE MODE")
        else:
            logger.setLevel(lg.INFO)
        
        lg.info("=" * 60)
        lg.info("=" * 20 + f"RUN N°{run_id} SEED {args.seed}" + "=" * 20)
        lg.info("=" * 60)
        lg.info("Parameters:")
        lg.info("=" * 20)
        lg.info(args)
        
        dataloaders = get_seg_loaders(args)

        if not args.no_wandb and not args.sweep:
            wandb.init(
                project=f"seg_{args.learner}",
                config=args.__dict__
            )
        
        if args.training_type == 'inc':
            original_n_classes = 19

            if not args.train:
                if args.model_state is not None:
                    model_state = args.model_state
                    # Try to infer task_id from checkpoint filename (e.g., ckpt_train2.pth -> 2)
                    import re
                    match = re.search(r'ckpt_train(\d+)\.pth', model_state)
                    if match:
                        eval_task_id = int(match.group(1))
                    else:
                        eval_task_id = args.n_tasks - 1
                    lg.info(f"Test mode: Loading checkpoint from {model_state}")
                    lg.info(f"  Inferred task_id: {eval_task_id}")
                else:
                    eval_task_id = args.n_tasks - 1
                    model_state = os.path.join(
                        args.ckpt_root,
                        f"{args.tag}/{args.run_id}/ckpt_train{eval_task_id}.pth"
                    )
                    lg.info(f"Test mode: Loading checkpoint from default path: {model_state}")
                
                learner.resume(model_state, args.buffer_state)

                increment_arg = getattr(args, 'increment', None)
                if increment_arg is not None:
                    from src.utils.seg_data import parse_seg_increment_config
                    task_classes_list = parse_seg_increment_config(
                        increment_arg, original_n_classes, args.labels_order
                    )
                else:
                    task_classes_list = None
                
                current_idx = 0
                all_seen_classes = []
                for t in range(eval_task_id + 1):
                    if task_classes_list is not None:
                        task_classes = task_classes_list[t]
                    else:
                        base_size = original_n_classes // args.n_tasks
                        remainder = original_n_classes % args.n_tasks
                        task_size = base_size + (1 if t >= args.n_tasks - remainder else 0)
                        task_classes = args.labels_order[current_idx:current_idx + task_size]
                        current_idx += task_size
                    all_seen_classes.extend(task_classes)
                
                lg.info(f"  Seen classes (up to task {eval_task_id}): {all_seen_classes}")
                
                from src.utils.seg_metrics import SegmentationMetrics
                import torch.nn.functional as F
                
                learner.model.eval()
                test_loader = dataloaders['test']
                
                metrics = SegmentationMetrics(
                    n_classes=args.n_classes,
                    ignore_index=args.ignore_index
                )
                
                pred_confidence_threshold = getattr(args, 'pred_confidence_threshold', None)
                
                lg.info("=" * 60)
                lg.info("Evaluating on FULL test set...")
                lg.info("=" * 60)
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_loader):
                        imgs = batch[0].to(learner.device)
                        masks = batch[1].to(learner.device)
                        logits = learner.model(imgs)
                        probs = F.softmax(logits, dim=1)
                        max_probs, preds = probs.max(dim=1)
                        if pred_confidence_threshold is not None and pred_confidence_threshold > 0:
                            low_conf_mask = max_probs < pred_confidence_threshold
                            preds[low_conf_mask] = args.ignore_index
                        
                        metrics.update(preds.cpu(), masks.cpu())
                        
                        if (batch_idx + 1) % 50 == 0:
                            lg.info(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

                results = metrics.get_results()
                per_class_iou = metrics.get_iou()
                
                lg.info("=" * 60)
                lg.info("TEST RESULTS (Full Test Set)")
                lg.info("=" * 60)
                lg.info(f"Overall mIoU: {results['miou']*100:.2f}%")
                lg.info(f"Pixel Accuracy: {results['pixel_acc']*100:.2f}%")
                lg.info(f"Mean Accuracy: {results['mean_acc']*100:.2f}%")
                lg.info("")
                lg.info("Per-class IoU:")
                
                from src.datasets.cityscapes import CITYSCAPES_CLASSES
                for c in all_seen_classes:
                    class_name = CITYSCAPES_CLASSES[c] if c < len(CITYSCAPES_CLASSES) else f"class_{c}"
                    iou = per_class_iou[c]
                    lg.info(f"  {c:2d} {class_name:15s}: {iou*100:5.2f}%")
                
                seen_ious = [per_class_iou[c] for c in all_seen_classes if per_class_iou[c] > 0]
                avg_miou = np.mean(seen_ious) if seen_ious else 0.0
                
                lg.info("")
                lg.info(f"mIoU (seen classes only): {avg_miou*100:.2f}%")
                lg.info("=" * 60)
                
                avg_fgt = 0.0
                runs_mious.append(avg_miou)
                runs_fgts.append(avg_fgt)
                
                continue

            increment_arg = getattr(args, 'increment', None)
            if increment_arg is not None:
                from src.utils.seg_data import parse_seg_increment_config
                task_classes_list = parse_seg_increment_config(
                    increment_arg, original_n_classes, args.labels_order
                )
                n_tasks = len(task_classes_list)
                args.n_tasks = n_tasks
            else:
                task_classes_list = None
                n_tasks = args.n_tasks
                base_size = original_n_classes // n_tasks
                remainder = original_n_classes % n_tasks
            
            current_idx = 0
            for task_id in range(n_tasks):
                if task_classes_list is not None:
                    task_classes = task_classes_list[task_id]
                else:
                    task_size = base_size + (1 if task_id >= n_tasks - remainder else 0)
                    task_classes = args.labels_order[current_idx:current_idx + task_size]
                    current_idx += task_size
                
                lg.info(f"Task {task_id}: Classes {task_classes}")
                learner.before_task(task_id, task_classes=task_classes)
                
                for epoch in range(args.epochs):
                    task_name = f"train{task_id}"
                    learner.train(
                        dataloader=dataloaders[task_name],
                        task_name=task_name,
                        task_id=task_id,
                        dataloaders=dataloaders
                    )
                learner.after_task(task_id)
                learner.before_eval()
                avg_miou, avg_fgt = learner.evaluate(dataloaders, task_id)
                
                if not args.no_wandb:
                    wandb.log({
                        "avg_miou": avg_miou,
                        "avg_fgt": avg_fgt,
                        "task_id": task_id
                    })
                    if args.wandb_watch:
                        wandb.watch(learner.model, learner.criterion, log="all", log_freq=1)
                
                learner.after_eval()
                if args.save_ckpt:
                    save_all = getattr(args, 'save_all_task_checkpoints', True)
                    if save_all or task_id == n_tasks - 1:
                        learner.save(f"ckpt_train{task_id}.pth")
            
            learner.save_results()
            
        elif args.training_type == 'blurry':
            learner.train(dataloaders['train'])
            avg_miou = learner.evaluate_offline(dataloaders, epoch=1)
            avg_fgt = 0
            
            if not args.no_wandb:
                wandb.log({"avg_miou": avg_miou})
            
            learner.save_results_offline()
            
        elif args.training_type == 'uni':
            for epoch in range(args.epochs):
                learner.train(dataloaders['train'], epoch=epoch)
                avg_miou = learner.evaluate_offline(dataloaders, epoch=epoch)
                avg_fgt = 0
                
                if not args.no_wandb:
                    wandb.log({
                        "mIoU": avg_miou,
                        "epoch": epoch
                    })
            
            learner.save_results_offline()
        
        runs_mious.append(avg_miou)
        runs_fgts.append(avg_fgt)
        
        if not args.no_wandb:
            wandb.finish()
    
    if args.n_runs > 1:
        df_miou = pd.DataFrame(runs_mious)
        df_fgt = pd.DataFrame(runs_fgts)
        results_dir = os.path.join(args.results_root, args.tag)
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        lg.info(f"Aggregated results saved in: {results_dir}")
        df_miou.to_csv(os.path.join(results_dir, 'runs_mious.csv'), index=False)
        df_fgt.to_csv(os.path.join(results_dir, 'runs_fgts.csv'), index=False)
        
        lg.info(f"Mean mIoU: {np.mean(runs_mious)*100:.2f}% ± {np.std(runs_mious)*100:.2f}%")
        lg.info(f"Mean Forgetting: {np.mean(runs_fgts)*100:.2f}% ± {np.std(runs_fgts)*100:.2f}%")
    
    sys.exit(0)


if __name__ == '__main__':
    main()

