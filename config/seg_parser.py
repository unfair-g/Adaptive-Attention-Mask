"""Parser for semantic segmentation continual learning experiments.

Extends the base parser with segmentation-specific arguments.
"""
import argparse
import yaml


class SegmentationParser:
    """Command line parser for segmentation experiments."""
    
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Online Continual Semantic Segmentation with SegFormer"
        )
        
        # Configuration
        parser.add_argument('--config', default=None,
                          help="Path to configuration file")
        
        # Training parameters
        parser.add_argument('--train', dest='train', action='store_true')
        parser.add_argument('--test', dest='train', action='store_false')
        parser.add_argument('--epochs', default=1, type=int,
                          help='Number of epochs per task')
        parser.add_argument('-b', '--batch-size', default=4, type=int,
                          help='Batch size (default: 4 for segmentation)')
        parser.add_argument('--learning-rate', '-lr', default=0.0001, type=float,
                          help='Initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight-decay', '--wd', default=0.01, type=float)
        parser.add_argument('--optim', default='AdamW',
                          choices=['Adam', 'AdamW', 'SGD'])
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--memory-only', '-mo', action='store_true')
        parser.add_argument('--grad-clip', type=float, default=1.0,
                          help='Gradient clipping value')
        
        # Model parameters
        parser.add_argument('--segformer-variant', default='mit_b0',
                          choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'],
                          help='SegFormer backbone variant')
        parser.add_argument('--freeze-encoder', action='store_true',
                          help='Freeze encoder weights')
        parser.add_argument('--pretrained', action='store_true', default=True,
                          help='Use pretrained weights')
        
        # Dataset parameters
        parser.add_argument('--data-root-dir', default='./data/cityscapes/',
                          help='Path to dataset root')
        parser.add_argument('--dataset', '-d', default='cityscapes',
                          choices=['cityscapes', 'bdd100k', 'mapillary'],
                          help='Dataset to use')
        parser.add_argument('--img-size', nargs=2, type=int, default=[512, 1024],
                          help='Image size (H W)')
        parser.add_argument('--n-classes', type=int, default=19,
                          help='Number of classes')
        parser.add_argument('--ignore-index', type=int, default=255,
                          help='Label to ignore in loss/metrics')
        parser.add_argument('--num-workers', '-w', type=int, default=4)
        parser.add_argument('--nb-channels', type=int, default=3)
        
        # Continual learning parameters
        parser.add_argument('--training-type', default='inc',
                          choices=['uni', 'inc', 'blurry'],
                          help='Training type')
        parser.add_argument('--n-tasks', type=int, default=5,
                          help='Number of incremental tasks')
        parser.add_argument('--labels-order', type=int, nargs='+',
                          help='Order of class labels')
        parser.add_argument('--class-order', default='sequential',
                          choices=['sequential', 'random', 'frequency', 'disjoint'],
                          help='How to order classes for incremental learning')
        parser.add_argument('--blurry-scale', type=int, default=500)
        parser.add_argument('--label-mode', default='unknown_as_background',
                          choices=['current_only', 'current_and_old', 
                                   'unknown_as_background', 'all_unknown_as_background'],
                          help='How to handle labels in incremental tasks: '
                               'current_only=only current task labels, others=255(ignore); '
                               'current_and_old=current+old visible, future=255; '
                               'unknown_as_background=current+old visible, unknown=class19(background); '
                               'all_unknown_as_background=only current visible, all others=class19')
        parser.add_argument('--background-class', type=int, default=19,
                          help='Index for background class (default: 19)')
        
        # Memory parameters
        parser.add_argument('--mem-size', type=int, default=100,
                          help='Memory buffer size')
        parser.add_argument('--mem-batch-size', '-mbs', type=int, default=4,
                          help='Number of samples to retrieve from memory')
        parser.add_argument('--buffer', default='seg_reservoir',
                          help='Buffer type')
        parser.add_argument('--drop-method', default='random',
                          choices=['random'])
        parser.add_argument('--mem-iters', type=int, default=1,
                          help='Memory iterations per batch')
        parser.add_argument(
            '--balanced-sampling-mode',
            default='old_classes',
            choices=[
                'old_classes',
                'all_seen',
                'minority',
                'mixed',
                'target_pixel_ratio',
                'random',
            ],
            help='Replay retrieval mode (ER_EMA_Attention_Seg / some ER_Seg configs read from YAML too)',
        )
        parser.add_argument(
            '--mixed-replay-ratios',
            nargs=2,
            type=int,
            metavar=('TPR_W', 'MINORITY_W'),
            default=None,
            help='For mixed mode: integer weights [target_pixel_ratio, minority], e.g. --mixed-replay-ratios 1 1',
        )
        
        # Learner parameters
        parser.add_argument('--learner', default='ER_Seg',
                          help='Learner to use')
        
        # Knowledge distillation parameters
        parser.add_argument('--kd-temperature', type=float, default=2.0)
        parser.add_argument('--alpha-kd', type=float, default=1.0)
        parser.add_argument('--derpp-alpha', type=float, default=0.5)
        parser.add_argument('--derpp-beta', type=float, default=0.5)
        parser.add_argument(
            '--stu-disable-margin-gate',
            action='store_true',
            help='Disable margin boundary gate in cosine STU; use full-map '
                 'adaptive blend w_diff + alpha * w_entropy.',
        )
        
        # EMA parameters
        parser.add_argument('--ema-alpha', type=float, default=0.999)
        parser.add_argument('--eval-teacher', action='store_true',
                          help='Evaluate using EMA teacher')
        
        # Logging parameters
        parser.add_argument('--tag', '-t', default='',
                          help='Experiment tag')
        parser.add_argument('--logs-root', default='./logs/')
        parser.add_argument('--results-root', default='./results/')
        parser.add_argument('--ckpt-root', default='./checkpoints/')
        parser.add_argument('--tb-root', default='./runs/')
        parser.add_argument('--tensorboard', action='store_true')
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--no-wandb', action='store_true')
        parser.add_argument('--wandb-watch', action='store_true')
        parser.add_argument('--sweep', action='store_true')
        
        # Checkpoint parameters
        parser.add_argument('--save-ckpt', action='store_true')
        parser.add_argument(
            '--no-save-all-task-checkpoints',
            dest='save_all_task_checkpoints',
            action='store_false',
            help='With --save-ckpt, only write the final incremental task .pth '
                 '(default: save after every task)',
        )
        parser.set_defaults(save_all_task_checkpoints=True)
        parser.add_argument('--resume', '-r', action='store_true')
        parser.add_argument('--model-state', default=None)
        parser.add_argument('--buffer-state', default=None)
        
        # Multi-run parameters
        parser.add_argument('--n-runs', type=int, default=1)
        parser.add_argument('--start-seed', type=int, default=0)
        parser.add_argument('--run-id', type=int, default=None)
        
        parser.set_defaults(train=True)
        self.parser = parser
        
    def parse(self, arguments=None):
        """Parse arguments and load config if specified."""
        import sys
        
        # Store original command line args for priority checking
        if arguments is not None:
            self._cli_args = arguments
            self.args = self.parser.parse_args(arguments)
        else:
            self._cli_args = sys.argv[1:]
            self.args = self.parser.parse_args()
        
        self.load_config()
        self.check_args()
        return self.args
    
    def load_config(self):
        """Load configuration from YAML file.
        
        Command line arguments take priority over YAML config for critical flags.
        """
        if self.args.config is not None:
            with open(self.args.config, 'r') as f:
                cfg = yaml.safe_load(f)
            
            # Check if --test was explicitly passed in command line
            test_in_cli = '--test' in self._cli_args
            
            for key, value in cfg.items():
                # Convert key format (YAML uses underscore, argparse uses both)
                attr_key = key.replace('-', '_')
                
                # Special case: --test flag should override YAML's train: true
                if attr_key == 'train' and test_in_cli:
                    continue  # Skip, keep command line value (train=False)
                
                setattr(self.args, attr_key, value)
    
    def check_args(self):
        """Validate and adjust arguments."""
        # Set image size based on SegFormer variant
        if self.args.segformer_variant == 'mit_b5':
            if self.args.img_size == [512, 1024]:
                self.args.img_size = [640, 1280]
        
        # Convert img_size to tuple
        if isinstance(self.args.img_size, list):
            self.args.img_size = tuple(self.args.img_size)
        
        # Adjust n_classes if using background class mode
        # Model needs to output 20 classes (19 original + 1 background)
        if 'background' in self.args.label_mode:
            original_n_classes = 19
            self.args.n_classes = 20  # 19 + background
            print(f"Using background class mode: n_classes set to {self.args.n_classes}")
        else:
            original_n_classes = self.args.n_classes
        
        # Set up class order (based on original 19 classes, not including background)
        if self.args.labels_order is None:
            from src.utils.seg_data import get_class_order_cityscapes
            self.args.labels_order = get_class_order_cityscapes(
                order_type=self.args.class_order,
                n_classes=original_n_classes,  # Use original 19 classes for ordering
                seed=self.args.seed
            )

        # Keep n_tasks consistent with increment BEFORE learner init.
        # Otherwise ContinualSegmentationMetrics is built with default n_tasks=5 while
        # main_seg / get_seg_loaders use len(parse_seg_increment_config(...)) (e.g. 11-1 -> 9).
        increment_arg = getattr(self.args, 'increment', None)
        if increment_arg:
            from src.utils.seg_data import parse_seg_increment_config
            task_classes_list = parse_seg_increment_config(
                increment_arg, original_n_classes, self.args.labels_order
            )
            if task_classes_list is not None:
                self.args.n_tasks = len(task_classes_list)
        elif original_n_classes % self.args.n_tasks != 0:
            print(f"Warning: original n_classes ({original_n_classes}) not divisible by "
                  f"n_tasks ({self.args.n_tasks})")

