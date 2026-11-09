"""Command line for the native no-pruner backend."""
import argparse
import json
from dataclasses import replace
from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='SmolVLM-256M native no_pruner lifecycle')
    parser.add_argument('command', choices=['prepare-data', 'prepare-features', 'validate-cache',
        'diagnose-train', 'verify-accumulation', 'train', 'verify-resume-checkpoint', 'evaluate', 'benchmark'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--subset', default='full', choices=['full', 'smoke'])
    parser.add_argument('--stage', default='finetune', choices=['finetune'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--skip-meteor', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--progress-log-every-steps', type=int)
    parser.add_argument('--progress-log-every-samples', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug-numerics', action='store_true')
    parser.add_argument('--debug-samples', type=int, default=1)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error('--batch-size must be positive')
        cfg = replace(cfg, evaluation=replace(cfg.evaluation, eval_batch_size=args.batch_size))
    if args.debug_samples < 1:
        parser.error('--debug-samples must be positive')
    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error('--max-steps must be positive')
        cfg = replace(cfg, training=replace(cfg.training,max_steps=args.max_steps,finetune_max_steps=None))
    for name in ('progress_log_every_steps', 'progress_log_every_samples'):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error('--'+name.replace('_','-')+' must be positive')
    if args.progress_log_every_steps is not None:
        cfg = replace(cfg, training=replace(cfg.training, progress_log_every_steps=args.progress_log_every_steps))
    if args.progress_log_every_samples is not None:
        field = 'benchmark_progress_log_every_samples' if args.command == 'benchmark' else 'eval_progress_log_every_samples'
        cfg = replace(cfg, evaluation=replace(cfg.evaluation, **{field:args.progress_log_every_samples}))
    from . import pipeline
    if args.command == 'prepare-data':
        from ..hf_data import prepare_data
        result = prepare_data(cfg, subset=args.subset, disable_progress=args.no_progress)
    elif args.command == 'prepare-features':
        from .model import build_model_and_processor
        from .cache import prepare_features
        model, processor = build_model_and_processor(cfg)
        import torch
        device = cfg.runtime.device if cfg.runtime.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        result = prepare_features(cfg, model, processor, subset=args.subset, disable_progress=args.no_progress)
    elif args.command == 'validate-cache':
        # Metadata-only validation must not allocate/download pretrained weights.
        from .model import load_processor_and_geometry
        from .cache import validate_cache
        processor, geometry = load_processor_and_geometry(cfg)
        result = validate_cache(cfg, processor, geometry)
    elif args.command == 'diagnose-train':
        result = pipeline.diagnose(cfg, debug_samples=args.debug_samples)
    elif args.command == 'verify-accumulation':
        result = pipeline.verify_accumulation(cfg)
    elif args.command == 'train':
        result = pipeline.train(cfg, resume=args.resume, debug=args.debug,
            debug_numerics=args.debug_numerics, debug_samples=args.debug_samples, disable_progress=args.no_progress)
    elif args.command == 'verify-resume-checkpoint':
        if not args.checkpoint:
            parser.error('--checkpoint is required')
        from .model import build_model_and_processor
        from .checkpoint import read
        model, _ = build_model_and_processor(cfg)
        payload = read(args.checkpoint, model, cfg, training=True)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        result = {'ok': True, 'metadata': payload['metadata']}
    else:
        if not args.checkpoint:
            parser.error('--checkpoint is required')
        result = pipeline.inference(cfg, args.checkpoint, benchmark=args.command == 'benchmark',
            max_samples=args.max_samples, skip_meteor=args.skip_meteor, disable_progress=args.no_progress)
    print(json.dumps(result, default=str, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    main()
