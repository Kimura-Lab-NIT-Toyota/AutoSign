"""Faster experimental entry point for AutoSign training.

This module intentionally leaves ``main.py`` unchanged.  It reuses the same
dataset, model, optimizer, scheduler, loss, and WER definitions while batching
the expensive autoregressive validation pass and improving host-to-GPU input
delivery.
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from typing import Optional

import torch
import tqdm
from torch.utils.data import DataLoader

import main as baseline


def create_run_record(args, *, program_name, kv_cache_enabled, output_root):
    """Build a complete, JSON-serializable record of the effective run options."""
    effective_eval_batch_size = (
        args.batch_size
        if args.eval_batch_size is None
        else args.eval_batch_size
    )
    return {
        "schema_version": 1,
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "program": program_name,
        "command_line": [program_name, *sys.argv[1:]],
        "output_root": output_root,
        "cli_arguments": vars(args).copy(),
        "effective_configuration": {
            "mode": args.mode,
            "gpu": args.gpu,
            "num_runs": args.num_runs,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": effective_eval_batch_size,
            "augmentation_enabled": not args.disable_augmentation,
            "include_face": args.include_face,
            "include_z": args.include_z,
            "kv_cache_enabled": kv_cache_enabled,
        },
        "dataloader_optimization": {
            "enabled": True,
            "train_num_workers": 10,
            "eval_num_workers": 8,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": True,
            "prefetch_factor": 2,
            "non_blocking_device_transfer": True,
            "train_shuffle": True,
            "eval_shuffle": False,
        },
        "results": None,
    }


def save_run_record(run_record):
    """Persist run configuration even when training later fails or is interrupted."""
    summary_dir = os.path.join(
        run_record["output_root"],
        run_record["effective_configuration"]["mode"],
    )
    os.makedirs(summary_dir, exist_ok=True)
    config_file = os.path.join(summary_dir, "run_config.json")
    with open(config_file, "w", encoding="utf-8") as output:
        json.dump(run_record, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"Run configuration saved to: {config_file}")
    return config_file


def finalize_run_records(run_record, run_results, *, post_to_discord=True):
    """Write WER results, update JSON, and post the summary to Discord."""
    if not run_results:
        raise ValueError("run_results must contain at least one completed run")

    best_wers = [result["best_wer"] for result in run_results]
    statistics_summary = {
        "average_best_wer": sum(best_wers) / len(best_wers),
        "median_best_wer": statistics.median(best_wers),
        "minimum_best_wer": min(best_wers),
        "maximum_best_wer": max(best_wers),
    }
    run_record["results"] = {
        "runs": run_results,
        **statistics_summary,
    }
    config_file = save_run_record(run_record)

    effective = run_record["effective_configuration"]
    dataloader = run_record["dataloader_optimization"]
    summary_dir = os.path.dirname(config_file)
    summary_file = os.path.join(summary_dir, "wer_summary.txt")
    summary_lines = [
        f"Program: {run_record['program']}",
        f"Training Mode: {effective['mode']}",
        f"GPU: {effective['gpu']}",
        f"Total Runs: {effective['num_runs']}",
        f"Epochs: {effective['epochs']}",
        f"Batch Size: {effective['batch_size']}",
        f"Eval Batch Size: {effective['eval_batch_size']}",
        f"Augmentation: {'Enabled' if effective['augmentation_enabled'] else 'Disabled'}",
        f"Include Face: {effective['include_face']}",
        f"Include Z: {effective['include_z']}",
        f"KV Cache: {'Enabled' if effective['kv_cache_enabled'] else 'Disabled'}",
        "DataLoader Optimization: Enabled",
        f"  Train Workers: {dataloader['train_num_workers']}",
        f"  Eval Workers: {dataloader['eval_num_workers']}",
        f"  Pin Memory: {dataloader['pin_memory']}",
        f"  Persistent Workers: {dataloader['persistent_workers']}",
        f"  Prefetch Factor: {dataloader['prefetch_factor']}",
        f"  Non-blocking Transfer: {dataloader['non_blocking_device_transfer']}",
        "",
        "Best WER for each run:",
    ]
    for result in run_results:
        summary_lines.append(
            f"  Run {result['run_id']}: {result['best_wer']:.4f} "
            f"(Epoch {result['best_epoch']})"
        )
    summary_lines.extend(
        (
            "",
            f"Average Best WER: {statistics_summary['average_best_wer']:.4f}",
            f"Median Best WER: {statistics_summary['median_best_wer']:.4f}",
            f"Min Best WER: {statistics_summary['minimum_best_wer']:.4f}",
            f"Max Best WER: {statistics_summary['maximum_best_wer']:.4f}",
        )
    )
    summary_content = "\n".join(summary_lines) + "\n"
    with open(summary_file, "w", encoding="utf-8") as output:
        output.write(summary_content)
    print(f"Summary saved to: {summary_file}")

    if post_to_discord:
        baseline.post_discord(message=f"```\n{summary_content}\n```")
        print("Results posted to Discord.")
    return config_file, summary_file


def send_inputs_to_device_fast(batch, device):
    """Move tensors to the target device, asynchronously when memory is pinned."""
    return {
        key: value.to(device=device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _rebuild_loader(loader, *, batch_size, shuffle):
    """Rebuild a loader with persistent workers and pinned CUDA memory."""
    num_workers = loader.num_workers
    options = {
        "dataset": loader.dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": baseline.custom_collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        options.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(**options)


def setup_training_data_fast(
    mode,
    batch_size=64,
    eval_batch_size: Optional[int] = None,
    use_augmentation=True,
    include_face=False,
    include_z=False,
):
    """Create the baseline datasets, then use faster DataLoader settings."""
    train_loader, dev_loader, vocab_info = baseline.setup_training_data(
        mode,
        batch_size=batch_size,
        use_augmentation=use_augmentation,
        include_face=include_face,
        include_z=include_z,
    )
    if eval_batch_size is None:
        eval_batch_size = batch_size
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be a positive integer")

    train_loader = _rebuild_loader(
        train_loader,
        batch_size=batch_size,
        shuffle=True,
    )
    dev_loader = _rebuild_loader(
        dev_loader,
        batch_size=eval_batch_size,
        shuffle=False,
    )
    return train_loader, dev_loader, vocab_info


@torch.inference_mode()
def generate_autoregressive_batched(
    model,
    pose_values,
    vocab_info,
    device,
    max_length=20,
    temperature=0.0,
    repetition_penalty=1.0,
    pose_lengths=None,
):
    """Greedily generate a whole validation batch autoregressively.

    Finished rows receive EOS padding while unfinished rows continue.  Decoding
    stops at the first EOS, so this is equivalent to stopping each row
    independently as the baseline implementation does.
    """
    model.eval()
    batch_size = pose_values.size(0)
    bos_token = vocab_info["vocab_map"].get("<bos>", 2)
    eos_token = vocab_info["vocab_map"].get("<eos>", 3)

    generated_ids = torch.full(
        (batch_size, 1),
        bos_token,
        dtype=torch.long,
        device=device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        outputs = model(
            pose_values=pose_values,
            pose_lengths=pose_lengths,
            input_ids=generated_ids,
            attention_mask=torch.ones_like(generated_ids),
            use_cache=False,
        )
        next_token_logits = outputs.logits[:, -1, :].clone()

        if repetition_penalty != 1.0:
            next_token_logits = baseline.apply_repetition_penalty(
                next_token_logits,
                generated_ids,
                repetition_penalty,
                vocab_info,
            )

        if temperature > 0:
            probabilities = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        next_token = torch.where(
            finished.unsqueeze(1),
            torch.full_like(next_token, eos_token),
            next_token,
        )
        generated_ids = torch.cat((generated_ids, next_token), dim=1)
        finished |= next_token.squeeze(1).eq(eos_token)
        if bool(finished.all()):
            break

    return generated_ids


def _decode_generated_batch(generated_ids, vocab_info):
    inv_vocab_map = vocab_info["inv_vocab_map"]
    eos_token = vocab_info["vocab_map"].get("<eos>", 3)
    special_tokens = {"<pad>", "<bos>", "<eos>"}
    predictions = []

    for generated_sequence in generated_ids.tolist():
        words = []
        for token_id in generated_sequence:
            if token_id == eos_token:
                break
            word = inv_vocab_map.get(token_id)
            if word is not None and word not in special_tokens:
                words.append(word)
        predictions.append(" ".join(words).strip())
    return predictions


def evaluate_model_with_wer_batched(
    model,
    dataloader,
    device,
    vocab_info,
    work_dir,
    epoch,
):
    """Evaluate validation loss and WER with batched greedy generation."""
    if len(dataloader.dataset) == 0:
        raise RuntimeError(
            "Cannot calculate WER: validation dataset is empty. "
            "Fix dev.txt and pose_data.txt instead of treating an empty evaluation as WER 0.0."
        )

    print(f"Starting batched autoregressive evaluation for epoch {epoch + 1}...")
    model.eval()
    all_predictions = []
    all_ground_truths = []
    total_validation_loss = 0.0
    total_validation_tokens = 0

    predictions_dir = os.path.join(work_dir, "pred_outputs")
    os.makedirs(predictions_dir, exist_ok=True)
    predictions_file = os.path.join(
        predictions_dir,
        f"predictions_autoregressive_epoch_{epoch + 1}.txt",
    )

    with open(predictions_file, "w", encoding="utf-8") as pred_file:
        pred_file.write(f"Epoch {epoch + 1} Autoregressive Predictions\n")
        pred_file.write("=" * 60 + "\n\n")

        with torch.inference_mode():
            for batch in tqdm.tqdm(
                dataloader,
                desc="Batched Autoregressive Eval",
                ncols=100,
            ):
                inputs = {
                    key: value for key, value in batch.items() if key != "file_path"
                }
                inputs = send_inputs_to_device_fast(inputs, device)
                pose_values = inputs["pose_values"]
                pose_lengths = inputs["pose_lengths"]

                validation_outputs = model(**inputs)
                if validation_outputs.loss is None:
                    raise RuntimeError(
                        "Validation loss is unavailable even though labels were provided"
                    )
                batch_validation_loss = validation_outputs.loss.item()

                valid_token_count = int(inputs["attention_mask"][:, 1:].sum().item())
                if valid_token_count > 0:
                    total_validation_loss += batch_validation_loss * valid_token_count
                    total_validation_tokens += valid_token_count

                batch_ground_truths = baseline.decode_labels(
                    inputs["labels"], vocab_info
                )
                generated_ids = generate_autoregressive_batched(
                    model,
                    pose_values,
                    vocab_info,
                    device,
                    pose_lengths=pose_lengths,
                    max_length=20,
                    temperature=0.0,
                    repetition_penalty=1.0,
                )
                batch_predictions = _decode_generated_batch(
                    generated_ids.cpu(), vocab_info
                )

                all_predictions.extend(batch_predictions)
                all_ground_truths.extend(batch_ground_truths)

                output_lines = []
                for prediction, ground_truth in zip(
                    batch_predictions, batch_ground_truths
                ):
                    match = "yes" if prediction.strip() == ground_truth.strip() else "no"
                    output_lines.extend(
                        (
                            f"GT:   {ground_truth}\n",
                            f"Pred: {prediction}\n",
                            f"Match: {match}\n",
                            "-" * 40 + "\n\n",
                        )
                    )
                pred_file.writelines(output_lines)

    if not all_predictions:
        raise RuntimeError("Cannot calculate WER: validation produced no predictions")
    if total_validation_tokens == 0:
        raise RuntimeError("Cannot calculate validation loss: no target tokens found")

    wer_score = baseline.wer_list(all_ground_truths, all_predictions)["wer"]
    correct = sum(
        prediction.strip() == ground_truth.strip()
        for prediction, ground_truth in zip(all_predictions, all_ground_truths)
    )
    accuracy = correct / len(all_predictions)
    avg_loss = total_validation_loss / total_validation_tokens

    model.train()
    print("Batched autoregressive evaluation complete:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Validation Loss: {avg_loss:.4f}")
    print(f"  WER: {wer_score:.4f}")
    print(f"  Predictions saved to: {predictions_file}")
    return avg_loss, wer_score


def enhanced_training_pipeline_fast(
    mode,
    gpu_id=0,
    run_id=1,
    epochs=100,
    batch_size=64,
    eval_batch_size=None,
    use_augmentation=True,
    include_face=False,
    include_z=False,
):
    """Baseline training pipeline with faster input and validation handling."""
    train_loader, dev_loader, vocab_info = setup_training_data_fast(
        mode,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        use_augmentation=use_augmentation,
        include_face=include_face,
        include_z=include_z,
    )
    work_dir = f"./training_outputs_fast/{mode}/run_{run_id}"
    os.makedirs(work_dir, exist_ok=True)

    from autosign.config import AutoSignConfig
    from autosign.model import AutoSignLMHeadModel

    config = AutoSignConfig(
        vocab_size=vocab_info["vocab_size"],
        attn_implementation="eager",
        gpt2_hf_model=None,
        include_face=include_face,
        include_z=include_z,
    )
    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    model = AutoSignLMHeadModel(config).to(device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    scheduler = None
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

    use_cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    best_wer = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []
    val_wer_scores = []
    learning_rates = []

    for epoch in range(epochs):
        print(f"\n{'=' * 50}")
        print(f"[EPOCH {epoch + 1}/{epochs}] Starting...")
        print(f"[EPOCH {epoch + 1}] Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        model.train()

        epoch_loss_sum = torch.zeros((), device=device)
        batch_count = 0
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(progress_bar):
            inputs = {
                key: value for key, value in batch.items() if key != "file_path"
            }
            inputs = send_inputs_to_device_fast(inputs, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_cuda_amp,
            ):
                outputs = model(**inputs)

            scaler.scale(outputs.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            detached_loss = outputs.loss.detach()
            epoch_loss_sum += detached_loss
            batch_count += 1
            if batch_idx % 50 == 0:
                progress_bar.set_postfix(
                    Loss=f"{detached_loss.item():.4f}",
                    LR=f"{optimizer.param_groups[0]['lr']:.2e}",
                    Best_WER=f"{best_wer:.4f}",
                    Scheduler="On" if scheduler else "Off",
                )

        avg_train_loss = (
            epoch_loss_sum.item() / batch_count if batch_count else 0.0
        )
        print(f"[EPOCH {epoch + 1}] Evaluating with batched WER...")
        val_loss, val_wer = evaluate_model_with_wer_batched(
            model,
            dev_loader,
            device,
            vocab_info,
            work_dir,
            epoch,
        )

        if scheduler is not None:
            scheduler.step()

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_wer_scores.append(val_wer)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        print(f"[EPOCH {epoch + 1}] RESULTS:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val WER: {val_wer:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        if val_wer < best_wer:
            best_wer = val_wer
            best_epoch = epoch
            baseline.save_best_model(
                model,
                optimizer,
                epoch + 1,
                avg_train_loss,
                val_loss,
                val_wer,
                work_dir,
            )

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE!")
    print(f"Best WER: {best_wer:.4f} (Epoch {best_epoch + 1})")
    print(f"Best model saved at: {os.path.join(work_dir, 'best_model.pt')}")
    print("=" * 60)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_wer_scores": val_wer_scores,
        "learning_rates": learning_rates,
        "best_wer": best_wer,
        "best_epoch": best_epoch,
        "model": model,
        "vocab_info": vocab_info,
        "scheduler_used": scheduler is not None,
        "work_dir": work_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Autoregressive evaluation batch size (default: batch_size)",
    )
    parser.add_argument("--disable_augmentation", action="store_true")
    parser.add_argument("--include_face", action="store_true")
    parser.add_argument("--include_z", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    effective_eval_batch_size = (
        args.batch_size
        if args.eval_batch_size is None
        else args.eval_batch_size
    )
    print(
        f"Starting FAST training with mode: {args.mode} on GPU: {args.gpu} "
        f"for {args.num_runs} runs (Epochs: {args.epochs}, "
        f"Batch Size: {args.batch_size}, Eval Batch Size: "
        f"{effective_eval_batch_size})"
    )
    print(
        f"Pose features: face={'enabled' if args.include_face else 'disabled'}, "
        f"coordinates={'xyz' if args.include_z else 'xy'}"
    )

    run_record = create_run_record(
        args,
        program_name="main_fast.py",
        kv_cache_enabled=False,
        output_root="training_outputs_fast",
    )
    save_run_record(run_record)
    run_results = []
    for run in range(1, args.num_runs + 1):
        if args.num_runs > 1:
            print(f"\n{'*' * 60}")
            print(f"*** STARTING RUN {run}/{args.num_runs} ***")
            print(f"{'*' * 60}\n")

        results = enhanced_training_pipeline_fast(
            args.mode,
            gpu_id=args.gpu,
            run_id=run,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            use_augmentation=not args.disable_augmentation,
            include_face=args.include_face,
            include_z=args.include_z,
        )
        baseline.plot_training_curves_with_wer(results)
        run_results.append(
            {
                "run_id": run,
                "best_wer": results["best_wer"],
                "best_epoch": results["best_epoch"] + 1,
                "work_dir": results["work_dir"],
            }
        )

    finalize_run_records(run_record, run_results, post_to_discord=True)


if __name__ == "__main__":
    main()
