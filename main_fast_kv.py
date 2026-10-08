"""Experimental AutoSign training entry point with batched KV-cached decoding.

``main.py`` and ``main_fast.py`` remain unchanged.  Training uses the same
model, loss, optimizer, and scheduler; only autoregressive validation uses a
specialized cached text-token path after the initial pose-prefill forward.
"""

import argparse
import os
from typing import Optional

import torch
import tqdm
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)

import main as baseline
import main_fast as fast


def _compressed_pose_lengths(transformer, pose_lengths, pose_token_count):
    """Apply the model's Conv1d length formula without recomputing pose features."""
    compressed_lengths = pose_lengths.to(dtype=torch.long).clone()
    if transformer.pose_cnn is not None:
        for layer in transformer.pose_cnn:
            if isinstance(layer, torch.nn.Conv1d):
                kernel_size = layer.kernel_size[0]
                stride = layer.stride[0]
                padding = layer.padding[0]
                dilation = layer.dilation[0]
                compressed_lengths = torch.div(
                    compressed_lengths
                    + 2 * padding
                    - dilation * (kernel_size - 1)
                    - 1,
                    stride,
                    rounding_mode="floor",
                ) + 1
    return compressed_lengths.clamp(min=0, max=pose_token_count)


def _forward_cached_text_token(
    model,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
):
    """Run one new text token against an existing pose-and-text KV cache."""
    transformer = model.transformer
    if transformer._attn_implementation != "eager":
        raise ValueError(
            "The experimental KV path currently supports eager attention only"
        )
    if input_ids.shape[1] != 1:
        raise ValueError("Cached decoding expects exactly one new text token")
    if not past_key_values:
        raise ValueError("past_key_values must be populated for cached decoding")

    token_embeddings = transformer.token_embedding(input_ids)
    position_embeddings = transformer.positional_embedding(position_ids)
    hidden_states = transformer.dropout(token_embeddings + position_embeddings)
    past_length = past_key_values[0][0].size(-2)

    causal_attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask=attention_mask,
        input_shape=(input_ids.size(0), 1),
        inputs_embeds=token_embeddings,
        past_key_values_length=past_length,
    )

    presents = ()
    for hidden_layer, layer_past in zip(
        transformer.hidden_layers, past_key_values
    ):
        outputs = hidden_layer(
            hidden_states,
            layer_past=layer_past,
            attention_mask=causal_attention_mask,
            use_cache=True,
        )
        hidden_states = outputs[0]
        presents += (outputs[1],)

    hidden_states = transformer.layer_norm(hidden_states)
    logits = model.language_model_head(hidden_states)
    return logits, presents


@torch.inference_mode()
def generate_autoregressive_batched_kv(
    model,
    pose_values,
    vocab_info,
    device,
    max_length=20,
    temperature=0.0,
    repetition_penalty=1.0,
    pose_lengths=None,
    cache_stats=None,
):
    """Generate a validation batch with one pose prefill and a text KV cache."""
    model.eval()
    if pose_lengths is None:
        pose_lengths = torch.full(
            (pose_values.size(0),),
            pose_values.size(1),
            dtype=torch.long,
            device=pose_values.device,
        )
    else:
        pose_lengths = pose_lengths.to(
            device=pose_values.device,
            dtype=torch.long,
        )

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
    past_key_values = None
    cached_attention_mask = None
    compressed_pose_lengths = None

    if cache_stats is not None:
        cache_stats.clear()
        cache_stats.update(pose_prefill_calls=0, cached_token_calls=0)

    for _ in range(max_length - 1):
        if past_key_values is None:
            outputs = model(
                pose_values=pose_values,
                pose_lengths=pose_lengths,
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :].clone()
            past_key_values = outputs.past_key_values
            if not past_key_values:
                raise RuntimeError("Model did not return a KV cache during prefill")

            pose_token_count = outputs.logits.size(1) - generated_ids.size(1)
            compressed_pose_lengths = _compressed_pose_lengths(
                model.transformer,
                pose_lengths,
                pose_token_count,
            )
            pose_positions = torch.arange(
                pose_token_count,
                device=device,
            ).unsqueeze(0)
            pose_attention_mask = pose_positions < compressed_pose_lengths.unsqueeze(1)
            cached_attention_mask = torch.cat(
                (
                    pose_attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=torch.bool,
                        device=device,
                    ),
                ),
                dim=1,
            )
            if cache_stats is not None:
                cache_stats["pose_prefill_calls"] += 1
                cache_stats["pose_token_count"] = pose_token_count
        else:
            new_input_ids = generated_ids[:, -1:]
            cached_attention_mask = torch.cat(
                (
                    cached_attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=cached_attention_mask.dtype,
                        device=device,
                    ),
                ),
                dim=1,
            )
            text_token_index = generated_ids.size(1) - 1
            position_ids = (
                compressed_pose_lengths.unsqueeze(1) + text_token_index
            )
            cached_logits, past_key_values = _forward_cached_text_token(
                model,
                input_ids=new_input_ids,
                position_ids=position_ids,
                attention_mask=cached_attention_mask,
                past_key_values=past_key_values,
            )
            next_token_logits = cached_logits[:, -1, :].clone()
            if cache_stats is not None:
                cache_stats["cached_token_calls"] += 1

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

    if cache_stats is not None and past_key_values:
        cache_stats["final_cache_length"] = past_key_values[0][0].size(-2)
        cache_stats["generated_length"] = generated_ids.size(1)
    return generated_ids


def evaluate_model_with_wer_batched_kv(
    model,
    dataloader,
    device,
    vocab_info,
    work_dir,
    epoch,
):
    """Evaluate loss normally and generate WER predictions with a KV cache."""
    if len(dataloader.dataset) == 0:
        raise RuntimeError(
            "Cannot calculate WER: validation dataset is empty. "
            "Fix dev.txt and pose_data.txt instead of treating an empty evaluation as WER 0.0."
        )

    print(f"Starting KV-cached autoregressive evaluation for epoch {epoch + 1}...")
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
                desc="KV-cached Autoregressive Eval",
                ncols=100,
            ):
                inputs = {
                    key: value for key, value in batch.items() if key != "file_path"
                }
                inputs = fast.send_inputs_to_device_fast(inputs, device)
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
                generated_ids = generate_autoregressive_batched_kv(
                    model,
                    pose_values,
                    vocab_info,
                    device,
                    pose_lengths=pose_lengths,
                    max_length=20,
                    temperature=0.0,
                    repetition_penalty=1.0,
                )
                batch_predictions = fast._decode_generated_batch(
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
    print("KV-cached autoregressive evaluation complete:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Validation Loss: {avg_loss:.4f}")
    print(f"  WER: {wer_score:.4f}")
    print(f"  Predictions saved to: {predictions_file}")
    return avg_loss, wer_score


def enhanced_training_pipeline_fast_kv(
    mode,
    gpu_id=0,
    run_id=1,
    epochs=100,
    batch_size=64,
    eval_batch_size: Optional[int] = None,
    use_augmentation=True,
    include_face=False,
    include_z=False,
):
    """Fast training pipeline whose validation generator also uses KV cache."""
    train_loader, dev_loader, vocab_info = fast.setup_training_data_fast(
        mode,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        use_augmentation=use_augmentation,
        include_face=include_face,
        include_z=include_z,
    )
    work_dir = f"./training_outputs_fast_kv/{mode}/run_{run_id}"
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
            inputs = fast.send_inputs_to_device_fast(inputs, device)
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
        print(f"[EPOCH {epoch + 1}] Evaluating with KV-cached WER...")
        val_loss, val_wer = evaluate_model_with_wer_batched_kv(
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
        help="KV-cached evaluation batch size (default: batch_size)",
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
        f"Starting FAST+KV training with mode: {args.mode} on GPU: {args.gpu} "
        f"for {args.num_runs} runs (Epochs: {args.epochs}, "
        f"Batch Size: {args.batch_size}, Eval Batch Size: "
        f"{effective_eval_batch_size})"
    )
    print(
        f"Pose features: face={'enabled' if args.include_face else 'disabled'}, "
        f"coordinates={'xyz' if args.include_z else 'xy'}"
    )

    run_record = fast.create_run_record(
        args,
        program_name="main_fast_kv.py",
        kv_cache_enabled=True,
        output_root="training_outputs_fast_kv",
    )
    fast.save_run_record(run_record)
    run_results = []
    for run in range(1, args.num_runs + 1):
        if args.num_runs > 1:
            print(f"\n{'*' * 60}")
            print(f"*** STARTING RUN {run}/{args.num_runs} ***")
            print(f"{'*' * 60}\n")
        results = enhanced_training_pipeline_fast_kv(
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

    fast.finalize_run_records(run_record, run_results, post_to_discord=True)


if __name__ == "__main__":
    main()
