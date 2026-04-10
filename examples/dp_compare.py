"""Compare 1 GPU bs=8 vs 6 GPU bs=48 training for 24 steps."""
import torch
import time
from torch.utils.data import DataLoader


MODEL = "google/gemma-2-2b-it"
SEQ = 128
LR = 1e-5
STEPS = 24


def train(state, devices, bs, label):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from infinity.config.training import CPUMasterConfig
    from infinity.model.cpu_master import CPUMasterModel
    from infinity import ChatDataset, collate_fn

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    dataset = ChatDataset(tok, SEQ, dataset_name="alpaca_gpt4_en",
                          dataset_dir="data")

    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager")
    hf.load_state_dict(state)
    config = CPUMasterConfig(
        model_name=MODEL, devices=devices, batch_size=bs,
        dataset_name="alpaca_gpt4_en", dtype=torch.bfloat16,
        attn_implementation="eager", max_seq_len=SEQ,
        learning_rate=LR, num_steps=STEPS)
    model = CPUMasterModel(hf, config)
    del hf

    optimizer = torch.optim.AdamW(model.get_parameters(), lr=LR, weight_decay=0.01)

    g = torch.Generator()
    g.manual_seed(42)
    dl = DataLoader(dataset, batch_size=bs, collate_fn=collate_fn,
                    shuffle=True, generator=g, drop_last=True)
    data_iter = iter(dl)

    losses = []
    step_times = []
    for step in range(STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        t0 = time.perf_counter()
        loss, ntok, timing = model.forward_and_backward(
            batch["input_ids"], batch["attention_mask"], batch["labels"])
        torch.nn.utils.clip_grad_norm_(model.get_parameters(), 1.0)
        optimizer.step()
        if model.shared_state is not None:
            model.shared_state.update_shared_flats()
        model._sync_params_to_gpu()
        model.zero_grad()
        optimizer.zero_grad()
        t1 = time.perf_counter()

        losses.append(loss)
        step_times.append(t1 - t0)
        if (step + 1) % 6 == 0:
            print(f"  [{label}] step {step+1:2d}/{STEPS}  loss={loss:.4f}  time={t1-t0:.2f}s",
                  flush=True)

    avg_time = sum(step_times) / len(step_times)
    tps = bs * SEQ / avg_time
    model.cleanup()
    del model, optimizer
    torch.cuda.empty_cache()
    return losses, avg_time, tps


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    # Shared initial weights
    state = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager"
    ).state_dict()

    print("=== 1 GPU, bs=8, 24 steps ===", flush=True)
    losses_1, t1, tps1 = train(state, [0], 8, "1GPU")

    print(flush=True)
    print("=== 6 GPUs, bs=48 (8/gpu), 24 steps ===", flush=True)
    losses_6, t6, tps6 = train(state, [0, 1, 2, 3, 4, 5], 48, "6GPU")

    print(flush=True)
    print("+------+--------------+--------------+--------------+--------------+")
    print("| Step |   1GPU loss  |   6GPU loss  |     diff     |    diff%     |")
    print("+------+--------------+--------------+--------------+--------------+")
    for i in range(STEPS):
        d = abs(losses_1[i] - losses_6[i])
        dp = d / max(losses_1[i], 1e-10) * 100
        print(f"| {i+1:4d} | {losses_1[i]:12.4f} | {losses_6[i]:12.4f} | {d:12.2e} | {dp:11.1f}% |")
    print("+------+--------------+--------------+--------------+--------------+")

    print()
    print(f"Final loss:  1GPU={losses_1[-1]:.4f}  6GPU={losses_6[-1]:.4f}")
    print(f"Avg step:    1GPU={t1:.3f}s  6GPU={t6:.3f}s")
    print(f"Throughput:  1GPU={tps1:.0f} tok/s  6GPU={tps6:.0f} tok/s  ({tps6/tps1:.2f}x)")
