import wandb
import pandas as pd

api = wandb.Api()
run = api.run("/lsdasu-university-of-california-davis/ebft_qa_occupancy/runs/o8kjungd")

# Fetch full history (no row cap)
df = run.history(samples=10000)

print(f"Run: {run.name}")
print(f"Status: {run.state}")
print(f"Total steps logged: {int(df['_step'].max())}")
print()

# --- Training metrics (logged every step) ---
train_cols = [c for c in df.columns if c.startswith("train/")]
key_train_cols = [c for c in train_cols if any(k in c for k in [
    "rl_loss", "kl", "reward", "ce_loss", "occupancy", "actor_lr", "advantage"
])]

if key_train_cols:
    train_df = df[["_step"] + key_train_cols].dropna(subset=key_train_cols, how="all")
    print("=== Training Metrics (every 50 steps) ===")
    print(train_df.iloc[::50].to_string(index=False))
    print()

# --- Eval metrics (logged every eval_steps) ---
eval_cols = [c for c in df.columns if c.startswith("eval/")]
if eval_cols:
    eval_df = df[["_step"] + eval_cols].dropna(subset=eval_cols, how="all")
    print("=== Eval Metrics (all eval checkpoints) ===")
    print(eval_df.to_string(index=False))
    print()

# --- Convergence summary ---
if key_train_cols:
    first = train_df.iloc[0]
    last = train_df.iloc[-1]
    print("=== Convergence Summary ===")
    for col in key_train_cols:
        if col in first and col in last:
            try:
                print(f"  {col:50s}  start={first[col]:.4f}  end={last[col]:.4f}")
            except (TypeError, ValueError):
                pass
