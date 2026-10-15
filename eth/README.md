# eth/: the walking controller

| File | What |
|---|---|
| `policy_force_new.onnx` | ETH's walking policy for the real B2W (an LSTM: velocity command -> 12 leg + 4 wheel targets at 50 Hz), from [leggedrobotics/sru-robot-deployment](https://github.com/leggedrobotics/sru-robot-deployment) (MIT, Fan Yang et al., Robotic Systems Lab, ETH Zurich) |
| `policy_force_new_rslrl.npz` | the same weights converted for RSL-RL, the starting point of our fine-tune (`training/train_eth_ft.py`) |
| `policy_stance_v5_600.pt` | **our fine-tune, the default**: keeps a wide rear stance after turns (`training/PHASE5_ETH_STANCE_FINETUNE.md`) |

`project/eth_loco.py` runs it as the base controller of the fetch tasks. `ETH_POLICY=/workspace/eth/policy_force_new.onnx`
brings back the original.
