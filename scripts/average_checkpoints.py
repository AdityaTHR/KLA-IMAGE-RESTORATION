import argparse
import torch

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()

ckpts = [torch.load(x, map_location="cpu", weights_only=False) for x in a.inputs]

key = "model_state_dict" if "model_state_dict" in ckpts[0] else "state_dict"

avg = {}
for k, v in ckpts[0][key].items():
    if torch.is_floating_point(v):
        avg[k] = sum(c[key][k] for c in ckpts) / len(ckpts)
    else:
        avg[k] = v

out = dict(ckpts[-1])
out[key] = avg
torch.save(out, a.output)

print("Saved:", a.output)
