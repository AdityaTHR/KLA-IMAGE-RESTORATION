#!/usr/bin/env python3
from pathlib import Path
import sys, torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model import BaselineRestorer
m=BaselineRestorer()
x=torch.randn(2,1,128,128)
y=m(x)
assert y.shape==(2,1,256,256), y.shape
print("PASS: model maps [B,1,128,128] -> [B,1,256,256]")
print("Parameters:",sum(p.numel() for p in m.parameters()))
