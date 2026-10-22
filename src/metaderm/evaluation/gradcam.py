from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model=model; self.target_layer=target_layer; self.activations=None; self.gradients=None
        self._fwd=target_layer.register_forward_hook(self._forward_hook)
    def _forward_hook(self,module,inp,out):
        self.activations=out
        if out.requires_grad:
            out.register_hook(self._capture_gradient)
    def _capture_gradient(self,grad): self.gradients=grad
    def close(self): self._fwd.remove()
    def __call__(self,x):
        self.model.zero_grad(set_to_none=True); out=self.model(x)["logit"]; out.sum().backward(); a=self.activations; g=self.gradients
        if a.ndim != 4: raise ValueError("GradCAM helper expects a convolutional NCHW target layer")
        w=g.mean(dim=(2,3),keepdim=True); cam=(w*a).sum(1,keepdim=True); cam=F.relu(cam); cam=F.interpolate(cam,size=x.shape[-2:],mode="bilinear",align_corners=False).squeeze(1)
        cam=cam-cam.amin(dim=(1,2),keepdim=True); cam=cam/(cam.amax(dim=(1,2),keepdim=True)+1e-8); return out.detach(),cam.detach()
