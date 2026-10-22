import torch
import torch.nn as nn
from metaderm.evaluation.gradcam import GradCAM

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(3,4,3,padding=1)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(4,1)
    def forward(self,x):
        z=torch.relu(self.conv(x)); e=self.pool(z).flatten(1); return {"logit":self.fc(e).squeeze(1)}

def test_gradcam_shape_and_bounds():
    model=Toy(); cam=GradCAM(model,model.conv); x=torch.randn(2,3,16,16); _,heat=cam(x); cam.close()
    assert tuple(heat.shape)==(2,16,16)
    assert float(heat.min())>=0.0
    assert float(heat.max())<=1.000001
