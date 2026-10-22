from multiprocessing import freeze_support
from _common import parser, setup
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, torch
from metaderm.data.derm7pt import Derm7ptDataset
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel
from metaderm.evaluation.gradcam import GradCAM
from metaderm.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from metaderm.utils.io import load_checkpoint, ensure_dir


def main():
    p=parser("Generate Grad-CAM post-hoc explanation baseline")
    p.add_argument("--blackbox-checkpoint",required=True); p.add_argument("--split",default="test",choices=["valid","test"]); p.add_argument("--n",type=int,default=12)
    args=p.parse_args(); cfg,seed,device=setup(args)
    if cfg["model"]["backbone"].lower()!="efficientnet_b3": raise ValueError("Current Grad-CAM baseline targets EfficientNet-B3. Use an EfficientNet-B3 checkpoint.")
    model=BlackBoxDiagnosisModel(cfg).to(device); load_checkpoint(args.blackbox_checkpoint,model,device); model.eval(); ds=Derm7ptDataset(cfg,args.split,train_transform=False); target=model.backbone.features[-1]; cammer=GradCAM(model,target); out=ensure_dir(Path(cfg["project"]["output_root"])/"gradcam"/f"seed{seed}"/args.split); mean=np.array(IMAGENET_MEAN)[:,None,None]; std=np.array(IMAGENET_STD)[:,None,None]
    for i in range(min(args.n,len(ds))):
        item=ds[i]; x=item["image"].unsqueeze(0).to(device); logit,cam=cammer(x); prob=float(torch.sigmoid(logit)[0].cpu()); img=item["image"].numpy()*std+mean; img=np.clip(img.transpose(1,2,0),0,1); heat=cam[0].cpu().numpy(); fig,ax=plt.subplots(figsize=(5,5)); ax.imshow(img); ax.imshow(heat,cmap="jet",alpha=.40); ax.axis("off"); ax.set_title(f"p(melanoma)={prob:.3f}"); fig.tight_layout(); fig.savefig(out/f"case_{int(item['source_index'])}.png",dpi=250,bbox_inches="tight"); plt.close(fig)
    cammer.close(); print("saved:",out)


if __name__ == "__main__":
    freeze_support()
    main()
