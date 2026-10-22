from multiprocessing import freeze_support
from _common import parser, setup
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from metaderm.data.derm7pt import make_loader
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.training.collect import collect_concept_outputs
from metaderm.utils.io import load_checkpoint, ensure_dir


def main():
    p=parser("PCA/t-SNE/UMAP analysis of learned visual representations")
    p.add_argument("--concept-checkpoint",required=True); p.add_argument("--split",default="test",choices=["train","valid","test"]); p.add_argument("--umap",action="store_true")
    args=p.parse_args(); cfg,seed,device=setup(args); model=ConceptPerceptionModel(cfg).to(device); load_checkpoint(args.concept_checkpoint,model,device); c=collect_concept_outputs(model,make_loader(cfg,args.split,shuffle=False),device)
    X=c["embeddings"].numpy(); y=c["labels"].numpy().astype(int); idx=c["source_indices"].numpy().astype(int); out=ensure_dir(Path(cfg["project"]["output_root"])/"representation"/f"seed{seed}"/args.split)

    def save_projection(name,coords):
        df=pd.DataFrame({"source_index":idx,"label":y,"x":coords[:,0],"y":coords[:,1]}); df.to_csv(out/f"{name}_coordinates.csv",index=False)
        fig,ax=plt.subplots(figsize=(6,5)); sc=ax.scatter(coords[:,0],coords[:,1],c=y,s=22,alpha=.8); ax.set_xlabel(f"{name.upper()} 1"); ax.set_ylabel(f"{name.upper()} 2"); ax.set_title(f"{name.upper()} representation — {args.split}"); fig.tight_layout(); fig.savefig(out/f"{name}_label.png",dpi=300); plt.close(fig)

    pca=PCA(n_components=2,random_state=seed).fit_transform(X); save_projection("pca",pca)
    if len(X)>=5:
        perplexity=max(2,min(30,(len(X)-1)//3)); tsne=TSNE(n_components=2,random_state=seed,init="pca",learning_rate="auto",perplexity=perplexity).fit_transform(X); save_projection("tsne",tsne)
    if args.umap:
        try:
            import umap
            coords=umap.UMAP(n_components=2,random_state=seed,n_neighbors=min(15,max(2,len(X)-1))).fit_transform(X); save_projection("umap",coords)
        except ImportError:
            print("UMAP requested but umap-learn is not installed. Install: pip install umap-learn")
    print("saved:",out)


if __name__ == "__main__":
    freeze_support()
    main()
