from glob import glob
import os
import collections
import numpy as np
import matplotlib.pyplot as plt
from dataset.data_utils import get_memorized_dataset


def plot_trace_heatmap_differences(
    result, savepdf=None, title=None, xlabel=None, modelname=None
):
    differences = np.clip(result["scores"] - result["low_score"], 0, None)
    low_score = 0  # result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context():
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


folder_pattern = "/projects/nlp/data/constanzam/cross_fact/causal_analysis/facebook__xglm-7.5B/{}*autorr_subset/"
lang1 = "en"
lang2 = "es"

dataset_name = "coastalcph/xlingual_mpararel_autorr"
model_name = "facebook/xglm-7.5B".replace("/", "__")
eval_dir = "/projects/nlp/data/constanzam/cross_fact/eval/"
only_subset = False
ds1 = get_memorized_dataset(
    "coastalcph/xlingual_mpararel_autorr",
    lang1,
    eval_dir,
    model_name,
    only_subset=only_subset,
    filter_trivial=True,
    resample_trivial=True,
)
ds2 = get_memorized_dataset(
    "coastalcph/xlingual_mpararel_autorr",
    lang2,
    eval_dir,
    model_name,
    only_subset=only_subset,
    filter_trivial=True,
    resample_trivial=True,
)
# Note that we're taking only one template per subj-relation, there might be more.
items1 = {"_".join(ex.split("_")[1:3]): ex for ex in ds1["id"]}
items2 = {"_".join(ex.split("_")[1:3]): ex for ex in ds2["id"]}
lang_to_mem_items = {lang1: items1, lang2: items2}
shared_mem_items = list(set(items1.keys()).intersection(items2.keys()))

i = 0
print(shared_mem_items[i])
plots_glob1 = os.path.join(
    folder_pattern.format(lang1), f"plots_resample_trivial/*{shared_mem_items[i]}*.pdf"
)
plots_glob2 = os.path.join(
    folder_pattern.format(lang2), f"plots_resample_trivial/*{shared_mem_items[i]}*.pdf"
)
#!imgcat $plots_glob1 $plots_glob2

i = 0
for lang in [lang1, lang2]:
    for kind in ["attn", "mlp", None]:
        f = os.path.join(
            folder_pattern.format(lang),
            f"cache_hidden_flow/{lang_to_mem_items[lang][shared_mem_items[i]]}{kind}.npz",
        )
        assert len(glob(f)) == 1, f
        numpy_results = np.load(glob(f)[0])
        plot_trace_heatmap_differences(
            numpy_results, f"./plt_output_{lang}_{kind}.pdf", model_name
        )
# !imgcat ./plt_output*.pdf
