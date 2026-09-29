**DualPriorSeg: Backbone-Agnostic Seam and Overlap Priors for  
Fine-Grained Hand and Wrist Bone Segmentation**

**Anonymous submission**

**Abstract**

Accurate bone segmentation in hand and wrist radiographs is limited by
two complementary structural failure modes: narrow inter-bone gaps can
be incorrectly bridged, while projection overlap can obscure or truncate
bone membership. Most segmentation networks address these errors
implicitly through backbone capacity, making it difficult to preserve
fine separation and recover overlap without redesigning the network for
a particular dataset. We present DualPriorSeg, a backbone-agnostic dual
structural-prior framework that separates these two problems. A seam
prior acts at the input side through an identity-initialized residual
adapter that suppresses spurious connections near predicted gaps, while
an image-conditioned overlap prior guides an output completion refiner
that restores structurally plausible foreground or instance membership.
The priors operate only through standard image and logit interfaces and
therefore do not require access to backbone feature maps. A
supervision-availability mask prevents binary-union annotations from
being treated as true overlap labels, and primary-gradient-anchored
routing limits conflict between structural objectives and the base
segmentation loss. We evaluate the framework on pediatric hand
radiographs (TSRS) and the RAM-W600 wrist benchmark, and test
portability across nnU-Net, TransUNet, and Swin-UMamba. DualPriorSeg
achieves **\[TBD: final TSRS summary\]** on TSRS and **\[TBD: final RAM
summary\]** on RAM-W600, while preserving competitive global overlap
accuracy. These results support structural priors as a reusable
interface for improving fine-grained bone segmentation without
committing to a single backbone.

**Introduction**

Hand and wrist radiographs contain dense collections of small bones
whose projected appearance changes with anatomy, age, positioning, and
disease. Accurate segmentation of these structures is important for
quantitative assessment of skeletal development, joint-space morphology,
bone erosion, deformity, fracture, and other musculoskeletal conditions.
Yet the task remains difficult even for strong medical-segmentation
systems because the errors that matter most are often local: a few
pixels can close a narrow gap between adjacent bones, while superimposed
structures may lose visible boundary evidence and be under-segmented.
These local errors can be clinically and anatomically meaningful even
when the global Dice score changes only slightly.

Modern segmentation backbones such as U-Net variants, nnU-Net,
Transformer hybrids, and state-space models provide strong generic
representations (Ronneberger, Fischer, and Brox 2015; Isensee et al.
2021; Chen et al. 2021; Liu et al. 2024). Their strength, however, does
not explicitly distinguish two opposite structural corrections.
Separation errors require discouraging foreground where two neighboring
bones should remain distinct; overlap errors require restoring
membership where projection ambiguity hides part of a bone. Applying the
same local correction to both situations risks trading one failure mode
for the other.

The two datasets considered in this work expose this complementarity.
TSRS contains pediatric hand radiographs spanning different
developmental stages, where the number and appearance of ossification
centers vary and narrow bone gaps are a recurring segmentation
challenge. RAM-W600 is a public wrist-radiograph benchmark with
pixel-level annotations for 14 bone structures and frequent anatomical
overlap in the carpal region (Yang et al. 2026). The former motivates
explicit separation cues, whereas the latter provides genuine
multi-label supervision for learning overlap-aware completion. A unified
method should exploit both signals without pretending that every dataset
contains the same annotation semantics.

We address this problem with DualPriorSeg, a dual structural-prior
interface placed around an otherwise generic segmentation backbone. The
input-side seam prior is converted into a bounded image residual and is
strongest where separation evidence is high and overlap evidence is
weak. The output-side completion prior refines backbone logits using the
image, the current member probability, context from other members, local
uncertainty, image gradients, and the interaction between seam and
overlap evidence. Neither prior requires internal backbone feature maps.
Consequently, the same modules can be reused with convolutional,
Transformer, or Mamba-style segmentation architectures by replacing only
a thin input/output adapter.

A second design goal is supervision correctness. RAM-W600 supplies
independent bone masks, so pixels belonging to multiple bones provide
observable overlap targets. TSRS is trained as a foreground-union
segmentation task; a union mask cannot reveal whether a foreground pixel
belongs to more than one anatomical instance. We therefore treat overlap
membership on TSRS as unknown rather than negative. The overlap prior
may still transfer as a weak image-conditioned cue, but the overlap loss
is masked when membership is not observed. We further use a preservation
term and parameter-space gradient routing to prevent auxiliary
structural objectives from overwhelming the base segmentation objective.

The resulting framework is intentionally not a new backbone. Its purpose
is to test whether structural information can be attached to multiple
strong segmentation systems in a controlled and reusable way. Following
the evaluation style of recent medical image-analysis work, we combine
overlap and boundary/surface metrics rather than relying on a single
scalar measure (Cheng et al. 2021; Nikolov et al. 2018; Maier-Hein et
al. 2024). Final experiments use matched training controls so that gains
cannot be attributed merely to additional fine-tuning.

Our contributions are as follows:

> **•** We formulate fine-grained hand and wrist bone segmentation as
> two complementary structural correction problems and introduce a
> dual-prior interface with an input-side seam adapter and an
> output-side overlap completion refiner.
>
> **•** We make the framework supervision-aware: true multi-membership
> is learned from RAM-W600, while TSRS overlap membership is explicitly
> treated as unknown; a preservation objective and
> primary-gradient-anchored routing protect the base segmentation
> behavior.
>
> **•** We design the priors around standard image/logit contracts
> rather than internal backbone features and evaluate portability across
> nnU-Net, TransUNet, and Swin-UMamba on two radiographic datasets using
> standard overlap, boundary, and surface metrics. Final quantitative
> gains are reported in **\[TBD: Tables 1–4\]**.

**Related Work**

**Medical image segmentation backbones.**

U-Net established the encoder-decoder architecture with skip connections
as a dominant design for biomedical segmentation (Ronneberger, Fischer,
and Brox 2015). nnU-Net subsequently showed that carefully configured
preprocessing, architecture, training, and inference can provide a
strong self-configuring baseline across heterogeneous medical tasks
(Isensee et al. 2021). Transformer hybrids such as TransUNet combine
convolutional localization with long-range self-attention (Chen et al.
2021), while more recent state-space architectures such as Swin-UMamba
target global context with reduced computational cost (Liu et al. 2024).
These models differ substantially in internal representation, making
them useful for testing whether a prior is genuinely portable rather
than tied to one feature hierarchy.

**Hand and wrist bone segmentation.**

Wrist segmentation is challenging because the carpal region contains
many small, closely spaced and frequently superimposed structures.
RAM-W600 provides a recent public benchmark for this setting, including
1,048 wrist radiographs from 388 patients and pixel-level instance
annotations for 618 images; its segmentation task includes 14 bone masks
and reports both overlap and boundary-sensitive performance (Yang et al.
2026). The official benchmark further shows that strong supervised
medical-segmentation architectures remain important for this anatomy.
Pediatric hand radiographs add a different difficulty: ossification
centers emerge and change with age, so the apparent number, size, and
spacing of structures are not fixed. **\[TBD: insert formal TSRS dataset
citation and final dataset description.\]**

**Structural and boundary-aware priors.**

Overlap-based losses are effective for global mask agreement but can
under-emphasize local contour errors. Boundary IoU was proposed to
increase sensitivity to segmentation boundaries (Cheng et al. 2021), and
surface Dice evaluates contour agreement within a tolerance band
(Nikolov et al. 2018). Boundary-aware losses and topology-oriented
objectives have likewise been explored to encode geometric preferences.
Our focus differs: instead of replacing the standard segmentation
objective, we supply explicit, spatially localized structural priors
around a generic backbone. One prior discourages false bridges, while
the other encourages completion under projection overlap. The priors are
allowed to disagree locally, so optimization must preserve the primary
segmentation gradient rather than treating all objectives symmetrically.

**Gradient conflict in multi-objective learning.**

Auxiliary objectives can improve representation learning but may also
oppose the primary task. Gradient-surgery approaches project conflicting
gradients to reduce destructive interference (Yu et al. 2020).
DualPriorSeg adopts an asymmetric variant suited to a mature
segmentation system: the base segmentation gradient is the anchor,
conflicting auxiliary gradients are projected away from its opposing
component, and each auxiliary contribution is norm-capped. This routing
is a local optimization safeguard, not a guarantee of non-degradation;
final acceptance is determined empirically on held-out validation data.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>[TBD FIGURE 1]<br />
TSRS seam failure / RAM overlap failure / corrected
predictions</strong></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

Figure 1: Representative hand and wrist segmentation failure modes. The
final figure should show a TSRS case with a narrow inter-bone gap and a
RAM-W600 case with projection overlap, together with backbone and
DualPriorSeg predictions. Cases must be selected without test-set
cherry-picking.

**Method**

DualPriorSeg surrounds a segmentation backbone with two complementary
structural modules. The seam branch acts before the backbone and is
intended to reduce false bridges; the overlap branch acts after the
backbone and is intended to recover missing or ambiguous membership.
Figure 2 summarizes the data flow. No ground-truth mask is used in the
inference path.

**Problem Formulation.**

Let x ∈ R^{1×H×W} denote a radiograph and let fθ denote an arbitrary
segmentation backbone. For union segmentation, the backbone predicts
foreground logits z ∈ R^{1×H×W}; for multi-label wrist segmentation, z ∈
R^{C×H×W}, with C bone members. Let s ∈ \[0,1\]^{1×H×W} be a frozen,
GT-free seam prior and o ∈ \[0,1\]^{1×H×W} an image-conditioned overlap
probability. The proposed model computes an adapted image x̃, backbone
logits z, and refined logits ẑ without accessing the internal feature
maps of fθ.

**Seam Prior and Input Adaptation.**

The seam prior represents local evidence that nearby foreground
structures should remain separated. It is produced independently of the
final backbone from image-derived center/geometry proposals and a frozen
relation model. Because pediatric anatomy varies with development, the
prior is image-conditioned rather than tied to a fixed number of bones.
At inference the prior is available without ground truth.

A small residual adapter gφ takes the image, normalized seam prior, and
overlap probability. Its last convolution is zero-initialized so the
complete model starts as an identity transformation. The adapted image
is

| x̃ = x + α tanh(gφ(\[x, s, o\])) ⊙ s ⊙ (1 − o), | \(1\) |
|------------------------------------------------|-------|

where α bounds the maximum image perturbation (0.10 in the current
implementation), \[·\] denotes channel concatenation, and ⊙ is
element-wise multiplication. The factor s(1−o) makes separation
corrections strongest where seam evidence is high but overlap evidence
is low, reducing the risk of suppressing genuinely superimposed anatomy.

**Image-Conditioned Overlap Prior.**

The overlap prior hω predicts a dense probability map from the
radiograph and the seam map. It uses a lightweight convolutional network
operating on a fixed low-resolution representation and is therefore
independent of the feature resolution of the downstream segmentation
backbone. The network is trained on RAM-W600, where independent member
masks make true projection overlap observable. For a multi-label target
y = {y_c}\_{c=1}^C, the overlap target is

| O(p) = 1 \[ Σ_c y_c(p) ≥ 2 \]. | \(2\) |
|--------------------------------|-------|

On TSRS, the foreground-union label does not expose simultaneous
membership. We therefore do not reinterpret foreground pixels as overlap
labels and do not train a strong TSRS overlap objective. Instead, the
RAM-trained predictor is used only as a transferred weak prior in the
forward path. This distinction is central to the annotation semantics of
the framework.

**Output Completion Refinement.**

After the backbone, a shared completion refiner Rψ operates on each
foreground member. For each member it receives the radiograph, that
member’s current probability, summary statistics of the remaining
members, predictive uncertainty, an image-gradient map, and the
interaction context c = s(1−o). The same refiner parameters are reused
across members, including the single foreground member in a
union-segmentation task. A residual output head is initialized to zero,
so the starting prediction exactly matches the backbone.

| ẑ\_c = z_c + Rψ(x, σ(z_c), context\_{−c}, u_c, ∇x, s(1−o)). | \(3\) |
|-------------------------------------------------------------|-------|

For RAM-W600, context from other member probabilities allows the refiner
to recover locally missing membership where another bone is already
predicted in the same projection. For TSRS, the output branch remains
conservative because its overlap membership is unobserved. The final
submission will report both seam-only and dual-prior variants to
quantify whether transferred overlap context helps or interferes with
the TSRS objective.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>[TBD FIGURE 2: FINAL METHOD DIAGRAM]<br />
Image + seam prior → bounded input adapter → generic backbone → logits →
overlap-aware completion refiner<br />
RAM: observed multi-label overlap supervision | TSRS: overlap membership
unknown</strong></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

Figure 2: DualPriorSeg overview. A frozen or jointly fine-tuned generic
backbone is wrapped by an input seam adapter and an output completion
refiner. The image-conditioned overlap prior is learned from observable
RAM multi-label overlap. TSRS does not fabricate overlap labels;
supervision availability gates the overlap loss. Only standard
image/logit interfaces are required, enabling the same prior modules to
wrap different backbones.

**Observable Unified Objective.**

Training combines the original segmentation objective with structural
terms whose validity depends on the available annotation. Let v be a
valid-pixel mask, y the target, p = σ(ẑ), and p_b a frozen backbone
probability used for preservation. The base term combines binary
cross-entropy, soft Dice, and a small foreground-recall penalty. The
separation region is the seam corridor restricted to observed
background, while the overlap completion region is active only when
multi-membership is observed.

| L = L_base + λ_s L_sep + q_o λ_o L_ov + λ_p L_pres, | \(4\) |
|-----------------------------------------------------|-------|

where q_o = 1 for RAM-W600 and q_o = 0 for TSRS overlap membership.
L_sep penalizes predicted foreground in seam-supported background. L_ov
encourages positive member logits in true multi-membership pixels.
L_pres penalizes deviations from the baseline probability outside
structural target regions, reducing broad changes that could improve a
local prior objective while degrading ordinary segmentation. In the
current pilot implementation, λ_s and λ_o are ramped from 0 to 0.035 and
λ_p to 0.10; final values will be selected using the predefined
validation protocol and then frozen before test evaluation.

**Primary-Gradient-Anchored Routing.**

To reduce destructive interference, we compute gradients in parameter
space. Let g_b be the gradient of L_base and g_a the gradient of one
auxiliary term. If the auxiliary gradient conflicts with the base
gradient, we remove its component opposing g_b:

| g′\_a = g_a − (g_aᵀ g_b / \|\|g_b\|\|²) g_b, if g_aᵀ g_b \< 0. | \(5\) |
|----------------------------------------------------------------|-------|

Otherwise g′\_a = g_a. Each routed auxiliary gradient is then scaled by
its requested weight and capped so its norm is at most a fixed fraction
of \|\|g_b\|\| (0.15 in current experiments). The base gradient itself
is never projected to satisfy an auxiliary objective. This mechanism
reduces local optimization conflict, whereas actual non-degradation is
assessed separately on held-out data.

**Backbone Contract.**

The structural modules consume only image-space priors and output
logits. A backbone adapter therefore needs to implement only the native
image-input contract and conversion of native outputs to
foreground/member logits. No encoder stage, decoder stage, or
feature-channel name is referenced by the prior modules. We evaluate
this interface with nnU-Net as the development backbone and with
TransUNet and Swin-UMamba as architecturally distinct portability tests.
**\[TBD: insert final portability implementation details after
experiments are completed.\]**

**Experiments**

**Setup.**

**Datasets.**

TSRS. We use a pediatric hand-radiograph dataset with per-image
bone/ossification-region annotations and a foreground-union segmentation
target. The final manuscript will report the frozen
train/validation/test split and the formal dataset citation after the
ongoing data-integrity audit: **\[TBD: TSRS citation, total images,
final train/val/test counts\]**. Labels are treated as per-image
anonymous instances for prior construction rather than fixed cross-image
anatomical classes. The primary segmentation task remains binary
foreground segmentation.

RAM-W600. RAM-W600 is a multi-task wrist-radiograph benchmark for
rheumatoid arthritis containing 1,048 images from 388 patients, with
pixel-level instance segmentation annotations for 618 radiographs (Yang
et al. 2026). We follow the official segmentation setup and 14-member
output representation. Model selection uses only the training and
validation portions; the test split is not used for threshold or
hyperparameter selection. **\[TBD: insert final split counts used by our
released experiment manifest.\]**

**Evaluation Protocol and Metrics.**

We intentionally use established segmentation metrics rather than
introducing a new paper-facing score. TSRS is evaluated with Dice, IoU,
precision, recall, Boundary IoU, Boundary F1, surface Dice at 2 and 5
pixels, HD95, and ASSD. RAM-W600 follows its benchmark metrics: DSC,
normalized surface Dice (NSD), volume overlap error (VOE), mean surface
distance (MSD), and relative absolute volume difference (RAVD).
Complementary internal diagnostics for seam and overlap regions are
retained for mechanism analysis but are not required for the primary
performance claim. This multi-metric design follows the recommendation
to combine overlap and boundary/surface measures when segmentation
quality cannot be represented by a single family of metrics (Maier-Hein
et al. 2024).

**Matched Controls.**

Final ablations start from the same native backbone checkpoint and use
matched additional training. For nnU-Net we compare Native-Continue,
Seam-Only, Overlap-Only, and Dual-Prior variants under the same split,
optimizer, schedule, augmentation, seed, checkpoint-selection rule, and
evaluator. This control is necessary because historical development runs
initialized TSRS from an R350 mature checkpoint and RAM from an R332
mature checkpoint; those runs are retained as development evidence but
are not used as the final causal ablation.

**Baselines and Backbone Portability.**

We use nnU-Net as the principal medical segmentation baseline (Isensee
et al. 2021). To test portability, we additionally wrap TransUNet (Chen
et al. 2021) and Swin-UMamba (Liu et al. 2024) with the same prior
modules. For the portability table, each architecture is compared only
against its matched +DualPriorSeg version; the full Seam/Overlap/Dual
ablation is performed on nnU-Net. RAM-W600 comparisons additionally
include the published benchmark systems reported by Yang et al. (2026),
subject to matching the official test protocol.

**Implementation Details.**

The current nnU-Net pilot uses a seam-input residual bound α=0.10, a
one-step output refinement, AdamW optimization, and a low learning rate
of 2×10^−5 for adapter parameters. Auxiliary loss weights are ramped
during early training, and each routed auxiliary gradient is capped at
15% of the base-gradient norm. Identity initialization is verified on
real batches before training. The final experiments will use either
adapter-only or joint-finetuning mode with matched controls; exact batch
size, backbone learning-rate ratio, epoch budget, augmentation, GPU
hardware, and random seeds will be inserted after the final run manifest
is frozen: **\[TBD: final implementation block\]**.

Table 1: Main segmentation results on TSRS and RAM-W600. All values are
final test results unless marked otherwise. Arrows indicate the
preferred direction.

| **Dataset** | **Method**             | **Dice/DSC ↑** | **IoU/VOE** | **Boundary/NSD ↑** | **HD95/MSD ↓** | **ASSD/RAVD ↓** |
|-------------|------------------------|----------------|-------------|--------------------|----------------|-----------------|
| TSRS        | nnU-Net (matched)      | **\[TBD\]**    | **\[TBD\]** | **\[TBD\]**        | **\[TBD\]**    | **\[TBD\]**     |
| TSRS        | DualPriorSeg + nnU-Net | **\[TBD\]**    | **\[TBD\]** | **\[TBD\]**        | **\[TBD\]**    | **\[TBD\]**     |
| RAM-W600    | nnU-Net (matched)      | **\[TBD\]**    | **\[TBD\]** | **\[TBD\]**        | **\[TBD\]**    | **\[TBD\]**     |
| RAM-W600    | DualPriorSeg + nnU-Net | **\[TBD\]**    | **\[TBD\]** | **\[TBD\]**        | **\[TBD\]**    | **\[TBD\]**     |

For TSRS, the detailed final table should additionally report Precision,
Recall, Boundary IoU/F1, Surface Dice 2/5 px, HD95, and ASSD. For
RAM-W600, report the official DSC/NSD/VOE/MSD/RAVD definitions exactly
as in the benchmark.

**Main Results**

Table 1 summarizes the primary two-dataset comparison. On TSRS,
DualPriorSeg changes global overlap from **\[TBD\]** to **\[TBD\]**
while improving the boundary/surface measures by **\[TBD\]**. The
expected pattern is that seam-sensitive contour metrics show a larger
gain than global Dice because the prior acts on a small fraction of
image pixels. On RAM-W600, the method achieves **\[TBD\]** DSC and
**\[TBD\]** NSD, with MSD reduced from **\[TBD\]** to **\[TBD\]**. Final
claims will be phrased according to the completed official test results
rather than the preliminary validation pilots.

The central question is not whether every metric increases by a large
margin, but whether the structural modules provide reproducible gains
while preserving the strong base segmentation. We therefore report all
standard metrics, not only the most favorable subset, and bind each
result row to a single checkpoint epoch and SHA-256 hash in the
experiment manifest.

Table 2: nnU-Net ablation of the two structural priors. The same native
checkpoint and matched training schedule must be used for every row.

| **Dataset** | **Variant**     | **Seam input** | **Overlap output** | **Primary overlap metric** | **Boundary/surface metric** | **Δ vs matched base** |
|-------------|-----------------|----------------|--------------------|----------------------------|-----------------------------|-----------------------|
| TSRS        | Native-Continue | –              | –                  | **\[TBD\]**                | **\[TBD\]**                 | –                     |
| TSRS        | Seam-Only       | ✓              | –                  | **\[TBD\]**                | **\[TBD\]**                 | **\[TBD\]**           |
| TSRS        | Overlap-Only    | –              | ✓                  | **\[TBD\]**                | **\[TBD\]**                 | **\[TBD\]**           |
| TSRS        | Dual-Prior      | ✓              | ✓                  | **\[TBD\]**                | **\[TBD\]**                 | **\[TBD\]**           |
| RAM         | Native-Continue | –              | –                  | **\[TBD\]**                | **\[TBD\]**                 | –                     |
| RAM         | Seam-Only       | ✓              | –                  | **\[TBD\]**                | **\[TBD\]**                 | **\[TBD\]**           |
| RAM         | Overlap-Only    | –              | ✓                  | **\[TBD\]**                | **\[TBD\]**                 | **\[TBD\]**           |
| RAM         | Dual-Prior      | ✓              | ✓                  | **\[TBD\]**                | **\[TBD\]**                 | **\[TBD\]**           |

The final paper should report mean ± standard deviation over the
selected number of seeds (target: at least three) or a paired bootstrap
confidence interval where appropriate.

**Ablations**

**Complementarity of seam and overlap priors.**

Table 2 separates the contribution of the two priors. On TSRS, we expect
the seam-only configuration to provide the clearest structural gain
because true overlap membership is unobserved; the overlap branch is
therefore evaluated as a transferred weak cue rather than assumed to be
beneficial. On RAM-W600, the overlap-only configuration directly
exploits observable multi-membership and should account for most of the
completion gain. The dual-prior model is considered successful when it
preserves the strength of the dominant prior while avoiding a systematic
penalty on the complementary structural objective. Final measured
effects: **\[TBD\]**.

**Observable versus unobservable overlap supervision.**

We ablate the supervision-availability mask by comparing the correct
formulation against **\[TBD: if ethically and scientifically useful, a
controlled diagnostic that incorrectly treats unobserved TSRS overlap as
negative\]**. This experiment is optional and should be included only if
it clarifies the method without creating a misleading training
condition. The primary method always treats TSRS multi-membership as
unknown.

**Gradient routing and preservation.**

To test whether structural objectives destabilize the base model, we
compare naive weighted summation with the primary-gradient-anchored
router and with/without the preservation term. We report the full
standard metric vector and the fraction of auxiliary steps that required
projection. The expected outcome is not guaranteed improvement in every
metric; rather, routing should reduce the frequency and magnitude of
degradations relative to naive multi-objective training. Final results:
**\[TBD\]**.

Table 3: Backbone portability of DualPriorSeg. Each +Prior row reuses
the same structural-prior interface and is compared against a matched
base backbone.

| **Dataset** | **Backbone** | **Base metric** | **+DualPriorSeg** | **Absolute gain** | **Boundary/surface gain** |
|-------------|--------------|-----------------|-------------------|-------------------|---------------------------|
| TSRS        | nnU-Net      | **\[TBD\]**     | **\[TBD\]**       | **\[TBD\]**       | **\[TBD\]**               |
| TSRS        | TransUNet    | **\[TBD\]**     | **\[TBD\]**       | **\[TBD\]**       | **\[TBD\]**               |
| TSRS        | Swin-UMamba  | **\[TBD\]**     | **\[TBD\]**       | **\[TBD\]**       | **\[TBD\]**               |
| RAM-W600    | nnU-Net      | **\[TBD\]**     | **\[TBD\]**       | **\[TBD\]**       | **\[TBD\]**               |
| RAM-W600    | TransUNet    | **\[TBD\]**     | **\[TBD\]**       | **\[TBD\]**       | **\[TBD\]**               |
| RAM-W600    | Swin-UMamba  | **\[TBD\]**     | **\[TBD\]**       | **\[TBD\]**       | **\[TBD\]**               |

Use the same preprocessing, data split, training budget, and
checkpoint-selection rule within each backbone pair. Do not compare a
separately tuned +Prior model against an under-trained base model.

**Backbone Portability**

Table 3 evaluates the main backbone-agnostic claim. The structural
modules are not re-designed for each architecture; only the native
input/output adapter is replaced. DualPriorSeg changes nnU-Net by
**\[TBD\]**, TransUNet by **\[TBD\]**, and Swin-UMamba by **\[TBD\]** on
TSRS, with corresponding RAM-W600 changes of **\[TBD\]**. Consistent
positive or non-inferior behavior across architecturally different
backbones would support the claim that the priors act as a reusable
structural interface rather than as a hidden nnU-Net-specific
modification.

Table 4: RAM-W600 benchmark comparison under the official segmentation
protocol.

| **Method**                       | **DSC ↑**          | **NSD ↑**          | **VOE ↓**          | **MSD ↓**          | **RAVD ↓**         |
|----------------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| U-Net / official benchmark       | **\[TBD/quoted\]** | **\[TBD/quoted\]** | **\[TBD/quoted\]** | **\[TBD/quoted\]** | **\[TBD/quoted\]** |
| TransUNet / official benchmark   | **\[TBD/quoted\]** | **\[TBD/quoted\]** | **\[TBD/quoted\]** | **\[TBD/quoted\]** | **\[TBD/quoted\]** |
| Swin-UMamba / official benchmark | **\[TBD/quoted\]** | **\[TBD/quoted\]** | **\[TBD/quoted\]** | **\[TBD/quoted\]** | **\[TBD/quoted\]** |
| DualPriorSeg + best backbone     | **\[TBD\]**        | **\[TBD\]**        | **\[TBD\]**        | **\[TBD\]**        | **\[TBD\]**        |

Published benchmark rows must be copied from the official RAM-W600 paper
using the same metric definitions. Our final row must come from a frozen
configuration evaluated on the official test split exactly once after
validation selection.

**Comparison with Strong RAM-W600 Baselines**

RAM-W600 provides a direct benchmark against strong supervised
architectures. Table 4 will compare the final DualPriorSeg configuration
with the official results of representative convolutional, Transformer,
and Mamba-based systems. We will use “competitive with state-of-the-art
architectures” when our method lies within the top published range
without dominating every metric, and reserve a stronger state-of-the-art
claim only if the official test result clearly exceeds the published
benchmark under matched definitions. Final result statement:
**\[TBD\]**.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>[TBD FIGURE 3]<br />
Qualitative TSRS and RAM-W600 examples with error overlays</strong></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

Figure 3: Qualitative results. Suggested layout: radiograph, ground
truth, matched backbone prediction, seam-only prediction, dual-prior
prediction, and error overlays. Include both successful and failure
cases. Highlight narrow-gap false bridges on TSRS and projection-overlap
completion on RAM-W600.

**Qualitative Analysis**

Figure 3 should illustrate the mechanism behind the aggregate metrics.
For TSRS, representative examples should show whether a narrow
background corridor remains open without eroding adjacent foreground.
For RAM-W600, examples should show whether the refiner restores the
missing member while avoiding indiscriminate foreground expansion.
Failure cases are equally important: likely errors include
over-suppression when seam evidence overlaps genuine projection
ambiguity and false completion when the transferred overlap prior is
over-confident. **\[TBD: insert final case-level observations after the
test set is frozen.\]**

**Limitations**

DualPriorSeg has several limitations. First, the seam prior is itself an
estimated structure and can be wrong; the method therefore depends on
conservative residual adaptation and preservation. Second, true overlap
supervision is available on RAM-W600 but not on TSRS, so cross-dataset
overlap transfer is weak evidence rather than a directly validated
anatomical label. Third, a backbone-agnostic interface does not imply
identical optimal hyperparameters across backbones, and the final study
must distinguish portability from per-model retuning. Fourth, the
current evaluation is limited to hand and wrist radiographs;
generalization to other projection anatomies remains untested. Finally,
the proposed gradient routing alleviates optimization conflict but
cannot mathematically guarantee non-degradation on unseen data.

**Conclusion**

We presented DualPriorSeg, a dual structural-prior framework for
fine-grained hand and wrist bone segmentation. The method separates two
complementary radiographic failure modes: an input-side seam prior
discourages false bridges, and an output-side overlap prior completes
ambiguous or superimposed bone membership. The priors interact with the
segmentation model only through image and logit interfaces, enabling the
same framework to wrap multiple generic backbones. Supervision
availability prevents binary-union annotations from being misinterpreted
as true overlap labels, while preservation and primary-gradient-anchored
routing reduce destructive auxiliary updates. On TSRS and RAM-W600,
final experiments show **\[TBD: concise final quantitative
conclusion\]**. Cross-backbone results demonstrate **\[TBD: portability
conclusion\]**, and the RAM-W600 official benchmark comparison indicates
**\[TBD: competitiveness statement\]**. These findings suggest that
explicit structural priors can complement increasingly powerful
segmentation backbones when local anatomical separation and projection
overlap remain difficult to learn from the base objective alone.

**References**

Cheng, B.; Girshick, R.; Dollár, P.; Berg, A. C.; and Kirillov, A. 2021.
Boundary IoU: Improving Object-Centric Image Segmentation Evaluation. In
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Chen, J.; Lu, Y.; Yu, Q.; Luo, X.; Adeli, E.; Wang, Y.; Lu, L.; Yuille,
A. L.; and Zhou, Y. 2021. TransUNet: Transformers Make Strong Encoders
for Medical Image Segmentation. arXiv:2102.04306.

Dice, L. R. 1945. Measures of the amount of ecologic association between
species. Ecology, 26(3): 297–302.

Isensee, F.; Jaeger, P. F.; Kohl, S. A. A.; Petersen, J.; and
Maier-Hein, K. H. 2021. nnU-Net: a self-configuring method for deep
learning-based biomedical image segmentation. Nature Methods, 18:
203–211.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.;
Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.-Y.; Dollár, P.; and
Girshick, R. 2023. Segment Anything. In IEEE/CVF International
Conference on Computer Vision (ICCV).

Liu, J.; Yang, H.; Zhou, H.-Y.; Xi, Y.; Yu, L.; Yu, Y.; Liang, Y.; Shi,
G.; Zhang, S.; Zheng, H.; and Wang, S. 2024. Swin-UMamba: Mamba-based
UNet with ImageNet-based pretraining. arXiv:2402.03302.

Ma, J.; He, Y.; Li, F.; Han, L.; You, C.; and Wang, B. 2024. Segment
anything in medical images. Nature Communications, 15: 654.

Maier-Hein, L.; Reinke, A.; Godau, P.; et al. 2024. Metrics reloaded:
recommendations for image analysis validation. Nature Methods, 21:
195–212.

Nikolov, S.; Blackwell, S.; Zverovitch, A.; et al. 2018. Deep learning
to achieve clinically applicable segmentation of head and neck anatomy
for radiotherapy. arXiv:1809.04430.

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-Net: Convolutional
Networks for Biomedical Image Segmentation. In Medical Image Computing
and Computer-Assisted Intervention (MICCAI), 234–241.

Yang, S.; Wang, H.; Fu, Y.; Tian, Y.; Kamishima, T.; Ikebe, M.; Ou, Y.;
and Okutomi, M. 2026. RAM-W600: A Multi-Task Wrist Dataset and Benchmark
for Rheumatoid Arthritis. Advances in Neural Information Processing
Systems, 38 (Datasets and Benchmarks Track).

Yu, T.; Kumar, S.; Gupta, A.; Levine, S.; Hausman, K.; and Finn, C.
2020. Gradient Surgery for Multi-Task Learning. Advances in Neural
Information Processing Systems, 33.

\[TBD\] Formal TSRS dataset reference and annotation protocol citation.

\[TBD\] Any additional hand/wrist segmentation papers used in the final
related-work comparison.
