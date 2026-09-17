# Recent authoritative literature for bone overlap priors

Search date: 2026-08-03.

## Scope and finding

The search focused on 2023--2026 peer-reviewed work in bone/rib X-ray overlap, amodal instance segmentation, occlusion-aware completion, and explicit medical shape priors. No peer-reviewed 2024--2026 paper was found that directly solves adult wrist X-ray multi-label bone-overlap segmentation with a learned per-bone wrist prior. The closest direct medical works are BLS-GAN (AAAI 2025) and Layer Separation Networks (ACM MM 2025), which decompose radiographic layers rather than predict complete per-bone instance masks. The strongest transferable solutions come from amodal segmentation at CVPR/ICCV.

## Ranked papers

| Priority | Paper | Venue | Core mechanism | Direct relevance |
|---:|---|---|---|---|
| 1 | Wang et al., BLS-GAN: A Deep Layer Separation Framework for Eliminating Bone Overlap in Conventional Radiographs | AAAI 2025 | Physics-informed reconstruction plus synthetic pretraining separates superposed bone layers | Very high: exact modality and exact bone-overlap phenomenon, but output is layer images rather than instance masks |
| 2 | Wang et al., Layer Separation: Towards Adjustable Joint Space Width Images Synthesis | ACM MM 2025 | Separates upper bone, lower bone, and soft-tissue layers, then reconstructs radiographs with controllable joint-space width | Very high: finger-joint radiographs and RA are close to RAM; still layer synthesis rather than per-bone amodal segmentation |
| 3 | Zhao et al., Learning with Explicit Topological Priors for Chest X-ray Rib Segmentation | MICCAI 2025 | Connectivity and interactivity violation maps impose plug-and-play constraints on multi-label rib masks | High: exact 2D X-ray multi-bone overlap, but its fixed 2D topology does not recover projection-lost spatial geometry |
| 4 | Zhan et al., Amodal Ground Truth and Completion in the Wild | CVPR 2024 | Uses 3D data to create authentic amodal GT; compares occluder-first completion and diffusion-feature completion | High methodological relevance: supports generating trustworthy hidden-shape targets instead of relying only on pixel penalties |
| 5 | Ozguroglu et al., pix2gestalt: Amodal Segmentation by Synthesizing Wholes | CVPR 2024 | Uses a pretrained diffusion generator to synthesize an unoccluded whole, then obtains the amodal mask | High conceptual relevance for complete-shape generation; natural-image appearance generation is unsafe to apply directly to radiographs |
| 6 | Tai et al., Segment Anything, Even Occluded | CVPR 2025 | Adapts SAM as an amodal decoder and uses a 300K-image synthetic occlusion dataset | High architectural relevance: inference-time completion adapter plus synthetic amodal training |
| 7 | Liu et al., Towards Efficient Foundation Model for Zero-shot Amodal Segmentation | CVPR 2025 | Joint modal/amodal learning, instance focusing, and occlusion-rate-conditioned mixture of experts | High conceptual relevance: overlap strength should condition the completion mechanism rather than use one global lambda |
| 8 | You et al., Learning With Explicit Shape Priors for Medical Image Segmentation | IEEE TMI 2025 | Global and local explicit shape priors inserted into CNN and Transformer backbones | Medium-high: authoritative plug-and-play medical prior, but not designed for occlusion or multi-label overlap |
| 9 | Wyburd et al., Anatomically Plausible Segmentations: Explicitly Preserving Topology through Prior Deformations | Medical Image Analysis 2024 | Deforms a prior with learned topology-preserving fields | Medium: strong medical anatomical prior, but topology preservation alone cannot infer two superposed instance memberships |
| 10 | Gao et al., Coarse-to-Fine Amodal Segmentation with Shape Prior | ICCV 2023 | Vector-quantized coarse amodal shape followed by image-feature convolutional refinement | High implementation value and official code; slightly older but directly supports coarse shape plus local refinement |
| 11 | Tran et al., Amodal Instance Segmentation with Diffusion Shape Prior Estimation | ACCV 2024 | Visible mask, occluder mask and class condition a diffusion shape-prior estimator, followed by amodal refinement | Medium-high: technically close, but ACCV has lower authority than CVPR/ICCV and the domain is natural imagery |
| Evidence | Yang et al., RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis | NeurIPS 2025 Datasets and Benchmarks | Expert multi-label wrist masks and explicit identification of frequent overlap and narrow joint spaces | Direct task evidence and benchmark, not an overlap-prior method |

## Watchlist, not yet equivalent evidence

- RAM-H1200 (arXiv 2026) extends the hand-radiograph benchmark to 1,200 images and full-hand instance masks, but is currently a preprint rather than a peer-reviewed overlap-prior method.
- Amodal SAM and learnable occlusion-guided shape-prototype papers appearing as 2026 preprints are technically aligned, but should not be cited as established authoritative evidence until formally accepted.
- Convex/star-shape priors can be effective on compatible anatomy, but wrist bones are irregular and multi-instance; a compulsory star-shape constraint risks forcing incorrect geometry.

## Design implication for RAM-W600

The literature does not support continuing with a training-only sparse violation loss as the complete prior. A stronger evidence-backed design is:

1. Predict visible/modal per-bone masks.
2. Predict an overlap/occlusion state map and per-pair interaction features.
3. Retrieve or generate an instance-specific complete-shape prior.
4. Apply an inference-time amodal completion/refinement decoder.
5. Preserve the observed projection through reconstruction consistency and preserve visible high-confidence bone pixels.
6. Train with synthetic overlaps or CT-derived projections so hidden-shape supervision is not inferred from ambiguous radiograph intensity alone.

This combines the most relevant ingredients from BLS-GAN, the rib topology paper, CVPR amodal work, and the TMI explicit-shape module without claiming that any single paper already solves RAM-W600.

## Primary links

- BLS-GAN: https://ojs.aaai.org/index.php/AAAI/article/view/32826
- Layer Separation Networks (ACM MM 2025): https://doi.org/10.1145/3746027.3755407
- Rib topology prior: https://papers.miccai.org/miccai-2025/0489-Paper0313.html
- Amodal Ground Truth and Completion: https://openaccess.thecvf.com/content/CVPR2024/html/Zhan_Amodal_Ground_Truth_and_Completion_in_the_Wild_CVPR_2024_paper.html
- pix2gestalt: https://openaccess.thecvf.com/content/CVPR2024/html/Ozguroglu_pix2gestalt_Amodal_Segmentation_by_Synthesizing_Wholes_CVPR_2024_paper.html
- SAMEO: https://openaccess.thecvf.com/content/CVPR2025/html/Tai_Segment_Anything_Even_Occluded_CVPR_2025_paper.html
- SAMBA: https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Towards_Efficient_Foundation_Model_for_Zero-shot_Amodal_Segmentation_CVPR_2025_paper.html
- Explicit Shape Priors (TMI): https://pubmed.ncbi.nlm.nih.gov/39331543/
- TEDS-Net (Medical Image Analysis): https://pubmed.ncbi.nlm.nih.gov/38936222/
- C2F-Seg: https://openaccess.thecvf.com/content/ICCV2023/html/Gao_Coarse-to-Fine_Amodal_Segmentation_with_Shape_Prior_ICCV_2023_paper.html
- AISDiff: https://openaccess.thecvf.com/content/ACCV2024/html/Tran_Amodal_Instance_Segmentation_with_Diffusion_Shape_Prior_Estimation_ACCV_2024_paper.html
- RAM-W600 benchmark: https://proceedings.neurips.cc/paper_files/paper/2025/hash/92059c0ca1f2db0018e49b588d4f05e9-Abstract-Datasets_and_Benchmarks_Track.html
