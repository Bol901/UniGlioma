# Input contract

All paths in the example manifests are invented. No original patient manifests are distributed.

## Volumes and spatial alignment

Inputs are HDF5 files, not DICOM or NIfTI. Prepare spatially co-registered volumes before using this code. This repository does not provide DICOM ingestion, registration, brain extraction, or clinical de-identification. Each volume must contain:

| H5 key | Shape / meaning |
|---|---|
| `image/iso1mm` | 3D DHW floating-point image in `[0,1]` |
| `brain_mask/iso1mm` | Same uncropped shape; explicit nonempty binary integer/bool brain mask |
| `meta/iso1mm/affine` | 4×4 affine mapping stored DHW indices to world coordinates |
| `meta/iso1mm/spacing` | Length-three positive spacing in stored array-axis order |
| `meta/native/spacing` | Original acquisition spacing where known; not invented from the resampled grid |

The inherited array convention is DHW with index directions inferior, posterior, right. Supply geometry that actually matches the stored array; do not replace a converter's reorientation by an assumed transpose. The configured crop/pad is centered and is applied consistently to images, labels and masks. The example image shape is `192×192×144`; VAE latents are `16×24×24×18`. Missing or malformed brain masks are errors, not intensity-threshold fallbacks. Labels must share the image's spatial grid.

Native-mode data require `image/native` and matching `meta/native/*` and mask metadata when needed. The example training uses `iso1mm`. The supplied preparation and affine conventions do not automatically make arbitrary H5 files compatible.

## Manifests

`train.json` and `val.json` are lists of records with `images: [{path, modality}]` and an optional `label_path`. Use one record per target modality, with all modalities of one study in the same directory; the loader uses the directory to group available modalities. `images[0]` is the target anchor. Relative image paths resolve under `h5_root`; relative label paths under `mmdit.seg_head.label_h5_root`.

MRI modality IDs: `0=T1`, `1=T1c`, `2=T2`, `3=FLAIR`. Generated segmentation uses target ID `4`; ID `5` is the null conditioning modality. For missing-modality synthesis, inputs must exclude the withheld target. For segmentation, valid input modalities must be unique.

Split by patient before creating manifests, retaining every study for a patient in the same split. Only your local tooling should maintain the mapping between study directories and patient identity. Do not publish those manifests or mappings.

The label file stores a discrete integer `image/iso1mm` array with classes `0..K-1`, with `K=4` in the example. Establish and document the tumor-class mapping yourself; different public dataset releases can use different labels. For the inherited inpainting policy, class `1` is the cavity/necrotic-region candidate and class `2` the enhancing-tumor candidate. This is a configurable convention, not an automatic dataset label mapping.

## Acquisition metadata and artifact simulation

The acquisition JSON maps a unique H5 basename without extension to `[quality, view, native_thickness_mm]`, e.g. the fabricated entries in `examples/acquisition.example.json`. Quality is `thin`, `medium`, `thick`, or `unknown`; view is `axial`, `sagittal`, `coronal`, or `unknown`. Base names must be globally unique across studies because lookup is by basename. Record actual acquisition information: resampling a thick image to 1 mm does not make its acquisition thin.

SR, deblurring, de-aliasing and field-of-view completion training require eligible thin targets. The motion simulation reads a pre-brain-extraction full-head sibling and then applies the target's explicit brain mask. De-aliasing uses a full-head target and full-head corruption. A brain-extracted `scan_bet.h5` therefore needs a co-registered `scan.h5` sibling. Do not create a false full-head sibling by copying a brain-extracted image. Include the full-head image in the latent-caching list for de-aliasing. The same full-head path convention applies to synthetic artifact inference.

## Cached latents

Use the supplied preparation scripts with the exact VAE that training and inference will use:

```text
latent/iso1mm/192x192x144/mean
latent/iso1mm/192x192x144/logvar
latent/mask/192x192x144/mean       # in the label H5
```

The scripts write cache version, VAE fingerprint, spatial size and latent shape attributes; mask latents additionally record the class count. Readers validate these attributes before use. Labels are rendered to one channel with `label_id/(K-1)` and encoded by the same VAE. Invalid labels must be remapped before caching. Generative segmentation is decoded and quantized into classes; it is separate from the auxiliary head.

Image restoration constructs corrupted references online and VAE-encodes them. Missing-modality and segmentation references use cached image latents. Relative-resolution training can also construct an online target. Inpainting requires valid labels and an explicit brain mask; pseudo-healthy filling has no paired healthy ground truth inside the original lesion.

## Prompt cache

`scripts/build_prompts.py` enumerates generic instructions through the same composition functions as the dataset. `scripts/encode_prompts.py` stores ragged text features in a directory of `part_*.h5` files. Exact instruction strings are cache keys; a missing key is an error. Use the same encoder snapshot, layer, chat-template settings and text length for training and cache preparation.

The included segmentation catalog covers all 15 nonempty subsets of four MRI modalities, with six generic paraphrases per subset. It is newly authored source material, not a patient-derived caption corpus. Changing this catalog or the prompt templates requires rebuilding the text cache.
