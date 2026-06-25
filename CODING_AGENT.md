# CODING_AGENT.md

## Project Goal

Build a minimal, modular prototype for a TRELLIS 2.0 based 3D editing pipeline with:

1. Crop-resize image preprocessing for strengthening the edited region signal.
2. TRELLIS 2.0 as the main image-to-3D / latent generation backbone.
3. VS3D as the auxiliary multi-view / view-consistency component.
4. Two-view guidance inspired by SplitFlow.
5. Simple config-based ablation support.

The first version should be simple and runnable. Do not over-engineer the codebase.

---

## Expected Pipeline

The intended pipeline is:

```text
input image + bbox/edit region + edit prompt
    ↓
crop-resize preprocessing
    ↓
TRELLIS 2.0 encode / generate latent 3D representation
    ↓
VS3D auxiliary view generation / view bridge
    ↓
two-view guidance update
    ↓
TRELLIS 2.0 decode / export final 3D result
    ↓
render views + evaluate
```

For the first version, do NOT add super-resolution. Use normal crop + bicubic resize only.

---

## File Structure

Create the following minimal structure:

```text
TrellisCropEdit/
├── configs/
│   ├── base.yaml
│   ├── full.yaml
│   ├── no_crop.yaml
│   ├── no_vs3d.yaml
│   └── no_two_view.yaml
│
├── src/
│   ├── preprocess.py
│   ├── trellis.py
│   ├── vs3d.py
│   ├── guidance.py
│   ├── pipeline.py
│   ├── eval.py
│   └── utils.py
│
├── scripts/
│   ├── run.py
│   └── ablation.py
│
├── data/
│   ├── input/
│   ├── bbox_or_mask/
│   └── prompts/
│
├── outputs/
│   ├── crops/
│   ├── views/
│   ├── results/
│   └── metrics/
│
└── third_party/
    ├── TRELLIS2/
    ├── VS3D/
    └── SplitFlow/
```

---

## Module Responsibilities

### `src/preprocess.py`

Purpose: strengthen the edited region signal before sending the image to TRELLIS.

Implement:

```python
def crop_resize(image, bbox, target_size=518, padding_scale=1.5):
    """
    Crop the edit region with padding and resize it to the target model input size.

    Args:
        image: PIL.Image or ndarray.
        bbox: [x1, y1, x2, y2].
        target_size: final square image size.
        padding_scale: expand bbox before cropping.

    Returns:
        crop_image: resized crop.
        meta: dict containing original bbox, padded bbox, scale info.
    """
```

Important:
- Use bicubic resize.
- Do not use Real-ESRGAN or diffusion SR in v1.
- Save debug crop to `outputs/crops/`.

Also implement:

```python
def identity_preprocess(image):
    """Return the original image without crop-resize for ablation."""
```

---

### `src/trellis.py`

Purpose: adapter wrapper around TRELLIS 2.0.

Implement a clean interface even if the internal calls are initially placeholders.

```python
class TrellisAdapter:
    def __init__(self, config):
        pass

    def encode_or_generate(self, image, prompt=None):
        """
        Send image to TRELLIS 2.0 and obtain latent / 3D representation.
        """
        pass

    def decode_or_export(self, latent, output_dir):
        """
        Decode/export final 3D result.
        Could save mesh, Gaussian, voxel, or rendered asset depending on TRELLIS API.
        """
        pass

    def render_views(self, latent, cameras=None, output_dir=None):
        """
        Render diagnostic views for guidance/evaluation.
        """
        pass
```

Do not modify TRELLIS source code directly. Put TRELLIS code under `third_party/TRELLIS2/` and call it through this adapter.

---

### `src/vs3d.py`

Purpose: adapter wrapper around VS3D.

```python
class VS3DAdapter:
    def __init__(self, config):
        pass

    def generate_aux_views(self, latent_or_image, num_views=2):
        """
        Generate or retrieve auxiliary views used by the guidance module.
        """
        pass

    def disabled_views(self, latent_or_image):
        """
        Return minimal fallback views when VS3D is disabled.
        """
        pass
```

For v1, this can return rendered TRELLIS views or placeholder views if VS3D integration is not complete.

---

### `src/guidance.py`

Purpose: implement two-view guidance inspired by SplitFlow.

```python
class TwoViewGuidance:
    def __init__(self, config):
        pass

    def apply(self, latent, views, edit_prompt):
        """
        Apply two-view guidance to update the latent representation.

        Args:
            latent: TRELLIS latent / 3D representation.
            views: two selected views.
            edit_prompt: text edit instruction.

        Returns:
            updated_latent.
        """
        pass
```

For v1:
- The interface is more important than the final algorithm.
- Implement a dummy identity update first.
- Then add the real guidance logic later.

Also provide:

```python
class IdentityGuidance:
    def apply(self, latent, views=None, edit_prompt=None):
        return latent
```

This is used for `no_two_view.yaml`.

---

### `src/pipeline.py`

Purpose: main orchestration file.

```python
class EditPipeline:
    def __init__(self, config):
        self.config = config

    def run(self, image_path, bbox=None, edit_prompt=None, output_dir="outputs/results"):
        """
        Full editing pipeline:
        1. load image
        2. preprocess image
        3. call TRELLIS
        4. call VS3D
        5. apply two-view guidance
        6. decode/export result
        7. save metadata
        """
```

Expected logic:

```python
if config["preprocess"]["enable_crop"]:
    image, crop_meta = crop_resize(image, bbox, ...)
else:
    image = identity_preprocess(image)

latent = trellis.encode_or_generate(image, prompt=edit_prompt)

if config["vs3d"]["enable"]:
    views = vs3d.generate_aux_views(latent, num_views=2)
else:
    views = vs3d.disabled_views(latent)

if config["guidance"]["enable_two_view"]:
    latent = two_view_guidance.apply(latent, views, edit_prompt)
else:
    latent = identity_guidance.apply(latent)

result = trellis.decode_or_export(latent, output_dir)
```

Save a `metadata.json` containing:
- input path
- bbox
- edit prompt
- config path
- crop metadata
- output paths

---

### `src/eval.py`

Purpose: simple evaluation and logging.

Implement only lightweight metrics first:

```python
def save_run_summary(output_dir, metrics: dict):
    pass
```

Optional later:
- render-view consistency
- CLIP similarity
- LPIPS
- geometry preservation
- edit-locality score

---

### `src/utils.py`

Include:
- image loading/saving
- bbox padding
- config loading
- seed setting
- JSON saving

---

## Config Design

### `configs/base.yaml`

```yaml
seed: 42

model:
  input_size: 518

preprocess:
  enable_crop: true
  target_size: 518
  padding_scale: 1.5
  resize_mode: bicubic

trellis:
  repo_path: third_party/TRELLIS2
  checkpoint: null

vs3d:
  enable: true
  repo_path: third_party/VS3D
  num_views: 2

guidance:
  enable_two_view: true
  method: splitflow_style
  strength: 1.0

output:
  root: outputs
  save_debug: true
```

### `configs/full.yaml`

Same as base. Everything enabled.

```yaml
inherit: base.yaml

preprocess:
  enable_crop: true

vs3d:
  enable: true

guidance:
  enable_two_view: true
```

### `configs/no_crop.yaml`

```yaml
inherit: base.yaml

preprocess:
  enable_crop: false

vs3d:
  enable: true

guidance:
  enable_two_view: true
```

### `configs/no_vs3d.yaml`

```yaml
inherit: base.yaml

preprocess:
  enable_crop: true

vs3d:
  enable: false

guidance:
  enable_two_view: true
```

### `configs/no_two_view.yaml`

```yaml
inherit: base.yaml

preprocess:
  enable_crop: true

vs3d:
  enable: true

guidance:
  enable_two_view: false
```

---

## Scripts

### `scripts/run.py`

Command:

```bash
python scripts/run.py \
  --config configs/full.yaml \
  --image data/input/example.png \
  --bbox 120 80 240 220 \
  --prompt "make the chair back more detailed"
```

Responsibilities:
- parse arguments
- load config
- instantiate `EditPipeline`
- call `pipeline.run(...)`

---

### `scripts/ablation.py`

Command:

```bash
python scripts/ablation.py \
  --image data/input/example.png \
  --bbox 120 80 240 220 \
  --prompt "make the chair back more detailed"
```

Run these configs:

```text
configs/full.yaml
configs/no_crop.yaml
configs/no_vs3d.yaml
configs/no_two_view.yaml
```

Save each result to:

```text
outputs/results/{config_name}/
```

---

## Development Priority

Follow this order:

1. Create file structure.
2. Implement config loading and command-line scripts.
3. Implement `preprocess.py` with crop-resize.
4. Implement `pipeline.py` with dummy TRELLIS / VS3D / guidance outputs.
5. Replace dummy TRELLIS adapter with real TRELLIS 2.0 call.
6. Replace dummy VS3D adapter with real VS3D call.
7. Implement real two-view guidance.
8. Add ablation script and save outputs.
9. Add evaluation metrics.

Do not start from the full algorithm. First make the end-to-end pipeline runnable with stubs.

---

## Important Design Rules

1. Do not directly edit third-party code unless absolutely necessary.
2. All third-party methods should be wrapped by adapters.
3. Every experimental component must be controlled by config.
4. The first version should not include super-resolution.
5. The main ablation is crop-resize vs no crop-resize.
6. Save debug images for every preprocessing step.
7. Keep outputs organized by config name.
8. Prefer readable code over excessive abstraction.

---

## Minimal Acceptance Criteria

The first working version is successful if:

1. `python scripts/run.py` runs without crashing.
2. It saves the crop-resized image.
3. It calls the TRELLIS adapter.
4. It saves a final result or placeholder result.
5. It writes `metadata.json`.
6. `python scripts/ablation.py` runs all four config variants.
7. Each ablation has its own output folder.

The real TRELLIS / VS3D / SplitFlow integrations can be added after the skeleton is stable.
