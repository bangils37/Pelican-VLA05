"""Pelican decoder rollout orchestration and camera projection."""
import numpy as np
from attnvis.core.attribution import cached_rollout
from attnvis.core.types import AttnResult


def explain(adapter, frame, method, action_step=0, action_dim=0, seed=0, target_sign=1, batch_override=None):
    import torch
    from scipy.ndimage import zoom
    from .gradient_capture import capture_final_step
    if method not in ("rollout", "gradient_rollout"):
        raise ValueError(f"unsupported explanation method: {method}")
    if adapter.capture != "auto":
        raise ValueError("rollout requires PELICANVLA_CAPTURE=auto to preserve checkpoint topology")
    if target_sign not in (-1,1):
        raise ValueError("target_sign must be -1 or 1")
    if not 0 <= action_step < adapter.config.chunk_size:
        raise ValueError("action_step outside checkpoint chunk")
    batch = adapter._build_batch(frame) if batch_override is None else batch_override
    device = torch.device(adapter._device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    metadata = dict(method=method, seed=seed, action_step=action_step,
        attribution_scope="language_decoder_image_tokens",
        temporal_input="repeated_single_frame" if batch_override is None else "recorded_history", checkpoint=adapter.ckpt)
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.random.default_generator.manual_seed(seed)
            for d in devices:
                torch.cuda.default_generators[d].manual_seed(seed)
        if method == "gradient_rollout":
            calls, target = capture_final_step(adapter,batch,action_step,action_dim,target_sign)
            metadata.update(target)
        else:
            calls = []
            lm = adapter.policy.model.vlm.language_model
            original = lm.forward
            def forward(*a,**kw):
                kw["output_attentions"] = True
                out = original(*a,**kw)
                if out.attentions is None or any(x is None for x in out.attentions):
                    raise RuntimeError("rollout requires eager attention outputs")
                fused = torch.stack([x[0].float().mean(0).cpu() for x in out.attentions]).numpy()
                # Keep prefix, continuation and most recent denoise only.
                if len(calls) < 3:
                    calls.append(fused)
                else:
                    calls[2] = fused
                return out
            lm.forward = forward
            try:
                with torch.no_grad():
                    adapter.policy.predict_action_chunk(batch,decode_image=False)
            finally:
                lm.forward = original
    if len(calls) != 3:
        raise RuntimeError(f"expected prefix, continuation, suffix; got {len(calls)} calls")
    ids = batch["observation.input_ids"][0].cpu().numpy()
    cols = np.flatnonzero(ids == adapter._img_token_id)
    grid = batch["observation.image_grid_thw"][0].cpu().numpy()
    from .pelicanvla import _slots_enabled, _num_slots
    slots = _num_slots(adapter.config) if _slots_enabled(adapter.config) else 0
    result = cached_rollout(*calls,cols,slots=slots,normalize=method == "rollout")
    action_row = calls[-1].shape[1] - adapter.config.chunk_size + action_step
    vec = result[action_row]
    counts = [int(t)*(int(h)//adapter._merge)*(int(w)//adapter._merge) for t,h,w in grid]
    if len(counts) != 3 or sum(counts) != len(cols):
        raise ValueError("image token grid does not match three camera slots")
    heat, tokens, offset = {}, {}, 0
    for i, count in enumerate(counts):
        t,h,w = map(int,grid[i]); h //= adapter._merge; w //= adapter._merge
        token_grid = vec[offset:offset+count].reshape(t,h,w)
        offset += count
        if i >= len(frame.cams):
            continue
        cam = frame.cams[i]
        tokens[cam] = token_grid
        # Undo the resize-with-pad used by _build_batch before overlaying RGB.
        H,W = adapter._image_resolution
        padded = zoom(token_grid.mean(0),(H/h,W/w),order=1)
        oh,ow = frame.rgb[cam].shape[:2]
        ratio = min(H/oh,W/ow)
        rh,rw = round(oh*ratio),round(ow*ratio)
        top,left = (H-rh)//2,(W-rw)//2
        crop = padded[top:top+rh,left:left+rw]
        heat[cam] = zoom(crop,(oh/rh,ow/rw),order=1).astype(np.float32)
    metadata.update(slots=slots, denoise_selection="last", camera_order=list(frame.cams[:3]))
    return AttnResult(heat=heat,raw=dict(metadata=metadata, token_maps=tokens,
        action_to_image=result[-adapter.config.chunk_size:], image_columns=cols,grid_thw=grid))
