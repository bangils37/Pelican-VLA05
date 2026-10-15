"""Conditional final-step gradient attention; not CVPR-2021 LRP.

Earlier flow steps are held fixed. Recompute context with gradients and explain
one scalar of the last Euler update. Production inference is never patched.
"""
import copy


def clone_cache(cache):
    """Clone DynamicCache without deepcopy of non-leaf autograd tensors."""
    result = copy.copy(cache)
    if hasattr(cache, "layers"):
        result.layers = [copy.copy(layer) for layer in cache.layers]
        for layer in result.layers:
            layer.keys = layer.keys.clone() if layer.keys is not None else None
            layer.values = layer.values.clone() if layer.values is not None else None
    elif hasattr(cache, "key_cache"):
        result.key_cache = [x.clone() for x in cache.key_cache]
        result.value_cache = [x.clone() for x in cache.value_cache]
    else:
        raise TypeError("unsupported KV cache for gradient capture")
    return result


def capture_final_step(adapter, batch, action_step, action_dim, sign):
    import torch
    model, policy = adapter.policy.model, adapter.policy
    if not 0 <= action_step < model.config.chunk_size:
        raise ValueError("action-step outside chunk")
    action_size = policy.config.output_features["action"].shape[0]
    if not 0 <= action_dim < action_size:
        raise ValueError("action-dim outside checkpoint action dimensions")
    state = policy.prepare_state(batch)
    images, masks = policy._preprocess_images(batch)
    kwargs = dict(images=images, img_masks=masks,
        pixel_values=batch["observation.pixel_values"],
        image_grid_thw=batch["observation.image_grid_thw"],
        lang_tokens=batch["observation.input_ids"], lang_masks=batch["observation.attention_mask"])
    steps = model.config.num_inference_steps
    if steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    # Mirror the model's Euler loop, retaining the final input/time exactly.
    with torch.no_grad():
        context = model._prefill_action_context(**kwargs)
        x = model.sample_noise((state.shape[0],model.config.chunk_size,model.config.max_action_dim),state.device)
        dt = torch.tensor(-1.0/steps,dtype=torch.float32,device=state.device)
        time = torch.tensor(1.0,dtype=torch.float32,device=state.device)
        while time >= -dt/2:
            last_x, last_time = x, time.clone()
            v = model.denoise_step(state, context.valid_tokens, clone_cache(context.key_values),
                context.last_position, x.to(state.dtype), time.expand(state.shape[0]).to(state.dtype))
            x = x + dt*v
            time = time + dt
    del context
    lm = model.vlm.language_model
    original = lm.forward
    params = list(policy.parameters())
    flags = [p.requires_grad for p in params]
    captured = []
    def forward(*args, **kw):
        kw["output_attentions"] = True
        # Explain decoder image tokens; no backprop through frozen vision tower.
        if not kw["inputs_embeds"].requires_grad:
            kw["inputs_embeds"] = kw["inputs_embeds"].detach().requires_grad_(True)
        out = original(*args, **kw)
        if out.attentions is None or any(a is None for a in out.attentions):
            raise RuntimeError("attention capture requires eager attention")
        captured.append(out.attentions)
        return out
    try:
        for p in params:
            p.requires_grad_(False)
        lm.forward = forward
        with torch.enable_grad():
            context = model._prefill_action_context(**kwargs)
            v = model.denoise_step(state,context.valid_tokens,clone_cache(context.key_values),
                context.last_position,last_x.to(state.dtype),last_time.expand(state.shape[0]).to(state.dtype))
            prediction = last_x + dt*v
            target = sign * prediction[0,action_step,action_dim]
            tensors = [a for call in captured for a in call]
            grads = torch.autograd.grad(target,tensors,allow_unused=True)
            if all(g is None for g in grads[-len(captured[-1]):]):
                raise RuntimeError("suffix attention tensors are disconnected from selected action")
            weights, pos = [], 0
            for call in captured:
                fused = []
                for a in call:
                    g = grads[pos]
                    # The final prefix/continuation layer may have no path to
                    # cached K/V used by suffix: its relevance is exactly zero.
                    product = (torch.zeros_like(a, dtype=torch.float32) if g is None else
                               a.detach().float()*g.detach().float())
                    fused.append(product.clamp_min(0).mean(1)[0].cpu())
                    pos += 1
                weights.append(torch.stack(fused).numpy())
            metadata = dict(target_value=float(target.detach()), action_step=action_step,
                action_dim=action_dim, target_sign=sign, target_units="normalized",
                gradient_scope="conditional_final_euler_step", denoise_time=float(last_time))
        return weights, metadata
    finally:
        lm.forward = original
        for p, flag in zip(params,flags):
            p.requires_grad_(flag)
