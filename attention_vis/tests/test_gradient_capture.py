"""Small real Qwen decoder tests; no checkpoint, GPU or robot needed."""
import unittest
from types import SimpleNamespace
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from attnvis.adapters.gradient_capture import clone_cache, capture_final_step

class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cfg=Qwen3VLTextConfig(vocab_size=20,hidden_size=16,intermediate_size=24,
            num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
            rope_parameters={"rope_type":"default","rope_theta":10000.,"mrope_section":[1,1,2]})
        cfg._attn_implementation='eager'
        self.lm=Qwen3VLTextModel(cfg).eval()
        self.head=torch.nn.Linear(16,2)
        self.model=SimpleNamespace(vlm=SimpleNamespace(language_model=self.lm),
            config=SimpleNamespace(chunk_size=2,max_action_dim=2,num_inference_steps=2),
            _prefill_action_context=self.prefill,denoise_step=self.denoise,
            sample_noise=lambda shape,device: torch.randn(shape,device=device))
        self.config=SimpleNamespace(output_features={'action':SimpleNamespace(shape=(2,))})
    @torch.no_grad()
    def predict_action_chunk(self,b,decode_image=False):
        context=self.prefill()
        x=self.model.sample_noise((1,2,2),torch.device('cpu'))
        for time in (1.,.5):
            x=x-.5*self.denoise(None,None,clone_cache(context.key_values),None,x,time)
        return x,None
    def prepare_state(self,b): return torch.zeros(1,2)
    def _preprocess_images(self,b): return torch.zeros(1),torch.ones(1,dtype=torch.bool)
    def call(self,ids,cache=None):
        n=len(ids); past=0 if cache is None else cache.get_seq_length()
        return self.lm(inputs_embeds=self.lm.embed_tokens(torch.tensor([ids])),
            attention_mask=torch.zeros(1,1,n,past+n),
            position_ids=torch.arange(past,past+n).view(1,1,n).expand(3,1,n),
            past_key_values=cache,use_cache=True)
    def prefill(self,**kw):
        a=self.call([1,2,3]); b=self.call([4,5],a.past_key_values)
        return SimpleNamespace(key_values=b.past_key_values,valid_tokens=None,last_position=None)
    def denoise(self,state,valid,cache,pos,x,t):
        out=self.call([6,7,8],cache)
        return self.head(out.last_hidden_state[:,-2:])+x*.1

class CaptureTests(unittest.TestCase):
    def test_capture_finite_and_restores_model(self):
        torch.manual_seed(12)
        policy=TinyPolicy(); original=policy.lm.forward
        batch={f'observation.{key}':torch.zeros(1) for key in
            ['pixel_values','image_grid_thw','input_ids','attention_mask']}
        weights,meta=capture_final_step(SimpleNamespace(policy=policy),batch,0,1,1)
        self.assertEqual(len(weights),3)
        self.assertEqual(weights[0].shape,(3,3,3))
        self.assertEqual(weights[1].shape,(3,2,5))
        self.assertEqual(weights[2].shape,(3,3,8))
        self.assertTrue(all(torch.isfinite(torch.tensor(x)).all() for x in weights))
        self.assertTrue(any((x>0).any() for x in weights))
        self.assertEqual(policy.lm.forward,original)
        self.assertTrue(all(p.requires_grad for p in policy.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.parameters()))
        self.assertEqual(meta['gradient_scope'],'conditional_final_euler_step')

    def test_adapter_projection_seed_and_raw_artifacts(self):
        import numpy as np
        from attnvis.adapters.rollout import explain
        from attnvis.core.types import Frame
        policy=TinyPolicy()
        policy.model.config.enable_bottleneck_tokens=False
        batch={f'observation.{key}':torch.zeros(1) for key in
            ['pixel_values','attention_mask']}
        batch['observation.input_ids']=torch.tensor([[9,9,9]])
        batch['observation.image_grid_thw']=torch.tensor([[[1,2,2]]*3])
        adapter=SimpleNamespace(policy=policy,config=policy.model.config,capture='auto',
            _device='cpu',_build_batch=lambda frame:batch,_img_token_id=9,_merge=2,
            _image_resolution=(4,4),ckpt='tiny-test')
        frame=Frame(rgb={c:np.zeros((3,5,3),dtype=np.uint8) for c in ['a','b','c']},
            state=np.zeros(2),instruction='test',cams=['a','b','c'])
        before=torch.random.get_rng_state().clone()
        for method in ['rollout','gradient_rollout']:
            a=explain(adapter,frame,method,seed=15)
            b=explain(adapter,frame,method,seed=15)
            self.assertEqual(a.heat['a'].shape,(3,5))
            self.assertEqual(a.raw['action_to_image'].shape,(2,3))
            np.testing.assert_array_equal(a.heat['a'],b.heat['a'])
            self.assertTrue(np.isfinite(a.heat['a']).all())
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_exception_restores_hooks_and_flags(self):
        policy=TinyPolicy(); original=policy.lm.forward
        original_denoise=policy.model.denoise_step
        def fail(*args,**kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError('test injected failure')
            return original_denoise(*args,**kwargs)
        policy.model.denoise_step=fail
        batch={f'observation.{key}':torch.zeros(1) for key in
            ['pixel_values','image_grid_thw','input_ids','attention_mask']}
        with self.assertRaisesRegex(RuntimeError,'injected'):
            capture_final_step(SimpleNamespace(policy=policy),batch,0,1,1)
        self.assertEqual(policy.lm.forward,original)
        self.assertTrue(all(p.requires_grad for p in policy.parameters()))

    def test_cache_clone_preserves_gradient_and_isolation(self):
        policy=TinyPolicy(); cache=policy.prefill().key_values
        copied=clone_cache(cache)
        self.assertIsNot(copied.layers[0],cache.layers[0])
        self.assertNotEqual(copied.layers[0].keys.data_ptr(),cache.layers[0].keys.data_ptr())
        grad=torch.autograd.grad(copied.layers[0].keys.sum(),policy.lm.embed_tokens.weight)[0]
        self.assertGreater(grad.abs().sum(),0)

if __name__=='__main__': unittest.main()
