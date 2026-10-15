import unittest
import numpy as np
from attnvis.core.attribution import cached_rollout, chefer_weights

class RolloutTests(unittest.TestCase):
    def test_cached_matches_full_graph(self):
        rng=np.random.default_rng(7)
        for slots in (0,2):
            P,C,S,L=3,4,2,3
            p=rng.random((L,P,P)); c=rng.random((L,C,P+C))
            s=rng.random((L,S,(slots or P+C)+S))
            for x in (p,c,s): x /= x.sum(-1,keepdims=True)
            full=np.eye(P+C+S)
            for i in range(L):
                a=np.zeros((P+C+S,P+C+S))
                a[:P,:P]=p[i]; a[P:P+C,:P+C]=c[i]
                keys=list(range(P+C-slots,P+C)) if slots else list(range(P+C))
                keys += list(range(P+C,P+C+S))
                a[P+C:,keys]=s[i]
                a += np.eye(P+C+S)
                a /= a.sum(-1,keepdims=True)
                full=a@full
            actual=cached_rollout(p,c,s,[0,2],slots=slots)
            np.testing.assert_allclose(actual,full[-S:][:,[0,2]],rtol=1e-6)

    def test_cannot_cross_two_hops_in_one_layer(self):
        p=np.ones((1,1,1)); c=np.array([[[1.,0.]]]); s=np.array([[[1.,0.]]])
        np.testing.assert_array_equal(cached_rollout(p,c,s,[0],slots=1),[[0.]])
        result=cached_rollout(np.repeat(p,2,0),np.repeat(c,2,0),np.repeat(s,2,0),[0],slots=1)
        np.testing.assert_allclose(result,[[.25]])

    def test_chefer_clips_before_head_average(self):
        r=np.array([[[[2.,3.]],[[4.,5.]]]])
        g=np.array([[[[-1.,2.]],[[3.,-4.]]]])
        np.testing.assert_array_equal(chefer_weights(r,g),[[[6.,3.]]])

    def test_reject_invalid(self):
        with self.assertRaises(ValueError):
            cached_rollout(np.zeros((1,2,3)),np.zeros((1,1,3)),np.zeros((1,1,4)),[0])
        with self.assertRaises(ValueError):
            chefer_weights(np.full((1,1,1,1),np.nan),np.ones((1,1,1,1)))

if __name__ == '__main__': unittest.main()
