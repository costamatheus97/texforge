"""Smoke tests for texforge.organic (run: python -m unittest discover -s tests -v)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from texforge import maps, organic  # noqa: E402
from texforge.common import blur, luminance, srgb_to_lin  # noqa: E402


def _rough_height(n=256, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(0.5 + blur(rng.standard_normal((n, n)).astype(np.float32), 4.0) * 0.9, 0, 1).astype(np.float32)


class Despeckle(unittest.TestCase):
    def test_injected_specks_removed_detail_kept(self):
        h = _rough_height()
        rng = np.random.default_rng(1)
        ys, xs = rng.integers(8, 248, 60), rng.integers(8, 248, 60)
        bad = h.copy()
        bad[ys[:30], xs[:30]] += 0.6          # single-pixel spikes
        bad[ys[30:], xs[30:]] -= 0.6          # single-pixel pits
        bad = np.clip(bad, 0, 1)
        out, st = organic.despeckle_height(bad)
        self.assertGreaterEqual(st["outlier_px_before"], 40)
        self.assertLess(st["outlier_px_after"], 0.2 * st["outlier_px_before"])
        self.assertLess(float(np.abs(out[ys, xs] - h[ys, xs]).mean()), 0.08)     # back near the clean surface
        keep = np.ones(h.shape, bool)
        keep[ys, xs] = False
        keep = ~__import__("scipy.ndimage", fromlist=["x"]).binary_dilation(~keep, iterations=2)
        self.assertLess(float(np.abs(out - bad)[keep].mean()), 1e-3)             # (almost) untouched elsewhere

    def test_real_features_survive(self):
        h = _rough_height()
        out, st = organic.despeckle_height(h)
        self.assertLess(st["pct_px_changed"], 1.0)


class Shadows(unittest.TestCase):
    def test_lift_reduces_height_correlation_and_keeps_mean(self):
        h = _rough_height()
        occ = maps.ao_map(h, 1.0)
        albedo_lin = np.stack([np.full(h.shape, v, np.float32) for v in (0.10, 0.16, 0.03)], -1)
        shaded = np.clip(albedo_lin * (occ ** 1.2)[..., None], 0, 1)
        from texforge.common import lin_to_srgb
        s = lin_to_srgb(shaded)
        out, info = organic.lift_shadows(s, h, strength=1.0, gain_max=4.0)
        self.assertLess(info["after"]["corr_lum_vs_ao"], info["before"]["corr_lum_vs_ao"] - 0.15)
        m0, m1 = luminance(srgb_to_lin(s)).mean(), luminance(srgb_to_lin(out)).mean()
        self.assertAlmostEqual(float(m0), float(m1), delta=0.002)
        self.assertGreater(info["fitted_exponent_b"], 0.5)


class Masks(unittest.TestCase):
    def test_masks_tile_and_sum_to_one(self):
        n = 192
        rng = np.random.default_rng(3)
        g = blur(rng.standard_normal((n, n)).astype(np.float32), 5.0, "xy")
        g = (g - g.min()) / (g.max() - g.min())
        base = np.stack([0.2 + 0.1 * g, 0.35 + 0.3 * g, 0.05 + 0 * g], -1).astype(np.float32)
        base[g < 0.25] = (0.25, 0.15, 0.08)                                     # a brown "soil" population
        mk = organic.material_masks(base, g, "moss")
        p = mk["prob"]
        self.assertTrue(np.allclose(p.sum(-1), 1.0, atol=1e-3))
        self.assertGreater(p[..., 2][g < 0.25].mean(), 0.5)                      # soil channel finds the brown dark low area
        shifted = np.roll(base, (37, 91), (0, 1))
        mk2 = organic.material_masks(shifted, np.roll(g, (37, 91), (0, 1)), "moss")
        self.assertLess(float(np.abs(np.roll(mk["prob"], (37, 91), (0, 1)) - mk2["prob"]).mean()), 0.02)   # translation-equivariant: tiles


class SoilGaps(unittest.TestCase):
    """Regression: soil gaps between moss clumps were flat plates (height clipped to one value, albedo ~constant, hard rim)."""

    def _synthetic(self, n=256, seed=5):
        rng = np.random.default_rng(seed)
        coarse = blur(rng.standard_normal((n, n)).astype(np.float32), 14.0, "xy")
        coarse = (coarse - coarse.min()) / (coarse.max() - coarse.min())
        fine = blur(rng.standard_normal((n, n)).astype(np.float32), 1.3, "xy")
        fine = fine / fine.std()
        soil = coarse < 0.38
        lum = np.where(soil, 0.045 + 0.012 * fine, 0.16 + 0.05 * fine)          # soil: dark but with real micro-detail
        rgb = np.stack([lum * np.where(soil, 1.0, 0.55), lum * np.where(soil, 0.8, 1.0), lum * np.where(soil, 0.5, 0.25)], -1)
        from texforge.common import lin_to_srgb
        base = lin_to_srgb(np.clip(rgb, 0, 1)).astype(np.float32)
        h = np.clip(0.2 + 0.6 * coarse + 0.05 * fine, 0, 1).astype(np.float32)
        return base, h, soil

    def test_soil_has_detail_and_no_plateau(self):
        base, h, soil = self._synthetic()
        d = organic.derive_organic(base, "organic", None, None, 1.0, h, None, "moss_test",
                                   dict(despeckle=True, shadow_lift=True, materials=True, profile="moss"))
        hh = d["height"]
        core = ndi_erode(soil, 6)
        hp = hh - blur(hh, 3.0, "xy")
        self.assertGreater(float(hp[core].std()), 0.004)                           # micro-relief inside the gaps
        self.assertLess(float((hh[core] <= hh.min() + 0.01).mean()), 0.01)         # no single-value plateau at the floor
        la = np.log(np.maximum(luminance(srgb_to_lin(d["albedo"])), 1e-3))
        self.assertGreater(float((la - blur(la, 3.0, "xy"))[core].std()), 0.05)    # albedo keeps high-frequency luminance
        nrm = d["normal"][..., :2] * 2 - 1
        self.assertGreater(float(nrm[core].std()), 0.03)                           # normal is not flat inside the gaps

    def test_soil_mask_edge_is_not_a_hard_step(self):
        n = 1024                                                                    # feather radius scales with the tile
        step = np.zeros((n, n), np.float32)
        step[:, n // 2:] = 1.0                                                      # worst case: a binary cut-out
        f = organic.soil_feather(step, n)
        grad = np.abs(np.diff(f[:, n // 4:3 * n // 4], axis=1)).max()      # (the wrap seam at col 0 is a step too)
        self.assertLess(float(grad), 0.15)                                          # ramps over >= ~8 px
        self.assertGreater(float((f[:, n // 2 - 100] < 0.05).mean()), 0.99)         # ...and does not bleed far


def ndi_erode(m, it):
    from scipy import ndimage as ndi
    return ndi.binary_erosion(m, iterations=it)


if __name__ == "__main__":
    unittest.main()
