"""Google Speech Commands v0.02 as a 12-class keyword-spotting federated task (audio domain).

Classes: the ten keywords (yes, no, up, down, left, right, on, off, stop, go), 'unknown' (the other 25 words) and
'silence' (1 s crops of the background-noise files). Features: 40-bin log-mel spectrogram, 48 frames (32 ms window,
20 ms hop), so a clip is a (40, 48) array that the 1D multi-exit CNN treats as 40 channels over 48 time steps.

Partition: the dataset's own validation/testing lists are speaker-disjoint from training. Each federated client is ONE
training speaker (the 24 speakers with most clips), so label skew arises naturally from which words a speaker
happened to record. Noise files are split by role (train / validation / test) so no noise recording is shared.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

KEYWORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
UNKNOWN, SILENCE = 10, 11
SR, N_FFT, HOP, N_MEL, N_FRAMES = 16000, 512, 320, 40, 48
NOISE_SPLIT = {"train": ("doing_the_dishes.wav", "exercise_bike.wav", "pink_noise.wav", "running_tap.wav"),
               "val": ("dude_miaowing.wav",), "test": ("white_noise.wav",)}


def _mel_filterbank():
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(20.0), hz2mel(SR / 2), N_MEL + 2))
    bins = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    fb = np.zeros((N_MEL, len(bins)), np.float32)
    for i in range(N_MEL):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)))
    return torch.from_numpy(fb)


_FB, _WIN = None, None


def logmel(wave: np.ndarray) -> np.ndarray:
    """wave: (n_clips, 16000) float32 in [-1, 1] -> (n_clips, 40, 48) float32."""
    global _FB, _WIN
    if _FB is None:
        _FB, _WIN = _mel_filterbank(), torch.hann_window(N_FFT)
    x = torch.from_numpy(wave)
    spec = torch.stft(x, N_FFT, HOP, window=_WIN, center=True, return_complex=True).abs() ** 2   # (n, 257, frames)
    mel = torch.einsum("mf,nft->nmt", _FB, spec)
    return torch.log(mel + 1e-6)[:, :, :N_FRAMES].numpy().astype(np.float32)


def _read(path: Path) -> np.ndarray:
    sr, w = wavfile.read(path)
    w = w.astype(np.float32) / 32768.0
    out = np.zeros(SR, np.float32)
    out[:min(SR, len(w))] = w[:SR]
    return out


def _label(word: str) -> int:
    return KEYWORDS.index(word) if word in KEYWORDS else UNKNOWN


def _noise_clips(root: Path, files, n: int, rng) -> np.ndarray:
    waves = []
    sig = [wavfile.read(root / "_background_noise_" / f)[1].astype(np.float32) / 32768.0 for f in files]
    for _ in range(n):
        s = sig[rng.integers(len(sig))]
        st = rng.integers(0, len(s) - SR)
        waves.append(s[st:st + SR] * rng.uniform(0.05, 1.0))
    return np.stack(waves).astype(np.float32)


def load_speech(root: str | Path = "data/speech_commands", n_clients: int = 24, seed: int = 0,
                unknown_frac: float = 0.25, silence_frac: float = 0.10, val_n: int = 1500, test_cap: int = 7000):
    """Return (clients, val, test): clients = {i: (X, y)} with float32 features/int64 labels."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    val_set = set((root / "validation_list.txt").read_text().split())
    test_set = set((root / "testing_list.txt").read_text().split())
    words = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))
    items = {"train": {}, "val": [], "test": []}
    for w in words:
        for f in sorted((root / w).glob("*.wav")):
            rel = f"{w}/{f.name}"
            spk = f.name.split("_nohash_")[0]
            if rel in test_set:
                items["test"].append((f, w))
            elif rel in val_set:
                items["val"].append((f, w))
            else:
                items["train"].setdefault(spk, []).append((f, w))

    def build(pairs, noise_role, cap=None):
        kw = [p for p in pairs if p[1] in KEYWORDS]
        unk = [p for p in pairs if p[1] not in KEYWORDS]
        n_unk = min(len(unk), int(round(unknown_frac / (1 - unknown_frac - silence_frac) * len(kw))))
        unk = [unk[i] for i in rng.permutation(len(unk))[:n_unk]]
        sel = kw + unk
        if cap and len(sel) > cap:
            sel = [sel[i] for i in rng.permutation(len(sel))[:cap]]
        n_sil = int(round(silence_frac / (1 - silence_frac) * len(sel)))
        waves = np.stack([_read(f) for f, _ in sel]) if sel else np.zeros((0, SR), np.float32)
        y = np.array([_label(w) for _, w in sel], np.int64)
        if n_sil:
            waves = np.concatenate([waves, _noise_clips(root, NOISE_SPLIT[noise_role], n_sil, rng)])
            y = np.concatenate([y, np.full(n_sil, SILENCE, np.int64)])
        return logmel(waves), y

    top = sorted(items["train"], key=lambda s: -len(items["train"][s]))[:n_clients]
    clients = {i: build(items["train"][s], "train") for i, s in enumerate(top)}
    val = build(items["val"], "val", cap=val_n)
    test = build(items["test"], "test", cap=test_cap)
    return clients, val, test


def load_speech_cached(root: str | Path = "data", **kw):
    cache = Path(root) / "speech_cache.npz"
    if cache.exists():
        z = np.load(cache)
        n = int(z["n_clients"])
        clients = {i: (z[f"cx{i}"], z[f"cy{i}"]) for i in range(n)}
        return clients, (z["vx"], z["vy"]), (z["tx"], z["ty"])
    clients, val, test = load_speech(Path(root) / "speech_commands", **kw)
    arrs = {"n_clients": len(clients), "vx": val[0], "vy": val[1], "tx": test[0], "ty": test[1]}
    for i, (x, y) in clients.items():
        arrs[f"cx{i}"], arrs[f"cy{i}"] = x, y
    np.savez_compressed(cache, **arrs)
    return clients, val, test
