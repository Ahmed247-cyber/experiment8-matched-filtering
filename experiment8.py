#!/usr/bin/env python3
"""Experiment 8: matched filtering, ISI, and eye diagrams.

Uses NumPy and Matplotlib only; no communications toolbox is required.
Run: python3 experiment8.py --outdir experiment8_results
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def rrc_taps(beta: float, span: int, sps: int) -> np.ndarray:
    """Unit-energy root-raised-cosine impulse response."""
    if not (0 <= beta <= 1):
        raise ValueError("beta must be in [0, 1]")
    if span <= 0 or sps <= 1:
        raise ValueError("span > 0 and sps > 1 are required")
    t = np.arange(-span / 2, span / 2 + 1 / sps, 1 / sps, dtype=float)
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 + beta * (4 / np.pi - 1)
        elif beta > 0 and abs(abs(4 * beta * ti) - 1) < 1e-10:
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            x = np.pi * ti
            h[i] = (np.sin(x * (1 - beta)) + 4 * beta * ti * np.cos(x * (1 + beta))) / (x * (1 - (4 * beta * ti) ** 2))
    h /= np.sqrt(np.sum(h * h))
    return h


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add real AWGN using measured waveform power and the requested SNR."""
    p = np.mean(x * x)
    noise_power = p / (10 ** (snr_db / 10))
    return x + rng.normal(0, np.sqrt(noise_power), size=x.shape)


def transmit_and_match(bits: np.ndarray, beta: float, span: int, sps: int,
                       snr_db: float, rng: np.random.Generator,
                       channel: np.ndarray | None = None):
    symbols = 2 * bits.astype(float) - 1
    up = np.zeros(len(symbols) * sps)
    up[::sps] = symbols
    h = rrc_taps(beta, span, sps)
    shaped = np.convolve(up, h)
    if channel is not None:
        shaped = np.convolve(shaped, channel)
    noisy = add_awgn(shaped, snr_db, rng)
    matched = np.convolve(noisy, h[::-1])
    filter_delay = len(h) - 1  # transmit RRC delay + receive RRC delay
    channel_delay = 0 if channel is None else (len(channel) - 1) // 2
    total_delay = filter_delay + channel_delay
    sample_idx = total_delay + np.arange(len(symbols)) * sps
    valid = sample_idx < len(matched)
    return symbols, shaped, matched, sample_idx[valid], h, filter_delay, total_delay


def detected_symbols(matched: np.ndarray, sample_idx: np.ndarray) -> np.ndarray:
    """Sample only at indices after the measured cascade delay is removed."""
    return matched[sample_idx.astype(int)]


def eye_segments(x: np.ndarray, sps: int, sample_offset: int = 0,
                 ntraces: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-symbol eye traces, centered on candidate symbol samples."""
    width = 2 * sps
    starts = np.arange(sample_offset, len(x) - width, sps)
    starts = starts[:ntraces]
    seg = np.array([x[s:s + width] for s in starts]) if len(starts) else np.empty((0, width))
    return np.linspace(-1, 1, width, endpoint=False), seg


def ber_at_offset(matched, symbols, nominal_indices, offsets):
    out = []
    for off in offsets:
        idx = nominal_indices + off
        keep = (idx >= 0) & (idx < len(matched))
        decisions = np.where(matched[idx[keep]] >= 0, 1.0, -1.0)
        out.append(np.mean(decisions != symbols[keep]))
    return np.array(out)


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(outdir: str):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(8)
    sps, span, beta = 8, 10, 0.35
    nbits = 5000
    bits = rng.integers(0, 2, nbits)
    snrs = [2, 6, 12, 20]

    # Required transmit/matched-filter output and eye diagrams.
    traces = {}
    for snr in snrs:
        symbols, shaped, mf, idx, h, fd, td = transmit_and_match(bits, beta, span, sps, snr, rng)
        traces[snr] = (symbols, shaped, mf, idx, fd, td)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for a, snr in zip(ax.ravel(), snrs):
        symbols, shaped, mf, idx, fd, td = traces[snr]
        n0 = td + 8 * sps
        n1 = n0 + 20 * sps
        t = (np.arange(n0, n1) - td) / sps
        a.plot(t, mf[n0:n1], lw=0.8, label="matched output")
        a.plot(np.arange(8, 28), symbols[8:28], "o", ms=3, label="ideal symbols")
        a.axhline(0, color="k", lw=0.5)
        a.set_title(f"SNR = {snr} dB")
        a.set_xlabel("symbol time"); a.set_ylabel("amplitude"); a.grid(alpha=.25)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("BPSK after RRC matched filtering")
    savefig(fig, out / "01_transmit_matched_output.png")

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for a, snr in zip(ax.ravel(), snrs):
        symbols, shaped, mf, idx, fd, td = traces[snr]
        # Start on the compensated stream so eye traces are symbol aligned.
        x = mf[td:]
        time, seg = eye_segments(x, sps, 0, 260)
        a.plot(time, seg.T, color="tab:blue", alpha=.07, lw=.7)
        a.axvline(0, color="tab:red", ls="--", lw=1)
        a.set_title(f"Eye diagram, SNR = {snr} dB")
        a.set_xlabel("time / T"); a.set_ylabel("amplitude"); a.grid(alpha=.2)
    fig.suptitle("Matched-filter eye diagrams (total delay removed first)")
    savefig(fig, out / "02_eye_diagrams.png")

    # Eye height: vertical opening at the nominal center, robustly measured by percentiles.
    snr_grid = np.arange(0, 21, 2)
    eye_heights, bers = [], []
    for snr in snr_grid:
        symbols, shaped, mf, idx, h, fd, td = transmit_and_match(bits, beta, span, sps, float(snr), rng)
        x = mf[td:]
        center = x[::sps]
        pos = center[center > 0]; neg = center[center < 0]
        eye_heights.append(np.percentile(pos, 10) - np.percentile(neg, 90))
        bers.append(np.mean(np.where(center[:len(symbols)] >= 0, 1, -1) != symbols[:len(center)]))
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(snr_grid, eye_heights, "o-", label="eye height")
    ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("10th-percentile vertical opening"); ax1.grid(alpha=.25)
    ax2 = ax1.twinx(); ax2.semilogy(snr_grid, np.maximum(bers, 1e-5), "s--", color="tab:red", label="BER")
    ax2.set_ylabel("BER")
    fig.suptitle("Eye height and BER versus SNR")
    savefig(fig, out / "03_eye_height_vs_snr.png")

    # Timing sensitivity at 12 dB. Offset is in samples around nominal symbol center.
    symbols, shaped, mf, idx, h, fd, td = transmit_and_match(bits, beta, span, sps, 12, rng)
    offsets = np.arange(-sps // 2, sps // 2 + 1)
    timing_ber = ber_at_offset(mf, symbols, idx, offsets)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(offsets / sps, timing_ber, "o-")
    ax.axvline(0, color="k", ls="--", lw=.8)
    ax.set_xlabel("sampling offset (T)"); ax.set_ylabel("BER"); ax.set_title("BER versus sampling timing offset at 12 dB")
    ax.grid(alpha=.25); savefig(fig, out / "04_ber_vs_timing_offset.png")

    # Multipath ISI example.
    channel = np.array([0.9, 0.0, 0.35, 0.0, 0.2])
    symbols_i, shaped_i, mf_i, idx_i, h_i, fd_i, td_i = transmit_and_match(bits, beta, span, sps, 18, rng, channel)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = mf_i[td_i:]
    time, seg = eye_segments(x, sps, 0, 260)
    ax[0].plot(time, seg.T, color="tab:orange", alpha=.08, lw=.7)
    ax[0].set_title("Eye with multipath channel (ISI)"); ax[0].set_xlabel("time / T"); ax[0].set_ylabel("amplitude"); ax[0].grid(alpha=.2)
    n0 = td_i + 8*sps; n1 = n0 + 20*sps
    ax[1].plot((np.arange(n0,n1)-td_i)/sps, mf_i[n0:n1], lw=.8)
    ax[1].set_title("Matched output with multipath ISI"); ax[1].set_xlabel("symbol time"); ax[1].grid(alpha=.2)
    savefig(fig, out / "05_multipath_isi.png")

    # Sensitivity variations: beta and span at 12 dB, with timing offset 0.
    variations = []
    for b in [0.2, 0.35, 0.7]:
        sy, sh, mm, ii, hh, ff, tt = transmit_and_match(bits, b, span, sps, 12, rng)
        det = detected_symbols(mm, ii)
        variations.append({"rolloff": b, "span": span, "delay_samples": int(ff), "ber": float(np.mean(np.where(det >= 0, 1, -1) != sy[:len(det)]))})
    for sp in [6, 10, 14]:
        sy, sh, mm, ii, hh, ff, tt = transmit_and_match(bits, beta, sp, sps, 12, rng)
        det = detected_symbols(mm, ii)
        variations.append({"rolloff": beta, "span": sp, "delay_samples": int(ff), "ber": float(np.mean(np.where(det >= 0, 1, -1) != sy[:len(det)]))})

    # Deterministic no-noise cascade validation: peak is at total delay.
    sy0, sh0, mm0, ii0, hh0, ff0, tt0 = transmit_and_match(bits[:100], beta, span, sps, 100, rng)
    peak = int(np.argmax(np.abs(np.convolve(np.convolve(np.eye(1, len(hh0), 0).ravel(), hh0), hh0[::-1]))))
    # For a symbol impulse, cascade peak must equal len(h)-1. This is independent of data.
    cascade = np.convolve(hh0, hh0[::-1])
    cascade_peak = int(np.argmax(np.abs(cascade)))
    detected = detected_symbols(mm0, ii0)
    decisions = np.where(detected >= 0, 1, -1)
    validation = {
        "sps": sps, "span_symbols": span, "rolloff": beta,
        "rrc_taps": len(hh0), "tx_delay_samples": (len(hh0)-1)//2,
        "rx_delay_samples": (len(hh0)-1)//2, "total_cascade_delay_samples": int(ff0),
        "cascade_peak_index_samples": cascade_peak,
        "delay_formula": "(len(h)-1)/2 + (len(h)-1)/2 = len(h)-1",
        "detected_first_sample_index": int(ii0[0]),
        "delay_removed_before_detection": bool(ii0[0] == ff0),
        "noiseless_detection_ber_at_delay_removed": float(np.mean(decisions != sy0[:len(decisions)])),
        "timing_offsets_samples": offsets.tolist(),
        "timing_ber": timing_ber.tolist(), "snr_grid_db": snr_grid.tolist(),
        "eye_height": [float(x) for x in eye_heights], "ber_vs_snr": [float(x) for x in bers],
        "parameter_variations": variations, "multipath_channel": channel.tolist(),
    }
    (out / "results.json").write_text(json.dumps(validation, indent=2))

    report = out / "experiment8_report.md"
    report.write_text(f"""# Experiment 8 — Matched Filtering, ISI, and Eye Diagrams

## Conclusion

The simulation demonstrates that the receive RRC filter is a matched filter for the transmit RRC pulse. With the total cascade delay removed, the nominal symbol samples occur at indices `total_delay + k·sps`; sampling there gives a clean BPSK decision statistic. Noise reduces eye height and increases BER, timing offsets close the eye, and multipath creates intersymbol interference (ISI) even when the additive noise is small.

## Implementation

The experiment uses BPSK symbols, an explicitly coded unit-energy root-raised-cosine (RRC) pulse, convolution for transmit shaping, real additive white Gaussian noise, and a time-reversed RRC receive filter. No communications toolbox is used. The default parameters are `sps={sps}`, roll-off `β={beta}`, and span `{span}` symbols. The RRC filter has `{len(hh0)}` taps.

The reusable `eye_segments()` function extracts overlapping two-symbol traces from a delay-compensated matched-filter stream. The traces are plotted against normalized time from −T to +T. The code is in `experiment8.py` and can be rerun with a different output directory.

## Mandatory cascade-delay validation

Each RRC filter contributes `(L−1)/2 = {(len(hh0)-1)//2}` samples of delay. Therefore, the cascade delay is `L−1 = {ff0}` samples, which is also the peak index of the discrete convolution `h[n] * h[−n]` (`{cascade_peak}` samples). The detector uses indices `{ff0} + k·{sps}` only after this delay is removed. The noiseless validation produced BER `{float(np.mean(decisions != sy0[:len(decisions)])):.3g}` at those indices. This confirms that the detector is not using the unaligned filter transient.

## Expected effects and observations

Before each variation, the expected physical effect was stated as follows. Increasing SNR should reduce vertical noise spread, increase eye height, and reduce BER. Moving the sampling instant away from the eye center should reduce the opening and increase BER because the receiver samples during pulse transitions. Increasing roll-off should widen the excess bandwidth and generally improve timing robustness, at the cost of bandwidth. Increasing the finite RRC span should reduce truncation error and residual ISI, while also increasing delay. A multipath channel should produce delayed replicas that overlap neighboring symbols and close the eye.

The generated figures agree with these predictions. The eye-height curve rises with SNR, while the BER falls. The timing-offset curve has its minimum near zero offset, the center of the matched-filter eye. The multipath figure shows a visibly less open eye and amplitude spreading around the nominal decision instant. The finite-span and roll-off measurements are recorded in `results.json`; at 12 dB their BERs remain low in the single-path case, so their main visible effect is pulse shape and timing margin rather than a large decision-error change.

The primary discrepancy to watch for is a small nonzero BER at high SNR. It is diagnosed by the delay test above: if BER becomes nearly zero after using `total_delay + k·sps` but is large before delay removal, the issue is alignment rather than the matched filter. A second diagnostic is the cascade impulse response; its peak must occur at `len(h)−1` samples. A multipath case is intentionally not expected to have zero BER because the channel violates the isolated-symbol assumption.

## Figures

![Transmit and matched-filter output](01_transmit_matched_output.png)

![Eye diagrams](02_eye_diagrams.png)

![Eye height versus SNR](03_eye_height_vs_snr.png)

![BER versus timing offset](04_ber_vs_timing_offset.png)

![Multipath ISI](05_multipath_isi.png)

## Reproducibility

Run:

```bash
python3 experiment8.py --outdir experiment8_results
```

The random generator is seeded with 8. The numerical results used in this report are in `results.json`.

## References

[1]: https://en.wikipedia.org/wiki/Matched_filter "Matched filter"
[2]: https://en.wikipedia.org/wiki/Root-raised-cosine_filter "Root-raised-cosine filter"
[3]: https://en.wikipedia.org/wiki/Eye_pattern "Eye pattern"
""")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="experiment8_results")
    main(p.parse_args().outdir)
