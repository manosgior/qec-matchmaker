# qec-matchmaker

**Pick a decoder before you decode.** Matchmaker reads a noisy QEC circuit and predicts, in seconds
and without decoding a single shot, whether the configuration is above threshold, roughly what its
logical error rate will be, how long each decoder will take, and which decoder to use.

> **Status: early design.** There is no code yet. This README describes what the tool will do and the
> evidence that it can; the interface below is a plan, not a working API.

## Why

Choosing a decoder is usually a guess, and a costly one. In the [ECCentric](https://github.com/TUM-DSE/eccentric)
benchmark (58 codes compiled onto 25 devices under 12 noise models, decoded by 7 decoders on the same
shots) we found that:

- **No decoder wins everywhere.** Tesseract is the most accurate where it finishes, Relay-BP gives the
  best accuracy for its cost, MWPM and Chromobius are about 1,000× faster where they apply, and
  BeliefFind and BP-LSD never beat BP-OSD.
- **Cost spans six orders of magnitude** across decoders and circuits, and grows 40–4,750× above
  threshold, where every decoder is equally wrong.
- **Half of the decoding compute was wasted:** 48% of the CPU time went to configurations with a
  logical error rate above 50%.

All of this is predictable from the circuit alone.

## What it will do

Input: a noisy [Stim](https://github.com/quantumlib/Stim) circuit (or its detector error model) and an
objective: the most accurate decoder, the cheapest decoder that is not significantly worse, or the best
decoder within a time budget per shot or per syndrome round.

Output:

- the probability that the configuration is above threshold,
- the predicted logical error rate, with an interval,
- the predicted time per shot (and memory) of each decoder,
- the recommended decoder for the objective,
- the expected cost of collecting N logical errors.

Planned interface:

```bash
qec-matchmaker circuit.stim --objective cheapest
qec-matchmaker circuit.stim --budget-ms 10
```

```python
import stim
from qec_matchmaker import advise

advice = advise(stim.Circuit.from_file("circuit.stim"), objective="accurate")
advice.above_threshold     # probability
advice.ler                 # point estimate and interval
advice.time_per_shot       # per decoder
advice.decoder             # e.g. "tesseract"
```

## How it works

1. **Features from the detector error model**, computed in seconds per circuit and independent of
   ECCentric: the probability that each detector fires (exact for independent error mechanisms), the
   total error probability, the number of error mechanisms and how many detectors each flips, the
   circuit's size and rounds. Routing, remote couplers and device noise all show up in these
   quantities, so the model applies to codes and devices it has not seen.
2. **Models trained on ECCentric's measurements**: logical error rates and decode times of thousands
   of configurations, with timeouts treated as censored observations rather than missing data.
3. **Validation on held-out codes and devices**, so the numbers below describe codes the model never
   saw.

Preliminary results (simple models, leave-one-code-out, from ECCentric's analysis of 1,519 circuits):

| Prediction | Result |
|---|---|
| Above or below threshold | 94% correct |
| Logical error rate | within a factor of 1.9 (median), rank correlation 0.90 |
| Time per shot | within a factor of 1.7 (BP-OSD, MWPM) to 2.9 (Tesseract), rank correlation 0.85–0.97 |
| Decoder choice | a decoder not significantly worse than the best for 81–91% of circuits |

A single quantity already goes a long way: configurations in which more than about 23% of detectors
fire per round are almost always above threshold, and none below 5% were.

## Decoders

Trained on: PyMatching (MWPM), BP-OSD, BP-LSD and BeliefFind (ldpc), Relay-BP, Tesseract and
Chromobius. Planned: MWPF.

## Limitations

- Timings are single-core software decoders, including the decoder's construction; predictions for
  other machines need a calibration run, and hardware (FPGA/ASIC) decoders are not modelled yet.
- Training data are memory experiments; logical operations such as lattice surgery are future work.
- Predictions of the logical error rate are for planning, not for publication: a factor of 1.9 is far
  wider than the interval of a measured rate.

## Roadmap

- [ ] Versioned training dataset exported from ECCentric (features, measured rates and times)
- [ ] Feature extraction from a Stim circuit or detector error model
- [ ] Models: gradient-boosted trees with quantile regression, censored regression for timeouts
- [ ] Recommendation for each objective, and a time-budget mode for real-time decoding
- [ ] CLI and Python API, packaged on PyPI
- [ ] Evaluation on [decoder-bench](https://github.com/satvikmaurya/decoder-bench) traces
- [ ] Calibration of timings to a new machine; prediction of memory use

## Related work

- [ECCentric](https://github.com/TUM-DSE/eccentric): the end-to-end QEC benchmark whose measurements
  train Matchmaker.
- [decoder-bench](https://github.com/satvikmaurya/decoder-bench) (Maurya et al., IISWC 2025):
  benchmarks decoders on fixed decoding problems; Matchmaker predicts how they will do before running
  them.

## License and citation

To be decided.
