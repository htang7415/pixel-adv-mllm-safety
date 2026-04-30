# Key Findings

## Main Result

The experiment does not support the original hypothesis that bounded pixel-space
PGD increases harmful compliance over clean images. Clean prompts already had a
high attack success rate (ASR), creating a ceiling effect. Under independent
human scoring, PGD more often reduced direct harmful compliance than increased
it.

## Quantitative Summary

| Condition | ASR | Mean score | Mean LPIPS |
|---|---:|---:|---:|
| Clean | 0.9333 | 3.8000 | N/A |
| Random 4/255 | 0.9111 | 3.7333 | 0.0045 |
| Random 8/255 | 0.9333 | 3.8000 | 0.0236 |
| FGSM 4/255 | 0.9333 | 3.8000 | 0.0392 |
| FGSM 8/255 | 0.9333 | 3.8000 | 0.1255 |
| PGD 4/255 | 0.8222 | 3.8000 | 0.0134 |
| PGD 8/255 | 0.6889 | 3.5778 | 0.0497 |

Paired against the clean condition, PGD at 8/255 improved ASR on 2 pairs,
degraded ASR on 13 pairs, and left 30 unchanged. A two-sided exact sign test on
the discordant paired ASR changes gives p = 0.0074.

## Interpretation

The targeted PGD attack appears to move the model distribution, but not in the
intended direction. Many PGD responses became less directly operational:
off-target room- or object-conditioned instructions, legalistic rewrites,
repetitive procedural loops, or partial/hedged responses. This suggests a
surrogate-objective mismatch: lowering the loss on the affirmative prefix
("Sure, here is") does not guarantee higher harmful-compliance ASR in final
generation.

## Recommended Storyline

Use the result as a negative but informative finding:

> In a highly vulnerable multimodal LLM, direct harmful prompts already produce
> high unsafe-compliance rates. Small pixel-space PGD perturbations can alter
> behavior under low perceptual distortion, but in this setup they tend to
> disrupt direct harmful compliance rather than amplify it.

## Limitations

- The clean condition has a very high ASR, so the design has limited headroom to
  show attack-driven increases.
- Only 15 prompts and 3 seed images are used; image-level conclusions are
  descriptive only.
- The study uses one model and one surrogate target prefix.
- Human scoring is independent, but a second annotator would improve reliability
  if time permits.

## Best Follow-Up

For a stronger positive attack-success study, run a second filtered experiment
using only prompt-image pairs where the clean response refuses or partially
refuses. That would directly test whether pixel perturbations can flip safe
responses into unsafe compliance.
