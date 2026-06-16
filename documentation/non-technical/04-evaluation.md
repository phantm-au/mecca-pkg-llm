# Evaluation

## Why evaluate?

A model can look impressive and still be wrong. Before trusting it (or shipping it), we need
hard numbers: How often does it produce a valid answer? Does it pick the right materials? Are
the weights roughly right? Evaluation answers these questions on a **held-out test set** -
products that were deliberately kept out of training, so the model can't have memorised them.

## The two-setting test (and the "distribution tax")

We run the same test products through the model in **two settings**:

1. **Realistic captions** - the model is fed plain, photo-style descriptions, exactly like
   what it will receive in production from a real photo. **This is the number that decides
   whether we ship.**

2. **Rich descriptions** - the model is fed the polished marketing descriptions instead. This
   is a best-case, "if the input were perfect" number.

The **gap between the two** is what we call the **distribution tax**: it shows how much
accuracy we lose when the model gets a plain caption instead of a rich description. A big gap
means the model leans too heavily on flowery wording; a small gap means it's robust.

## What we measure

For each prediction we check things like:

- **Did it produce a valid answer?** (Is the output well-formed and complete?)
- **Did it pick the right materials?** (Compared against the known-correct recipe.)
- **Is the total weight about right?**
- **Did it get the number of packaging parts right?**

These are averaged across all test products into a small scorecard.

## Two other checks

- **Smoke test** - takes a handful of *real product photos* and runs the **entire** pipeline
  end to end (photo → description → packaging recipe), confirming the whole chain works on
  actual images, not just text.

- **Offline sanity test** - a free, no-server check that catches silent mistakes before we
  ever spend money on the cloud. Most importantly, it verifies that the exact wording the
  model was *trained* on matches the wording it will be *asked* at run time. If those ever
  drift apart, the model quietly gets worse - so this test guards against it.

## An honest note on current numbers

Right now the scores are **low on purpose**. The only model deployed so far is the cheap,
small **4-billion-parameter "dev" model**, trained as a quick shakedown on an older data
format - and its answers sometimes get cut off before the full recipe finishes. So a current
scorecard shows roughly **1 in 5 answers fully valid** in the realistic setting. That's the
*expected* result before the real, larger training run. The point of having evaluation in
place now is that the moment the real model is trained, we can measure it immediately and see
the improvement.

---

*Previous: [Fine-tuning](03-sft.md) · Next: [The Streamlit test app →](05-streamlit-app.md) ·
For the code-level version, see [technical/04-evaluation.md](../technical/04-evaluation.md).*
