# Fine-Tuning the Model (SFT)

"SFT" stands for **Supervised Fine-Tuning**: we show the model thousands of example
question-and-answer pairs and nudge it to give answers like the ones in our data.

## The key idea: two steps, one model

The clever part of this project is how it uses images without ever training on images.

The dataset has **no photos** attached to products - only text. But in the real product, a
user will upload a *photo*. So at run time the system does two passes through the **same**
Gemma model:

1. **Step 1 - look at the photo (not trained).** Gemma already knows how to look at an image
   and describe it. We use that built-in ability, untouched, to produce a short written
   description of the product photo.

2. **Step 2 - write the packaging recipe (this is what we train).** We feed that description
   (plus known facts like brand and size) into Gemma and ask for the packaging recipe.

**Only Step 2 is trained.** Step 1's image ability is deliberately left exactly as Google
built it.

## How we protect the image ability

To make sure training Step 2 doesn't accidentally damage Step 1, we **freeze** the parts of
the model that handle images - they are locked and receive no changes during training. Only
the language ("writing") part of the model is adjusted. This gives us a guarantee: the
photo-understanding step stays exactly as good as the original Gemma.

## The trickiest detail: matching training to real use

Here's a subtle trap we had to avoid. At run time, Step 2 receives a *photo caption* - plain,
generic, visual language ("a slim cylindrical tube with a twist base"). But our training data
only has polished *marketing descriptions* ("a luxurious sustainable matte lipstick…").

If we trained on marketing copy but fed it captions in production, the model would be
confused - it would be seeing a different *style* of input than it learned from. So during
data preparation we **rewrite each marketing description into caption-style text**: we
automatically strip out brand names, marketing buzzwords, and explicit material names,
leaving only the plain visual cues a camera could actually see. That way training input and
real-world input look the same.

We also show the model the *same* product described a few different ways, so it learns the
packaging recipe doesn't depend on the exact wording.

## What the model is asked to produce

The training answer is always the same shape: a structured packaging recipe listing, for each
part, its name, size, whether it's rigid or soft, whether it's reusable, and its materials
with weights and recycled-content percentage. Notably, the model is **not** asked to produce
carbon or water figures - those are looked up separately from a trusted catalog (see
[preprocessing](02-preprocessing.md)).

## Training cheaply and safely

Training runs in the cloud (AWS SageMaker) under a strict **$300 total budget**. A few choices
keep it cheap:

- A small, lightweight training technique (called QLoRA) that adjusts only a tiny slice of the
  model instead of all of it - fast and inexpensive.
- A **cheap "dev" run first** using the smaller 4-billion-parameter model (a dollar or two) to
  prove the whole pipeline works, before the real run on the larger 12-billion model.
- Every launch **prints a cost estimate and asks for confirmation** before spending anything.

The result of training is a single, ready-to-use model file saved to cloud storage, which the
test app can then serve.

---

*Previous: [Preprocessing](02-preprocessing.md) · Next: [Evaluation →](04-evaluation.md) ·
For the code-level version, see [technical/03-sft.md](../technical/03-sft.md).*
