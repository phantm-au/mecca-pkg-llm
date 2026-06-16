# Data Preprocessing

## Why clean the data at all?

Raw data, even synthetic data, has rough edges: some records are missing a name, some have a
broken packaging recipe, and some are near-identical copies of each other. If we trained on
that as-is, the model would learn from noise and waste effort memorising duplicates. So before
training we run the data through a short cleanup pipeline.

This pipeline follows the same *idea* as a standard data-prep toolkit (sometimes called
"DPK-style"): **ingest → validate → de-duplicate → profile → build catalog**. We implement it
as lightweight local steps because 10,000 products is small enough not to need heavy tooling.

## The five steps, in everyday terms

1. **Reshape (ingest).** Rearrange each raw record into a clean two-part shape: the *inputs*
   (the facts about the product) and the *target* (the packaging recipe to predict). This
   makes every example look identical and easy to work with.

2. **Validate (quality check).** Throw out records that can't be used - no product name, no
   primary packaging, a material with no name, a material from an unknown category, or a
   weight that isn't a real number. We keep a tally of *why* each rejected record was dropped.

3. **De-duplicate (remove copies).** Synthetic data can produce many products that are
   essentially the same. We detect near-duplicates and keep only one of each, so the model
   isn't over-trained on clones. (Two products count as "near-duplicates" when their
   name, brand, description, and packaging structure overlap heavily.)

4. **Profile (take a census).** Produce a summary of the cleaned set - how many products,
   the average number of parts per tier, how often each material type shows up. This is a
   quick health check on the data.

5. **Build the materials catalog (the dictionary).** Scan the cleaned data and compile the
   master list of every material that appears - **66 materials** in total, grouped into 7
   types (Plastic, Paper/Board, Metal, Glass, Wood, Textile, Other). This list matters a lot:
   later, the model is only allowed to choose materials *from this list*, which keeps its
   answers consistent and prevents it from inventing made-up materials. It is, in effect, the
   single biggest lever on the model's accuracy.

## A second dictionary: the environmental catalog

There's one more important file that lives alongside the materials catalog: the
**environmental catalog**. For each of the 66 materials, it records real-world environmental
numbers:

- **Carbon** - how much CO₂ is emitted to produce a kilogram of that material.
- **Water** - how many litres of water it takes.
- **Recyclability** - how recyclable it is.
- **Fossil-fuel use** - how much fossil energy goes into making it.

These numbers come from real materials data (not invented). They are stored as "per kilogram"
intensities - so later, once the model predicts *how much* of each material a product uses,
we multiply by these intensities to get the product's actual environmental footprint.

**Why this matters:** the model is never asked to guess carbon or water numbers - those would
be unreliable. Instead the model predicts only *what material and how much*, and the
environmental figures are looked up from this trusted catalog. This keeps the sustainability
metrics honest and grounded in real data.

## What comes out of this stage

- A **clean products file** - the tidy training examples.
- The **materials catalog** - the master list of 66 allowed materials.
- A **profile** - the census/summary of the cleaned data.
- (Already prepared) the **environmental catalog** - the trusted carbon/water/recyclability
  numbers used much later for sustainability metrics.

---

*Previous: [The dataset](01-dataset.md) · Next: [Fine-tuning the model →](03-sft.md) · For the
code-level version, see [technical/02-preprocessing.md](../technical/02-preprocessing.md).*
