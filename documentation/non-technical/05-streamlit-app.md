# The Streamlit Test App
## What it does

The app is a simple website you run on your own machine. You give it a product in one of three
ways:

- **Upload a photo** of the product, or
- **Type a description** of it, or
- **Both** - a photo plus a few notes.

Then you click one button - **"Recommend packaging"** - and after a few seconds it shows you:

1. **A written description** of the product (what the system understood it to be).
2. **A packaging bill of materials** - the full recipe, broken into Primary / Secondary /
   Tertiary tiers, listing each part, its shape and size, its materials, and their weights.
3. **Sustainability metrics** - the total mass, carbon footprint (kg CO₂e), water use (litres),
   recyclability, and recycled content.

The sustainability numbers are **looked up from a trusted catalog**, not guessed by the model
- so they're grounded in real data (see [preprocessing](02-preprocessing.md)).

## How it works behind the scenes

When you click the button, the app runs the same two-step process described in the
[fine-tuning doc](03-sft.md):

- If you gave a **photo**, the model first *looks at it* and writes a description.
- If you gave **text**, it tidies up your note into a clean description.
- Either way, that description is then fed into the trained model, which produces the packaging
  recipe.
- Finally the app joins each predicted material to the environmental catalog to compute the
  footprint, and lays everything out in neat tables.

## The most important feature: the on/off switch

The model runs on a rented cloud GPU server, which **costs money for every hour it's on**. The
single biggest budget risk in the whole project is leaving that server running by accident -
a large one can cost over **$120 a day**.

So the app's sidebar is built around protecting the budget:

- **Check status** - see whether the server is currently on.
- **Deploy** - turn the server on (it warns you this "spins up GPU $$" and takes about
  8–12 minutes).
- **🗑️ DELETE endpoint (stop billing)** - a big button to turn the server off the moment
  you're done.
- A **red warning banner** stays on screen the whole time the server is running, as a constant
  reminder.

The rule of thumb, repeated throughout the project: **always delete the endpoint when you
finish.**

## Who is this for?

This app is a *test and demo* tool - for the team to sanity-check the model on real products,
show stakeholders what it does, and explore its answers. It's not the final production product;
it's the place where you can see the whole pipeline working end to end in one click.

---

*Previous: [Evaluation](04-evaluation.md) · Back to the [overview](../README.md) · For the
code-level version, see [technical/05-streamlit-app.md](../technical/05-streamlit-app.md).*
