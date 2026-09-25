---
title: SOCR Brain Image Generator backend
emoji: 🧠
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.10"
app_file: app.py
pinned: false
short_description: API backend for the SOCR Brain-Image-Generator website
---

# SOCR Brain Image Generator: backend

This is the backend for the [SOCR Brain-Image-Generator](https://github.com/SOCR/Brain-Image-Generator)
website. The five GAN models run on CPU. The atlas-guided conditional diffusion model
(braingen_CondDiffuser_BraTS_v1) runs on ZeroGPU.

Weights and the conditioning bank are downloaded at startup from a private model repo (`ASSETS_REPO`).
The source, and the full deployment notes, are in `backend/hf_space/` in the GitHub repo.
