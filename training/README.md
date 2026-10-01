# Nivora AI Improvement Pipeline

This folder prepares explicitly approved feedback examples for future model fine-tuning.

Flow:
1. User gives positive feedback.
2. The example is stored with the feedback record.
3. `build_dataset.py` keeps only explicitly approved examples.
4. The resulting JSONL can be reviewed before any training job.
5. A separate training job can fine-tune a chosen base model.

Important: this is an improvement pipeline, not uncontrolled self-training. Nivora AI must not automatically change its model weights from arbitrary user conversations.

A real independently trained foundation model requires a large, licensed dataset, tokenizer, architecture, substantial compute, evaluation, and repeated training runs. This repository now has the data/evaluation side needed to move toward that later, but it does not claim that a foundation model has already been trained.
