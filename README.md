## Project structure and requirements

This project uses the following structure assuming that the data has been fetched from the 2015 directory (also shared in the course page). It also assumes the required local installation for Ollama and a local GPU to perform training.

Requirements are listed in the ```requirements.txt``` file
```
factkg/
├─ README.md  requirements.txt  .gitignore  env.sh
├─ data/ models/ *_outputs/        # Which is ignored
├─ src/
│  ├─ common/                config.py config_phase2.py kg_access.py triple_verbalizer.py
│  ├─ phase1_linking/        llm_extractor local_kg_index entity_validator ngram_fallback
│  │                         lookup_fallback pipeline evaluate error_analysis compare_runs run_phase1
│  ├─ phase2_retrieval/      candidate_retrieval rerankers evidence_eval run_phase2 ablate_candidates
│  │                         diagnose_phase2 build_rerank_data train_cross_encoder
│  ├─ phase3_verification/   llm_verifier run_phase3_llm
│  ├─ adversarial/           adversarial live_pipeline run_adversarial
│  └─ app/                   app.py static/index.html
└─ docs/                     presentation_outline.md adversarial_notes.md slides.pdf
```