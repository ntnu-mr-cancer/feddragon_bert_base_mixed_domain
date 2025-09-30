# FedDRAGON BERT Base Mixed-domain Centralized Algorithm

Adaptation of the updated [DRAGON baseline](https://github.com/ntnu-mr-cancer/dragon_baseline) (version `0.4.6`), with pretrained foundational model `joeranbosma/dragon-bert-base-mixed-domain`.

For details on the pretrained foundational model, check out HuggingFace: [huggingface.co/joeranbosma/dragon-bert-base-mixed-domain](https://huggingface.co/joeranbosma/dragon-bert-base-mixed-domain).

The following adaptations were made to the DRAGON baseline:

```python
class DragonSubmission(DragonBaseline):
    def __init__(self, **kwargs):
        # Example of how to adapt the DRAGON baseline to use a different model
        """
        Adapt the DRAGON baseline to use the joeranbosma/dragon-roberta-base-mixed-domain model.
        Note: when changing the model, update the Dockerfile to pre-download that model.
        """
        super().__init__(**kwargs)
        # self.model_name_or_path = "joeranbosma/dragon-roberta-large-domain-specific"
        self.model_name_or_path = "joeranbosma/dragon-bert-base-mixed-domain"
        self.per_device_train_batch_size = 8
        self.gradient_accumulation_steps = 1
        self.gradient_checkpointing = False
        self.max_seq_length = 512
        self.learning_rate = 1e-05
        self.num_train_epochs = 15

        self.model_kwargs.update({
             "lr_scheduler_type": "constant",
             "save_total_limit": 1,
             "warmup_ratio": 0,

             # precision/math
             "optim": "adamw_torch_fused",
             "use_liger_kernel" : True,
             "fp16" : True,

             # logging/checkpointing
             "logging_steps": 1000,
             "save_strategy": "best",
             "save_only_model": True,
             "load_best_model_at_end": True,
             "metric_for_best_model": "dragon",
             "greater_is_better": True,

         })
```
Additionally an early stopping callback was added to the trainer to further reduce unnecessary training time. The following parameters were used for the early stopping
```python
from transformers import EarlyStoppingCallback
EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
```

See [process.py](process.py) for implementation details.

**References:**

[1] J. S. Bosma, K. Dercksen, L. Builtjes, R. André, C, Roest, S. J. Fransen, C. R. Noordman, M. Navarro-Padilla, J. Lefkes, N. Alves, M. J. J. de Grauw, L. van Eekelen, J. M. A. Spronck, M. Schuurmans, A. Saha, J. J. Twilt, W. Aswolinskiy, W. Hendrix, B. de Wilde, D. Geijs, J. Veltman, D. Yakar, M. de Rooij, F. Ciompi, A. Hering, J. Geerdink, and H. Huisman on behalf of the DRAGON consortium. The DRAGON benchmark for clinical NLP. *npj Digital Medicine* 8, 289 (2025). [https://doi.org/10.1038/s41746-025-01626-x](https://doi.org/10.1038/s41746-025-01626-x)

[2] J. S. Bosma, K. Dercksen, L. Builtjes, R. André, C, Roest, S. J. Fransen, C. R. Noordman, M. Navarro-Padilla, J. Lefkes, N. Alves, M. J. J. de Grauw, L. van Eekelen, J. M. A. Spronck, M. Schuurmans, A. Saha, J. J. Twilt, W. Aswolinskiy, W. Hendrix, B. de Wilde, D. Geijs, J. Veltman, D. Yakar, M. de Rooij, F. Ciompi, A. Hering, J. Geerdink, H. Huisman, DRAGON Consortium (2024). DRAGON Statistical Analysis Plan (v1.0). Zenodo. https://doi.org/10.5281/zenodo.10374512

[3] PENDING FedDRAGON paper.