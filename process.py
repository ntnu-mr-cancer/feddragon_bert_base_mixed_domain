import re
from typing import List, Union
from dragon_baseline import DragonBaseline
from transformers import EarlyStoppingCallback

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
        # self.update_default_training_settings(self.model_kwargs)

    def custom_text_cleaning(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Perform custom text cleaning on the input text.

        Args:
            text (Union[str, List[str]]): The input text to be cleaned. It can be a string or a list of strings.

        Returns:
            Union[str, List[str]]: The cleaned text. If the input is a string, the cleaned string is returned.
            If the input is a list of strings, a list of cleaned strings is returned.

        """
        if isinstance(text, str):
            # Remove HTML tags and URLs:
            text = re.sub(r"<.*?>", "", text)
            text = re.sub(r"http\S+", "", text)

            return text
        else:
            # If text is a list, apply the function to each element
            return [self.custom_text_cleaning(t) for t in text]

    def preprocess(self):
        # Example of how to adapt the DRAGON baseline to use a different preprocessing function
        super().preprocess()

        # Uncomment the following lines to use the custom_text_cleaning function
        # for df in [self.df_train, self.df_val, self.df_test]:
        #     df[self.task.input_name] = df[self.task.input_name].map(self.custom_text_cleaning)


if __name__ == "__main__":
    mdl = DragonSubmission()
    mdl.setup()
    mdl.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01))
    mdl.train()
    predictions = mdl.predict(df=mdl.df_test)
    mdl.save(predictions)
    mdl.verify_predictions()