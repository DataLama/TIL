from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase, BatchEncoding, default_data_collator, DataCollatorWithPadding
from lightning_transformers.core.nlp import HFDataModule

class TextClassificationDataModule(HFDataModule):
    """
    Defines the ``LightningDataModule`` for Text Classification Datasets.
    
    Args:
        tokenizer: ``PreTrainedTokenizerBase`` for tokenizing data.
        cfg: Contains data specific parameters when processing/loading the dataset (Default ``HFTransformerDataConfig``)
            -> 
            HFTransformerDataConfig + 
    
    reference
        https://github.com/PyTorchLightning/lightning-transformers/blob/master/lightning_transformers/task/nlp/text_classification/data.py
    """
    
    def process_data(self,
                     dataset: Union[Dataset, DatasetDict],
                     stage: Optional[str] = None) -> Union[Dataset, DatasetDict]:
        """Process for glue-style text classification."""
        dataset = TextClassificationDataModule.preprocess(
            dataset,
            tokenizer = self.tokenizer,
            label2id = self.label2id,
            cfg=self.cfg
        )
        
        cols_to_keep = [
            x for x in ["input_ids", "attention_mask", "token_type_ids", "labels"] if x in dataset["train"].features
        ]
        dataset.set_format("torch", columns=cols_to_keep)
        self.labels = dataset["train"].features["labels"]
        return dataset
    
    @property
    def id2label(self) -> Dict:
        return {i:l for i, l in enumerate(self.cfg.id2label)}
    
    @property
    def label2id(self) -> Dict:
        return {l:i for i, l in enumerate(self.cfg.id2label)}
    
    @property
    def num_classes(self) -> int:
        return len(self.cfg.id2label)
    
    @property
    def model_data_kwargs(self) -> Dict[str, int]:
        return {"num_labels": self.num_classes}
    
    @staticmethod
    def convert_to_features(
        examples: Any, _, tokenizer: PreTrainedTokenizerBase, label2id, cfg
    ) -> BatchEncoding:
        # Tokenize the text (glue style)
        args = (
            (examples[cfg.sentence1_key],) if cfg.sentence2_key is None else (examples[cfg.sentence1_key], examples[cfg.sentence2_key])
        )
        result = tokenizer(*args, padding=cfg.padding, max_length=cfg.max_length, truncation=cfg.truncation)
        
        # Map the labels
        if isinstance(examples[cfg.label_key], str):
            labels = label2id[examples[cfg.label_key]] if not cfg.is_regression else float(examples[cfg.label_key])
        elif isinstance(examples[cfg.label_key], list):
            labels = list(map(lambda l: label2id[l] if not cfg.is_regression else float(l), examples[cfg.label_key]))

        # update the label
        result['labels'] = labels
        return result
    
    @staticmethod
    def preprocess(ds: Dataset, **fn_kwargs) -> Dataset:
        ds = ds.map(
            # todo: change this to self.convert_to_features for users to override
            TextClassificationDataModule.convert_to_features,
            batched=True,
            with_indices=True,
            fn_kwargs=fn_kwargs,
        )
        return ds
    
    @property
    def collate_fn(self) -> Optional[Callable]:
        if self.cfg.padding == 'max_length':
            return default_data_collator
        else:
            return DataCollatorWithPadding(self.tokenizer) # todo update for case training_args.fp16