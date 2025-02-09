from types import MethodType
import hydra
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training
from utils import (
    FrozenModelSentenceGivenPrompt,
)
from lightning_module import NextSentenceGFNTask
from lightning_data import PromptDataModule


@hydra.main(version_base=None, config_path="./configs/", config_name="train")
def train(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)

    model, tokenizer = get_model(config)

    try:
        end_of_sentence_token_id = tokenizer.encode(
            "A sentence\n", add_special_tokens=False
        )[-1]
    except:
        end_of_sentence_token_id = tokenizer.convert_tokens_to_ids("\n")
    
    print(end_of_sentence_token_id)

    reward = get_reward(config, end_of_sentence_token_id, tokenizer)
    category=config.task.training.category

    data = PromptDataModule(
        data_path=config.task.data.path,
        tokenizer=tokenizer,
        train_size=config.task.data.train_size,
        batch_size=config.task.training.batch_size,
        category=category,
    )
    data.setup("fit")
    possible_output = []
    with open(config.task.reward.title_logp_path, 'r', encoding='utf-8') as file:
        for line in file:
            possible_output.append(line.strip())
    
    task = NextSentenceGFNTask(
        model=model,
        tokenizer=tokenizer,
        reward=reward,
        lr=config.task.training.lr,
        subtb_lambda=config.task.training.subtb_lambda,
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,
        batch_size=config.task.training.batch_size,
        min_sentence_len=config.task.constraints.min_sentence_len,
        max_sentence_len=config.task.constraints.max_sentence_len,
        use_4bit=config.task.training.use_4bit,
        valid_path = config.task.training.valid_path,
        test_data_path=config.task.data.inference_path,
        title_logp_path = config.task.reward.title_logp_path,
        possible_output=possible_output,
        real_token_distri=config.task.data.token_distri_compute,
        category=category,
        token_distri=config.task.data.token_distri,
        threshold = config.task.training.threshold
    )
    
    trainer = pl.Trainer(
        accelerator=config.device.accelerator,
        max_epochs=config.task.training.epochs,
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,
        logger=config.logger
        if isinstance(config.logger, bool)
        else hydra.utils.instantiate(config.logger),
        callbacks=[hydra.utils.instantiate(c) for c in config.task.callbacks],
    )

    if config.task.training.use_4bit:
        task.to = MethodType(lambda s, _: s, task)
        task.cuda = MethodType(lambda s: s, task)

    trainer.fit(model=task, datamodule=data)


def get_model(config: DictConfig):
    # Use 4-bit quantization for lower memory use
    if config.task.training.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Get the model
    tokenizer = AutoTokenizer.from_pretrained(
        config.task.model.path, add_bos_token=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.task.model.path, device_map="auto", quantization_config=bnb_config
    )

    # Prepare model for k-bit training
    if config.task.training.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )
    return model, tokenizer


def get_reward(config: DictConfig, sentence_token_id, tokenizer):

    reward = FrozenModelSentenceGivenPrompt(
        sentence_token_id=sentence_token_id,
        min_len=config.task.constraints.min_sentence_len,
        title_logp_path = config.task.reward.title_logp_path,
        tokenizer = tokenizer,
        real_token_distri=config.task.data.token_distri
    )
    return reward


if __name__ == "__main__":
    train()
