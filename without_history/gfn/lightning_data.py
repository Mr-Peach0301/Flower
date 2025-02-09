from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from lightning.pytorch import LightningDataModule
import json
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

class PromptDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer,
        category,
        train_size=0.95,
        limit_prompts=None,
        batch_size=32,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        self.tokenizer = tokenizer
        self.train_data = None
        self.batch_size = batch_size
        self.val_data = None
        self.category = category

    def setup(self, stage):
        with open(self.hparams.data_path, "r") as input_file:
            data = json.load(input_file)
        instructions = [_['instruction'] for _ in data]
        inputs = [_['input'] for _ in data]
        prompts = [generate_prompt(instruction, self.category, input) for instruction, input in zip(instructions, inputs)]
        if self.hparams.limit_prompts is not None:
            prompts = prompts[: self.hparams.limit_prompts]
        num_train = int(len(prompts) * self.hparams.train_size)
        self.train_data = PromptDataPipe(prompts[:num_train], self.tokenizer)
        self.val_data = PromptDataPipe(prompts[num_train:], self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=0)


class PromptDataPipe(MapDataPipe):
    def __init__(self, prompts, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompt = self.tokenizer(
            self.prompts[index],
            return_tensors="pt",
        )["input_ids"]
        return prompt
    
def generate_prompt(instruction, category, input=None):
    if input:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Please recommend a {category} to the user. Directly output the title of the {category}.
### Response:
"""