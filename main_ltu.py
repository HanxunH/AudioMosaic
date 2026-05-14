import warnings
import os
import logging
# Only silence specific noisy ones
warnings.filterwarnings(
    "ignore",
    message=r".*xformers\.components is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated.*please import via timm\.layers",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.optim\.optim_factory is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*xformers\.components is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Using `TRANSFORMERS_CACHE` is deprecated.*Use `HF_HOME` instead\.",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated.*please import via timm\.layers",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.",
    category=UserWarning,
)
# Silence DeepSpeed, Transformers, and other verbose libraries
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DEEPSPEED_LOG_LEVEL"] = "ERROR"
os.environ["DEEPSPEED_ZERO_LOGGING"] = "none"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Set Python logging levels
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import argparse
import torch
import mlconfig
import dataset
import models
import losses
import util
import misc
import os
import sys
import numpy as np
import time
from exp_mgmt import ExperimentManager
from models.modeling_ltu import LTULlamaForCausalLM, AudioMosaicLlamaForCausalLM
from modeling_ltu.transformers import LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from util import Prompter
from peft_custom import (
    LoraConfig,
    get_peft_model,
)
from datasets import load_dataset

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


parser = argparse.ArgumentParser(description='AudioMosaic')
# General Options
parser.add_argument('--seed', type=int, default=7, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
# distributed training parameters
parser.add_argument('--ddp', action='store_true', default=False)
parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
# Debugging Options
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)


def save_model(model, optimizer, epoch=None):
    # Save model
    exp.save_state(model, 'model_state_dict')
    exp.save_state(optimizer, 'optimizer_state_dict')
    if epoch is not None:
        exp.save_state(model, 'model_state_dict_epoch{:d}'.format(epoch))


def main():
    # Set up Experiments
    logger = exp.logger
    config = exp.config

    # Based on LTU codebase
    prompter = Prompter(config.prompt_template_name)
    if config.model_type == "LTU":
        model = LTULlamaForCausalLM.from_pretrained(
            config.base_model,
            load_in_8bit=False,
            device_map="auto",
        )
    elif config.model_type == "AudioMosaic-LTU":
        model = AudioMosaicLlamaForCausalLM.from_pretrained(
            config.base_model,
            load_in_8bit=False,
            device_map="auto",
        )
        
        if "audio_encoder_path" in config:
            current_model_dict = model.model.audio_encoder.state_dict()
            loaded_state_dict = torch.load(config.audio_encoder_path, map_location='cpu', weights_only=False)
            filtered_state_dict = {}
            for k, v in loaded_state_dict.items():
                if "module." in k:
                    k = k.replace("module.", "")
                if "_orig_mod." in k:
                    k = k.replace("_orig_mod.", "")
                if "fc_norm" in k:
                    k = k.replace("fc_norm", "norm")
                if k in current_model_dict:
                    if v.size() == current_model_dict[k].size():
                        filtered_state_dict[k] = v
                    else:
                        print(
                            f"loading parameter: {k}, required shape: {current_model_dict[k].size()}, loaded shape: {v.size()}",
                            flush=True,
                        )
                        if "pos_embed" in k:
                            # Interpolate position embedding to match current patch grid
                            try:
                                n_patches_old = v.shape[1] - 1
                                new_h, new_w = model.patch_embed.patch_hw
                                assert (
                                    n_patches_old % new_w == 0
                                ), f"Cannot infer original (T,F) from pos_embed of length {n_patches_old} with new F={new_w}"
                                orig_h = n_patches_old // new_w
                                orig_w = new_w
                                orig_size = (orig_h, orig_w)
                                new_size = (new_h, new_w)
                                temp = {"pos_embed": v}
                                interpolate_pos_embed_audio(model, temp, orig_size, new_size)
                                filtered_state_dict[k] = temp["pos_embed"]
                                print(
                                    f"Interpolated {k}: ({orig_h},{orig_w}) -> ({new_h},{new_w})",
                                    flush=True,
                                )
                            except Exception as e:
                                print(f"Failed to interpolate {k}: {e}", flush=True)
                                # Skip if interpolation fails
                        else:
                            # Skip keys with mismatched shapes
                            pass
                else:
                    # Key not in current model; ignore
                    print(f"loading parameter: {k}, not in current model", flush=True)

            msg = model.model.audio_encoder.load_state_dict(filtered_state_dict, strict=False)
            print(msg)
            # print(f"Loaded audio encoder from {config.audio_encoder_path}, msg: {msg}")

    tokenizer = LlamaTokenizer.from_pretrained(config.base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )

    tokenizer.padding_side = "left"  # Allow batched inference
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=config.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < config.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"]
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not config.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=config.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if config.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # for audio params, lora always trainable, llama always frozen
    for name, param in model.named_parameters():
        if exp.config.trainable_params == 'all':
            if "audio" in name:
                param.requires_grad = True
                #print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        if exp.config.trainable_params == 'proj':
            if "audio_proj" in name:
                param.requires_grad = True
                #print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        if exp.config.trainable_params == 'proj+norm':
            if "audio_proj" in name:
                param.requires_grad = True
                print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
            if "audio_encoder.norm" in name and config.model_type == "AudioMosaic-LTU":
                param.requires_grad = True
                print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    if exp.config.data_path.endswith(".json") or exp.config.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=exp.config.data_path)
    else:
        data = load_dataset(exp.config.data_path)

    if config.resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            config.resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                config.resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            config.resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            state_dict = torch.load(checkpoint_name, map_location='cpu')
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if "model_path" in config:
        state_dict = torch.load(config.model_path, map_location='cpu')
        print(state_dict.keys(), flush=True)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded AudioMosaic-LTU model weights from {config.model_path}, msg: {msg}")
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if config.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=config.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    use_wandb = True
    wandb_run_name = args.exp_path + args.exp_name

    gradient_accumulation_steps = config.batch_size // config.micro_batch_size
    if args.ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // args.world_size
    try:
        dataloader_num_workers = os.environ['SLURM_CPUS_PER_TASK']
        if dataloader_num_workers is not None:
            dataloader_num_workers = int(dataloader_num_workers)
        print('setting n_workers base on SLURM, n_workers is {}'.format(dataloader_num_workers))
    except:
        dataloader_num_workers = 8
        print('setting n_workers base on SLURM failed, n_workers is {}'.format(8))
    
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=config.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="steps",
            eval_steps=None,
            save_steps=config.save_steps,
            dataloader_num_workers=dataloader_num_workers,
            output_dir=exp.checkpoint_path,
            save_total_limit=50,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if args.ddp else None,
            group_by_length=config.group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            remove_unused_columns=False,
            # ddp_backend="nccl" if args.ddp else None,
            # Pass integer local rank (GPU index) to Trainer
            local_rank=args.gpu if args.ddp else -1,
            # project="AudioMosaic-LTU",
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    model.save_pretrained(exp.checkpoint_path)

    return 


if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    if args.ddp:
        if 'SLURM_PROCID' in os.environ:
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        else:
            print('Not using distributed mode')
        # Set correct env vars: process rank and local GPU index
        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        # Pin this process to the selected GPU if CUDA is available
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(args.gpu)
            except Exception as e:
                print(f"Warning: failed to set CUDA device {args.gpu}: {e}")
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.manual_seed(args.seed)
        seed = args.seed
    # Keep args.gpu as an integer (local rank) for DDP; do not overwrite with device
    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(
        exp_name=args.exp_name, exp_path=args.exp_path,
        config_file_path=config_filename, args=args,
        enable_online_log=False,
    )
    
    if misc.get_rank() == 0:
        logger = experiment.logger
        logger.info("PyTorch Version: %s" % (torch.__version__))
        logger.info("Python Version: %s" % (sys.version))
        try:
            logger.info('SLURM_NODELIST: {}'.format(os.environ['SLURM_NODELIST']))
        except:
            pass
        if torch.cuda.is_available():
            device_list = [torch.cuda.get_device_name(i)
                           for i in range(0, torch.cuda.device_count())]
            logger.info("GPU List: %s" % (device_list))
        for arg in vars(args):
            logger.info("%s: %s" % (arg, getattr(args, arg)))
        for key in experiment.config:
            logger.info("%s: %s" % (key, experiment.config[key]))
    start = time.time()
    exp = experiment
    main()
    end = time.time()
    cost = (end - start) / 86400
    if misc.get_rank() == 0:
        payload = "Running Cost %.2f Days" % cost
        logger.info(payload)
    # if args.ddp: 
        # misc.destroy_process_group()
