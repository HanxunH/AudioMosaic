import os
import torchaudio
import fire
import json
import torch
import time
import argparse
from peft_custom import (
    LoraConfig,
    get_peft_model,
)
from models.modeling_ltu import LTULlamaForCausalLM, AudioMosaicLlamaForCausalLM, LlamaConfig
from modeling_ltu.transformers import LlamaTokenizer, GenerationConfig
from util import Prompter
from tqdm import tqdm
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
parser.add_argument('--base_model', default='checkpoints/ltu_pretrained_mdls', type=str)
parser.add_argument('--model_path', default='checkpoints/ltu_ori_paper.bin', type=str)
parser.add_argument('--model_type', default='LTU', type=str)
parser.add_argument('--audio_encoder_path', default='experiments/AudioMosaic/pretrain/checkpoints/model_state_dict.pt', type=str)
parser.add_argument('--prompt_template', default='alpaca_short', type=str)
parser.add_argument('--eval_dataset', default='clotho', type=str)
parser.add_argument('--output_dir', default='ltu_inference_outputs', type=str)
parser.add_argument('--debug', action='store_true', help='debug mode')
task_dict = {
    'clotho': {
        "data_path": '/PATH/TO/datasets/openaqa/ltu_eval_data/clotho_caption_evaluation_prep.json',
        "instruction": 'Close-ended question: Create a caption for audio, in Clotho style.',
    },
    'audiocaps': {
        "data_path": '/PATH/TO/datasets/openaqa/ltu_eval_data/audiocaps_test_prep.json',
        "instruction": 'Close-ended question: Write an audio caption describing the sound, in AudioCaps style.',
    },
    'as':{
        "data_path": '/PATH/TO/datasets/openaqa/ltu_eval_data/audioset_eval_prep.json',
        "instruction": 'Close-ended question: Write an audio caption describing the sound.',
    },
    'esc':{
        "data_path": '/PATH/TO/datasets/openaqa/ltu_eval_data/esc50_prep.json',
        "instruction": 'Close-ended question: Write an audio caption describing the sound.',
    },
    'vs':{
        "data_path": '/PATH/TO/datasets/openaqa/ltu_eval_data/vocalsound_test_prep.json',
        "instruction": 'Close-ended question: Write an audio caption describing the sound.',
    },
    'tut':{
        "data_path": '/PATH/TO/datasets/openaqa/ltu_eval_data/tut_17_prep.json',
        "instruction": 'Close-ended question: Write an audio caption describing the sound.',
    },
    'fsd':{
        "data_path": '/PATH/TO/datasets/openaqa/ltu_eval_data/fsd50k_eval_prep.json',
        "instruction": 'Close-ended question: Write an audio caption describing the sound.',
    },
    'bjo':{
        "data_path": '/PATH/TO/datasets/openaqa/ltu_eval_data/beijing_opera_prep.json',
        "instruction": 'Close-ended question: Write an audio caption describing the sound.',
    },
    'vgg':{
        "data_path": '/PATH/TO/datasets/openaqa/ltu_eval_data/vggsound_eval_prep.json',
        "instruction": 'Close-ended question: Write an audio caption describing the sound.',
    },
    'dcase':{
        "data_path": '/PATH/TO/datasets/openaqa/ltu_eval_data/dcase17_prep.json',
        "instruction": 'Close-ended question: Write an audio caption describing the sound.',
    }
}

def load_audio(filename):
    waveform, sr = torchaudio.load(filename)
    if waveform.shape[1] < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000
    # convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                              use_energy=False, window_type='hanning',
                                              num_mel_bins=128, dither=0.0, frame_shift=10)
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    # normalize the fbank
    fbank = (fbank + 5.081) / 4.4849
    # fbank = (fbank + 4.2677393) / 4.5689974
    return fbank

def main(args):
    base_model = args.base_model
    prompter = Prompter(args.prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    
    if args.model_type == 'LTU':
        model = LTULlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
    elif args.model_type == 'AudioMosaic-LTU':
        model = AudioMosaicLlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
        current_model_dict = model.model.audio_encoder.state_dict()
        loaded_state_dict = torch.load(args.audio_encoder_path, map_location='cpu', weights_only=False)
        filtered_state_dict = {}
        for k, v in loaded_state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "_orig_mod." in k:
                k = k.replace("_orig_mod.", "")
            # if "fc_norm" in k:
            #     k = k.replace("fc_norm", "norm")
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
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    temp, top_p, top_k = 0.1, 0.95, 500
    
    # change it to your model path
    eval_mdl_path = args.model_path
    state_dict = torch.load(eval_mdl_path, map_location='cpu')
    # print(state_dict.keys())
    for k in list(state_dict.keys()):
        print(f"Loaded parameter: {k}, shape: {state_dict[k].shape}")
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model = model.to(device)
    model.eval()

    result_json = []
    result_json_filename = f'{(args.model_path).replace("/", "_")}_{args.eval_dataset}_results.json'
    data_json = json.load(open(task_dict[args.eval_dataset]["data_path"], "r"))
    for data_point in tqdm(data_json):
        # instruction = data_point["instruction"]
        instruction = task_dict[args.eval_dataset]["instruction"]
        prompt = prompter.generate_prompt(instruction, None)
        # print('Input prompt: ', prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        audio_fbank = load_audio(data_point["audio_id"]).unsqueeze(0).to(device).to(torch.float16)
        # print("audio_fbank shape:", audio_fbank.shape)
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,
            max_new_tokens=400,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            num_return_sequences=1
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                audio_input=audio_fbank,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=400,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)[6:-4]

        result_json.append(
            {
                'prompt': instruction, 
                'pred': output[len(prompt):], 
                'ref': data_point["output"], 
                'audio_id': data_point["audio_id"]
            }
        )
        if args.debug:
            print("audio_id:", data_point["audio_id"])
            print("ref:", data_point["output"])
            print("pred:", output[len(prompt):])
        # print("Audio ID:", data_point["audio_id"])
        # print("Generated Caption:", prompter.get_response(output))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, result_json_filename), 'w') as f:
        json.dump(result_json, f, indent=4)
        
    return 


if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)