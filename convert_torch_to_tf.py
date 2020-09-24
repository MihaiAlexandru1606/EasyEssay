import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import GPT2LMHeadModel, TFGPT2LMHeadModel

def compute_name(tokens_name, switch_table):
    name_return = "tfgp_t2lm_head_model/transformer/h_._{}/".format(tokens_name[2])
    remain_token = tokens_name[3:-1]
    name_return += "/".join(remain_token)
    name_return += "/" + switch_table[tokens_name[-1]]

    return name_return


def convert_name_torch_tf(name_torch):
    simple_convert = {
        "transformer.ln_f.weight": "tfgp_t2lm_head_model/transformer/ln_f/gamma:0",
        "transformer.ln_f.bias": "tfgp_t2lm_head_model/transformer/ln_f/beta:0",
        "transformer.wte.weight": "tfgp_t2lm_head_model/transformer/wte/weight:0",
        "transformer.wpe.weight": "tfgp_t2lm_head_model/transformer/wpe/embeddings:0"
    }

    if name_torch in simple_convert:
        return simple_convert[name_torch]

    if ".attn.bias" in name_torch or "attn.masked_bias" in name_torch or "lm_head.weight" in name_torch:
        return ""

    switch_ln = {"weight": "gamma:0", "bias": "beta:0"}
    switch = {"weight": "weight:0", "bias": "bias:0"}

    token_name = name_torch.split(".")
    if "ln_1" in token_name or "ln_2" in token_name:
        return compute_name(token_name, switch_ln)
    else:
        return compute_name(token_name, switch)


def convert_torch_tf(path_model):
    model = GPT2LMHeadModel.from_pretrained(path_model)
    model_tf = TFGPT2LMHeadModel.from_pretrained("gpt2")

    for name, value in model.state_dict().items():
        for i in range(len(model_tf.trainable_variables)):
            if model_tf.trainable_variables[i].name == convert_name_torch_tf(name):
                value_np = value.numpy()
                value_np = value_np.reshape(model_tf.trainable_variables[i].shape)
                model_tf.trainable_variables[i].load(value_np)

    director = "tf_model/" + path_model.split("/")[-1]
    if not os.path.exists(director):
        os.makedirs(director)

    model_tf.save_pretrained(director)


if __name__ == '__main__':
    path_model = [
        "./model/gpt2/gpt2-romana",
        "./model/gpt2/gpt2-v2",

        "./model/gpt2/gpt2-v2-2",
        "./model/gpt2/gpt2-v2-3",

        "./model/gpt2/gpt2-v3-2",
        "./model/gpt2/gpt2-v3-3",

        "./model/gpt2/gpt2-v4-2",
        "./model/gpt2/gpt2-v4-3",

        "./model/gpt2/gpt2-v5-2",
        "./model/gpt2/gpt2-v5-3",
    ]

    for path in path_model:
        convert_torch_tf(path)

