from tkinter import *

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.gpt2.generate_essay import generate_beam_search, generate_greedy, generate_sampling_top_k
from src.gui.global_var import Global_Var
from src.mta.MTA_Wrapper import *
from src.seq2seq.Seq2SeqModel import *

OPTION_MODELS = [
    "seq2seq",
    "mta-lstm",
    "gpt2",
    "gpt2-v1",
    "gpt2-v2",
    "gpt2-v3"
]

OPTION_DECODER = [
    "greedy",
    "beam",
    "sampling"
]


def init_title(root: Tk):
    font_title = ("Bold", 30)
    color_bg = '#ffffff'
    color_fg = "purple"
    text_title = "EasyEssay"

    title = Label(root, text=text_title, font=font_title, bg=color_bg, fg=color_fg)

    return title


def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True


def get_text_essay_mta(topic, type_decoder, beamWidth, top_k, temperature, max_length):
    text = ""
    if type_decoder == "greedy":
        text = Global_Var.model_mat.predict_greedy(topic, max_length)
    elif type_decoder == "beam":
        text = Global_Var.model_mat.predict_beam_search(topic, beam_width=beamWidth, max_len_predict=max_length)
    elif type_decoder == "sampling":
        text = Global_Var.model_mat.predict_sampling(topic, k=top_k, temperature=temperature,
                                                     max_len_predict=max_length)

    return text


def get_text_essay_seq2seq(topic, type_decoder, beamWidth, top_k, temperature):
    text = ""
    if type_decoder == "greedy":
        text = Global_Var.model_seq2seq.predict_greedy(topic)
    elif type_decoder == "beam":
        text = Global_Var.model_seq2seq.predict_beam_search(topic, beam_width=beamWidth)
    elif type_decoder == "sampling":
        text = Global_Var.model_seq2seq.predict_sampling(topic, k=top_k, temperature=temperature)

    return text


def get_text_essay_gpt2(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, topic, type_decoder, beamWidth, top_k,
                        temperature, max_length):
    text = ""
    min_generate = 200
    max_min_limit = 300
    max_length = max(max_min_limit, max_length)
    no_repeat_ngram_size = 3
    topic = topic + "<|endoftext|>"

    if type_decoder == "greedy":
        text = generate_greedy(model, tokenizer, topic, min_generate, max_length)
    elif type_decoder == "beam":
        text = generate_beam_search(model, tokenizer, topic, min_generate, max_length, beamWidth)

    elif type_decoder == "sampling":

        text = generate_sampling_top_k(model, tokenizer, topic, min_generate, max_length, no_repeat_ngram_size, top_k,
                                       temperature)

    return text


def generate_essay():
    topics = Global_Var.input_topics.get("1.0", "end-1c")
    model_generation = Global_Var.model_var.get()
    type_generation = Global_Var.decoder_var.get()

    length_len = Global_Var.input_length.get("1.0", "end-1c")
    if length_len == "" or length_len is None or not length_len.isnumeric():
        length_len = 50
    else:
        length_len = max(int(length_len), 50)

    beamWidth = Global_Var.input_beam.get("1.0", "end-1c")
    if beamWidth == "" or beamWidth is None or not beamWidth.isnumeric():
        beamWidth = 3
    else:
        beamWidth = max(int(beamWidth), 1)

    top_k = Global_Var.input_k.get("1.0", "end-1c")
    if top_k == "" or top_k is None or not top_k.isnumeric():
        top_k = 7
    else:
        top_k = max(int(top_k), 1)

    temperature = Global_Var.input_temperature.get("1.0", "end-1c")
    if temperature == "" or temperature is None or not isfloat(temperature):
        temperature = 0.7
    else:
        temperature = min(max(float(temperature), 0.1), 1.0)

    text = "random random random"
    if model_generation == "seq2seq":
        text = get_text_essay_seq2seq(topics, type_generation, beamWidth, top_k, temperature)
    elif model_generation == "mta-lstm":
        text = get_text_essay_mta(topics, type_generation, beamWidth, top_k, temperature, length_len)
    else:

        if model_generation == "gpt2":

            text = get_text_essay_gpt2(Global_Var.model_gpt2, Global_Var.model_gpt2_token, topics, type_generation,
                                       beamWidth, top_k, temperature, length_len)
        elif model_generation == "gpt2-v1":

            text = get_text_essay_gpt2(Global_Var.model_gpt2_v1, Global_Var.model_gpt2_v1_token, topics,
                                       type_generation,
                                       beamWidth, top_k, temperature, length_len)
        elif model_generation == "gpt2-v2":

            text = get_text_essay_gpt2(Global_Var.model_gpt2_v2, Global_Var.model_gpt2_v2_token, topics,
                                       type_generation,
                                       beamWidth, top_k, temperature, length_len)
        elif model_generation == "gpt2-v3":

            text = get_text_essay_gpt2(Global_Var.model_gpt2_v3, Global_Var.model_gpt2_v3_token, topics,
                                       type_generation,
                                       beamWidth, top_k, temperature, length_len)

    Global_Var.display_text.delete('1.0', END)
    Global_Var.display_text.insert(END, text)


def init_model():
    # MTA
    Global_Var.model_mat = MTA_Wrapper()
    Global_Var.model_mat.restore()

    # Seq2Seq
    Global_Var.model_seq2seq = Seq2SeqMode()
    Global_Var.model_seq2seq.last_model()

    # GPT2
    Global_Var.model_gpt2 = GPT2LMHeadModel.from_pretrained("../../model/gpt2/gpt2-v2-2/")
    Global_Var.model_gpt2_token = GPT2Tokenizer.from_pretrained("../../model/gpt2/gpt2-v2-2/")

    Global_Var.model_gpt2_v1 = GPT2LMHeadModel.from_pretrained("../../model/gpt2/gpt2-v2-3/")
    Global_Var.model_gpt2_v1_token = GPT2Tokenizer.from_pretrained("../../model/gpt2/gpt2-v2-3/")

    Global_Var.model_gpt2_v2 = GPT2LMHeadModel.from_pretrained("../../model/gpt2/gpt2-v4-2/")
    Global_Var.model_gpt2_v2_token = GPT2Tokenizer.from_pretrained("../../model/gpt2/gpt2-v4-2/")

    Global_Var.model_gpt2_v3 = GPT2LMHeadModel.from_pretrained("../../model/gpt2/gpt2-v5-2/")
    Global_Var.model_gpt2_v3_token = GPT2Tokenizer.from_pretrained("../../model/gpt2/gpt2-v5-2/")


def run_app():
    init_model()

    name_title = "Easy Essay"
    size_windows = "1000x700"

    root = Tk()
    root.title(name_title)
    root.geometry(size_windows)
    root.config(bg='#FFFFFF')

    Global_Var.root = root

    # Title text
    title = init_title(root)
    title.pack()
    title.place(relx=0.5, rely=0.1, anchor=CENTER)

    # # Text of essay
    text_box = Text(root, wrap=WORD, width=110, height=12, fg='blue')
    text_box.pack()
    text_box.place(relx=0.5, rely=0.7, anchor=CENTER)

    Global_Var.display_text = text_box

    # input topics
    label_topics = Label(root, text="Topic:", font=("Times New Roman", 20), bg="#ffffff", fg="#0080ff")
    label_topics.pack()
    label_topics.place(relx=0.05, rely=0.2)

    text_box_topics = Text(root, width=100, height=1, fg="#800040")
    text_box_topics.pack()
    text_box_topics.place(relx=0.5, rely=0.3, anchor=CENTER)

    x_pos = 0.05
    x_delta = 0.16
    y_pos_1 = 0.44
    y_pos_2 = 0.46
    # for beam
    label_beam = Label(root, text="BeamWidth:", font=("Times New Roman", 15), bg="#ffffff", fg="#0080ff")
    label_beam.pack()
    label_beam.place(relx=x_pos + 0 * x_delta, rely=y_pos_1)

    text_box_beam = Text(root, width=10, height=1, fg="#0080ff")
    text_box_beam.pack()
    text_box_beam.place(relx=x_pos + 1 * x_delta, rely=y_pos_2, anchor=CENTER)

    # # for top-k sampling
    label_k = Label(root, text="Top-K:", font=("Times New Roman", 15), bg="#ffffff", fg="#0080ff")
    label_k.pack()
    label_k.place(relx=x_pos + 2 * x_delta, rely=y_pos_1)

    text_box_k = Text(root, width=10, height=1, fg="#0080ff")
    text_box_k.pack()
    text_box_k.place(relx=x_pos + 3 * x_delta, rely=y_pos_2, anchor=CENTER)

    label_temperature = Label(root, text="Temperature:", font=("Times New Roman", 15), bg="#ffffff", fg="#0080ff")
    label_temperature.pack()
    label_temperature.place(relx=x_pos + 4 * x_delta, rely=y_pos_1)

    text_temperature = Text(root, width=10, height=1, fg="#0080ff")
    text_temperature.pack()
    text_temperature.place(relx=x_pos + 5 * x_delta, rely=y_pos_2, anchor=CENTER)

    # go to global
    Global_Var.input_beam = text_box_beam
    Global_Var.input_topics = text_box_topics
    Global_Var.input_k = text_box_k
    Global_Var.input_temperature = text_temperature

    # popup menu
    # model
    label_model = Label(root, text="Model:", font=("Times New Roman", 15), bg="#ffffff", fg="green")
    label_model.pack()
    label_model.place(relx=x_pos + 0 * x_delta, rely=0.35)

    model = StringVar(root)
    model.set(OPTION_MODELS[0])
    model_label = OptionMenu(root, model, *OPTION_MODELS)
    model_label.config(bg="#FFFFFF", width=6, fg="green")
    model_label.pack()
    model_label.place(relx=x_pos + 1 * x_delta, rely=0.37, anchor=CENTER)
    Global_Var.model_var = model

    #  decoder
    label_model = Label(root, text="Decoder:", font=("Times New Roman", 15), bg="#ffffff", fg="green")
    label_model.pack()
    label_model.place(relx=x_pos + 2 * x_delta, rely=0.35)

    decoder = StringVar(root)
    decoder.set(OPTION_DECODER[0])
    decoder_label = OptionMenu(root, decoder, *OPTION_DECODER)
    decoder_label.config(bg="#FFFFFF", fg="green")
    decoder_label.pack()
    decoder_label.place(relx=x_pos + 3 * x_delta, rely=0.37, anchor=CENTER)
    Global_Var.decoder_var = decoder

    # max length
    label_max = Label(root, text="MaxLength:", font=("Times New Roman", 15), bg="#ffffff", fg="green")
    label_max.pack()
    label_max.place(relx=x_pos + 4 * x_delta, rely=0.35)

    text_box_max = Text(root, width=10, height=1, fg="#0080ff")
    text_box_max.pack()
    text_box_max.place(relx=x_pos + 5 * x_delta, rely=0.37, anchor=CENTER)
    Global_Var.input_length = text_box_max

    # button
    button_generate = Button(root, text="Generate Essay", font=("Helvetica", 15), width=10, command=generate_essay)
    button_generate.pack()
    button_generate.place(rely=0.9, relx=0.5, anchor=CENTER)
    button_generate.config(bg='#FFFFFF', fg='blue', activebackground="#33B5E5", relief=SOLID)

    root.mainloop()


if __name__ == '__main__':
    run_app()
