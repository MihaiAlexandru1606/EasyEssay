from rb.processings.ro_corrections.ro_correct import correct_text_ro
import nltk.translate.bleu_score as bleu
from nltk.translate.meteor_score import meteor_score as meteor
from rouge import Rouge
import unicodedata
import re


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()

    return w


def get_print_mistake(text):
    output_correct_text_ro = correct_text_ro(text, True)
    tokens = output_correct_text_ro["split_text"]
    line_mistake = ""

    for correction in output_correct_text_ro['corrections']:

        type_mistake = 'Type : ' + correction['mistake'] + '\n'

        token_mistakes = []
        for index in correction['index']:
            if index[0] != 0:
                continue
            if index[1] > len(tokens[0]) - 1:
                continue
            token_mistakes.append(tokens[index[0]][index[1]])

        mistake = "\tMistake : " + " ".join(token_mistakes) + "\n"
        line_mistake += type_mistake + mistake

    return line_mistake


def get_ref(path_reference="../../dataset/valid/valid_text.txt", max_len=200, processing=False):
    reference = []
    with open(path_reference, 'r') as file_read:
        data = file_read.read()
        if processing:
            data = preprocess_sentence(data)
        token_data = data.split(" ")
        reference.append(" ".join(token_data[:max_len]))
    return reference


def double_print(strings, log):
    print(*strings)
    print(*strings, file=log)


def eval_essay(hypothesis, log_file, path_reference="../../dataset/valid/valid_text.txt", max_len=200,
               processing=False):
    references = get_ref(path_reference, max_len, processing)
    log_file = "../../log/" + log_file
    log = open(log_file, 'w+')
    h_new = []
    rouge = Rouge()
    for _ in range(len(references)): h_new.append(hypothesis)

    text = ["For test : \n", hypothesis, "\n1\n\n\n"]
    corect_text = [get_print_mistake(hypothesis)]
    bleu_score = bleu.sentence_bleu(references, hypothesis)
    meteor_score = meteor(references, hypothesis)
    rouge_score = rouge.get_scores(h_new, references, avg=True)

    double_print(text, log)
    double_print(corect_text, log)
    double_print(["Bleu {}".format(bleu_score)], log)
    double_print(["meteor {}".format(meteor_score)], log)
    double_print(["Rouge {}".format(rouge_score)], log)
