# текст саммари находится в results/cnndm.-1.candidate
f = open("../results/cnndm.-1.candidate", "r")
if f.mode == 'r':
    out_text = f.read()
from nltk.tokenize import sent_tokenize

out_text = out_text.replace('<q>', '. ')
input_sen = out_text  # 'hello! how are you? please remember capitalization. EVERY time.'
sentences = sent_tokenize(input_sen)
sentences = [sent.capitalize() for sent in sentences]
print(sentences)
text_summary = ' '.join([str(elem) for elem in sentences])

print(text_summary)