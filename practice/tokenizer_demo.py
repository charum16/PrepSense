from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sentence = "The bank by the river is not a financial bank."
tokens = tokenizer.tokenize(sentence)

print("Original sentence:")
print(sentence)
print("\nTokens:")
print(tokens)
print("\nNumber of tokens:", len(tokens))
print("\n--- Subword tokenization ---")
words = ["unbelievable", "tokenization", "transformers", "PrepSense"]
for word in words:
    tokens = tokenizer.tokenize(word)
    print(f"{word} → {tokens}")


from transformers import AutoModel
import torch
from bertviz import head_view
from IPython.display import HTML

model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

inputs = tokenizer("The bank by the river is not a financial bank.", return_tensors="pt")
outputs = model(**inputs)

attention = outputs.attentions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

html_obj = head_view(attention, tokens, html_action='return')

with open("practice/attention_viz.html", "w") as f:
    f.write(html_obj.data)

print("Saved! Open practice/attention_viz.html in your browser.")