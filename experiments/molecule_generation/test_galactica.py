import pdb

from transformers import AutoTokenizer, OPTForCausalLM
tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-120b")
model = OPTForCausalLM.from_pretrained("facebook/galactica-120b", device_map="auto", load_in_8bit=True, cache_dir="/home/ec2-user/checkpoints")
input_text = "The Transformer architecture [START_REF]"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

pdb.set_trace()
pass