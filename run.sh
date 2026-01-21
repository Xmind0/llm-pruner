(

python3 simple_slice.py --orientation fisher --model daryl149/llama-2-7b-chat-hf --cal-dataset wikitext2 --sparsity 0.2; \
python3 simple_slice.py --orientation pca --model daryl149/llama-2-7b-chat-hf --cal-dataset wikitext2 --sparsity 0.2; \
python3 simple_slice.py --orientation fisher --model daryl149/llama-2-7b-chat-hf --cal-dataset wikitext2 --sparsity 0.5; \
python3 simple_slice.py --orientation pca --model daryl149/llama-2-7b-chat-hf --cal-dataset wikitext2 --sparsity 0.5 
 
) \
2>&1 | tee output.log

# daryl149/llama-2-7b-chat-hf   JackFram/llama-68m   facebook/opt-125m

# python3 simple_slice.py --orientation pca --model daryl149/llama-2-7b-chat-hf --cal-dataset wikitext2 --sparsity 0.2