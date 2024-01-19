# encode SMILES string
# CUDA_VISBLE_DEVICES=2 nohup python -u encode_drug.py \
#     --model_name "laituan245/molt5-large-smiles2caption" \
#     --batch_size 16 \
#     --data_path "./data/Processed" \
#     --output_path "./data/embeddings" \
#     > encode.log &

# CUDA_VISBLE_DEVICES=2 python -u encode_drug.py \
#     --model_name "laituan245/molt5-large-smiles2caption" \
#     --batch_size 16 \
#     --data_path "./data/Processed" \
#     --output_path "./data/embeddings" 
# encode captions
# TODO