# test 8M ESM2 to encode proteins
# nohup python -u encode_protein.py \
#     --model_name "facebook/esm2_t6_8M_UR50D" \
#     --batch_size 16 \
#     --data_path "./data/Processed" \
#     --output_path "./data/embeddings" \
#     > encode.log &

# nohup python -u encode_protein.py \
#     --model_name "facebook/esm2_t33_650M_UR50D" \
#     --batch_size 16 \
#     --data_path "./data/Processed" \
#     --output_path "./data/embeddings" \
#     > encode.log &

# encode protein for primekg dataset
CUDA_VISBLE_DEVICES=0 nohup python -u encode_protein.py \
    --model_name "facebook/esm2_t36_3B_UR50D" \
    --batch_size 16 \
    --data_path "./data/Processed" \
    --output_path "./data/embeddings" \
    > encode.log &
