
# ############# MGI - MPO #############

# # encode all the mouse phenotype nodes
# CUDA_VISIBLE_DEVICES=0  python -u encode_mouse_phenotype.py \
#     --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
#     --batch_size 16 \
#     --data_path "./data/mouse_protein/processed" \
#     --output_path "./data/mouse_protein"

# encode all the mouse protein nodes
# CUDA_VISBLE_DEVICES=0 nohup python -u encode_mouse_protein.py \
#     --model_name "facebook/esm2_t36_3B_UR50D" \
#     --batch_size 8 \
#     --data_path "./data/mouse_protein/processed" \
#     --output_path "./data/mouse_protein" \
#     > encode.log &

# ############# MGI - HPO #############
# encode all the human phenotype nodes
# CUDA_VISIBLE_DEVICES=0  python -u encode_human_phenotype.py \
#     --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
#     --batch_size 16 \
#     --data_path "./data/mouse_protein/processed" \
#     --output_path "./data/mouse_protein/embeddings"


# encode all the mouse protein nodes
# CUDA_VISBLE_DEVICES=0 nohup python -u encode_mouse_protein.py \
#     --model_name "facebook/esm2_t36_3B_UR50D" \
#     --batch_size 8 \
#     --data_path "./data/mouse_protein/mgi_hpo_task" \
#     --output_path "./data/mouse_protein" \
#     > encode.log &

