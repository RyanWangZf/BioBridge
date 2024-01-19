# # encode all the disease nodes
CUDA_VISIBLE_DEVICES=1  python -u encode_disease.py \
    --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --batch_size 16 \
    --data_path "./data/Processed" \
    --output_path "./data/embeddings"


# # encode all the biological process nodes
CUDA_VISIBLE_DEVICES=1 python -u encode_bp.py \
    --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --batch_size 16 \
    --data_path "./data/Processed" \
    --output_path "./data/embeddings"


# # encode all the mollecular function nodes
CUDA_VISIBLE_DEVICES=1 python -u encode_mf.py \
    --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --batch_size 16 \
    --data_path "./data/Processed" \
    --output_path "./data/embeddings"


# # encode all the cellular componenet nodes
CUDA_VISIBLE_DEVICES=1  python -u encode_cc.py \
    --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --batch_size 16 \
    --data_path "./data/Processed" \
    --output_path "./data/embeddings"


# merge disease embeddings
python -u merge_disease_embedding.py