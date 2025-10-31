


# Make pairs for MAST3R
python ../mast3r/make_pairs.py \
    --dir data/gerrard-hall_subset/images/ \
    --output pairs.txt \
    --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
    --retrieval_model ../mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
    --scene_graph retrieval-30-5 \

# Run MAST3R mapping

python ../mast3r/kapture_mast3r_mapping.py --dir data/gerrard-hall_subset/images/ --pairsfile_path pairs.txt --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric -o reconstruction_output/
