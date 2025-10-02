CUDA_VISIBLE_DEVICES=0,5 torchrun --nproc_per_node=2 generate.py \
 --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B \
 --lora_dir ./Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1 \
 --dit_fsdp --t5_fsdp --ulysses_size 2 \
 --base_seed 42 \
 --prompt_file examples/i2v_prompt_list.txt --image_path_file examples/i2v_image_path_list.txt \
 --lora_dir2 /data/yaofu/lora_wan\


# #
# #实际执行删除时间后缀
# for file in *.mp4; do
#   new_file="${file%_????????_??????.mp4}.mp4"
#   mv "$file" "$new_file"
# done

