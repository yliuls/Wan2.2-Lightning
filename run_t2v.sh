CUDA_VISIBLE_DEVICES=0,5 torchrun --nproc_per_node=2 generate.py --task t2v-A14B --size "1280*720" \
 --ckpt_dir ./Wan2.2-T2V-A14B --lora_dir ./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-250928 \
 --dit_fsdp --t5_fsdp --ulysses_size 2 \
 --base_seed 42 --prompt_file examples/prompt_list.txt \
#  --lora_dir2 /data/yaofu/lora_wan\

# 测试删除最后的时间后缀
# for file in *.mp4; do
#   # 匹配规则：_????????_??????.mp4（? 代表任意单个字符，8个?=日期，6个?=时间）
#   new_file="${file%_????????_??????.mp4}.mp4"
#   echo "原文件：$file → 新文件：$new_file"
# done

#实际执行删除时间后缀
# for file in *.mp4; do
#   new_file="${file%_????????_??????.mp4}.mp4"
#   mv "$file" "$new_file"
# done