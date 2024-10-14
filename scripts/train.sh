export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"
export DATASET_NAME="datasets/"
export DATASET_META_NAME="datasets/calvin/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=256 \
  --video_sample_size=256 \
  --token_sample_size=256 \
  --video_sample_stride=2 \
  --video_sample_n_frames=49 \
  --train_batch_size=4 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=1000 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --use_came \
  --use_deepspeed \
  --train_mode="inpaint" \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --resume_from_checkpoint "latest" \
  --report_to "wandb" \
  --tracker_project_name "cog" \