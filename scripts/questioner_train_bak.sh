#!/bin/bash
set -x

solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

echo "Starting questioner training with solver: $solver_model_path, questioner: $questioner_model_path, save to: $experiment_name"

bash scripts/start_solver_services.sh $solver_model_path &
SOLVER_PID=$!

trap "echo 'Shutting down solver services...'; kill $SOLVER_PID; sleep 2; echo 'Cleanup completed.'" EXIT

echo "Solver services ready. Starting questioner training..."

export SOLVER_MODEL_NAME=$experiment_name

CUDA_VISIBLE_DEVICES=1,2,3 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/geo/data_questioner_train/train.parquet \
    data.val_files=data/geo/data_questioner_train/test.parquet \
    data.train_batch_size=48 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$questioner_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=reward_fuction/caller.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["swanlab"]' \
    trainer.project_name='geo' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1

echo "Training completed. Merging model..."

PROJECT_NAME='geo'
CHECKPOINT_DIR="checkpoints/${PROJECT_NAME}/${experiment_name}"

LATEST_STEP=$(ls -1 $CHECKPOINT_DIR | grep "global_step_" | sort -V | tail -1)

if [ -n "$LATEST_STEP" ]; then
    echo "Found latest checkpoint: $LATEST_STEP"
    
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${CHECKPOINT_DIR}/${LATEST_STEP}/actor \
        --target_dir ${CHECKPOINT_DIR}/${LATEST_STEP}/actor/huggingface
    
    mkdir -p ../models/${experiment_name}
    cp -rL ${CHECKPOINT_DIR}/${LATEST_STEP}/actor/huggingface/* ../models/${experiment_name}
    echo "Model merged successfully: ${CHECKPOINT_DIR}/${LATEST_STEP}/actor/huggingface"
else
    echo "Warning: No checkpoint found in $CHECKPOINT_DIR"
fi

echo "Shutting down solver services..."
pkill -f "start_vllm_server.py"
pkill -f "VLLM::EngineCor"
sleep 2
echo "Cleanup completed."