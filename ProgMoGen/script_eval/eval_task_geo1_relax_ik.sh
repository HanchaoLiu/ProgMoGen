


remote_dir="/home/cscg/liuhc/motion_result_3"
mkdir -p ${remote_dir}

MODEL_PATH="/home/cscg/liuhc/code4/mdm_data/save/mdm_raw/model000475000.pt"



eval_method="ik"
ret_type="pos"
text_split="test_plane_v0_id"
num_samples_limit=32
save_tag="plane_relax"


# data/eval, data/demo
save_fig_dir="${remote_dir}/eval/${save_tag}_n${num_samples_limit}/${eval_method}_${ret_type}_npy"

task_config="eval_task_geo1_relax_ik_config"


# generate motion
python3 tasks/eval_task_goal_relaxed_baseline.py \
    --model_path ${MODEL_PATH} \
    --use_ddim_tag 1 \
    --mask_type 'root_horizontal' \
    --eval_mode "debug" \
    --save_tag "${save_tag}" \
    --ret_type "${ret_type}" \
    --save_fig_dir "${save_fig_dir}" \
    --text_split "${text_split}" \
    --num_samples_limit ${num_samples_limit} \
    --task_config ${task_config} \
    --eval_task "geo1" \
    --diffusion_type "ddim"


# eval result 
python3 eval/main_eval_geo1_relax.py --input_path "${save_fig_dir}/gen.npy"