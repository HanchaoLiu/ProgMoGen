

remote_dir="/home/cscg/liuhc/motion_result_3"
mkdir -p ${remote_dir}

MODEL_PATH="/home/cscg/liuhc/code4/mdm_data/save/mdm_raw/model000475000.pt"



eval_method="mdmedit"
ret_type="rot"
text_split="test_all_id"
num_samples_limit=512
save_tag="headheight_gt"


save_fig_dir="${remote_dir}/eval/${save_tag}_n${num_samples_limit}/${eval_method}_${ret_type}_npy"

task_config="eval_task_hsi1_mdmedit_config"


# generate motion
python3 tasks/eval_task_hsi1.py \
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
    --diffusion_type "ddim_inpaint"



# eval result 
python3 eval/main_eval_hsi1.py --input_path "${save_fig_dir}/gen.npy"


# sample gt motion multiple times
# sh script_eval/eval_task_hsi1_fid_nruns.sh "${save_fig_dir}/gen.npy"