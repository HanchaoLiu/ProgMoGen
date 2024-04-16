


remote_dir="/home/cscg/liuhc/motion_result_3"
mkdir -p ${remote_dir}

MODEL_PATH="/home/cscg/liuhc/code4/mdm_data/save/mdm_raw/model000475000.pt"



eval_method="ours"
ret_type="pos"
text_split="test_plane_v0_id"
num_samples_limit=32
save_tag="hold_ball_pbg"

save_fig_dir="${remote_dir}/demo/${save_tag}_n${num_samples_limit}/${eval_method}_${ret_type}_npy"

task_config="task_pbg2_config"


# generate motion
python3 tasks/run_demo.py \
    --model_path ${MODEL_PATH} \
    --use_ddim_tag 1 \
    --mask_type 'root_horizontal' \
    --eval_mode "debug" \
    --save_tag "${save_tag}" \
    --ret_type "${ret_type}" \
    --save_fig_dir "${save_fig_dir}" \
    --text_split "${text_split}" \
    --num_samples_limit ${num_samples_limit} \
    --task_config ${task_config}

# exit


# generate mesh
idx=0
python3 -m visualize.render_mesh_each_fixhand --input_path "${save_fig_dir}/gen.npy" --selected_idx ${idx}

# render image/video
cd ../TEMOS-master
input_joint_file="${save_fig_dir}/gen.npy"
mesh_file="${save_fig_dir}/gen_smpl/gen${idx}_smpl_params_fixhand.npy"
echo ${mesh_file}
blender_app="/home/cscg/Downloads/blender-2.93.0/blender"
${blender_app} --background --python render_demo_pbg2.py -- npy=${mesh_file} +npy_joint=${input_joint_file} +npy_joint_idx=${idx} canonicalize=true mode="sequence"
${blender_app} --background --python render_demo_pbg2.py -- npy=${mesh_file} +npy_joint=${input_joint_file} +npy_joint_idx=${idx} canonicalize=true mode="video"
echo "[Results are saved in ${save_fig_dir}/gen_smpl]"
cd -



