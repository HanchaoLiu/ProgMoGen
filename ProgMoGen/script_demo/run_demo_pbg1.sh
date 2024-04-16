


remote_dir="/home/cscg/liuhc/motion_result_3"
mkdir -p ${remote_dir}

MODEL_PATH="/home/cscg/liuhc/code4/mdm_data/save/mdm_raw/model000475000.pt"



eval_method="ours"
ret_type="pos"
text_split="test_plane_v0_id"
num_samples_limit=32
save_tag="stand_balanced"

save_fig_dir="${remote_dir}/demo/${save_tag}_n${num_samples_limit}/${eval_method}_${ret_type}_npy"

task_config="task_pbg1_config"


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
idx=1
python3 -m visualize.render_mesh_each --input_path "${save_fig_dir}/gen.npy" --selected_idx ${idx}

# render image/video
cd ../TEMOS-master
input_joint_file="${save_fig_dir}/gen.npy"
mesh_file="${save_fig_dir}/gen_smpl/gen${idx}_smpl_params.npy"
echo ${mesh_file}
blender_app="/home/cscg/Downloads/blender-2.93.0/blender"
${blender_app} --background --python render_demo_pbg1.py -- npy=${mesh_file}  canonicalize=true mode="sequence"
${blender_app} --background --python render_demo_pbg1.py -- npy=${mesh_file}  canonicalize=true mode="video"
echo "[Results are saved in ${save_fig_dir}/gen_smpl]"
cd -


# TODO: need to change the frame number for video. for sequence demonstration we only show 3 frames.
# TEMOS-master/temos/render/blender/render_base_rawheight_massctr_trans.py
# line 181: frameidx = [44,124,184] --> for mode="sequence", comment this line for mode="video"

