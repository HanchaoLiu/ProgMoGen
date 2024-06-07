

###############################################################################
# define task
###############################################################################

eval_method="ours"
ret_type="pos"
text_split="test_plane_v0_id"
num_samples_limit=32
save_tag="limited_space"

save_fig_dir="result/demo/${save_tag}_n${num_samples_limit}/${eval_method}_${ret_type}_npy"

task_config="task_hsi3_config"


###############################################################################
# generate motion
###############################################################################
python3 tasks/run_demo_simple.py \
    --use_ddim_tag 1 \
    --mask_type 'root_horizontal' \
    --eval_mode "debug" \
    --save_tag "${save_tag}" \
    --ret_type "${ret_type}" \
    --save_fig_dir "${save_fig_dir}" \
    --text_split "${text_split}" \
    --num_samples_limit ${num_samples_limit} \
    --task_config ${task_config}

idx=3


###############################################################################
# generate stick figure animation
###############################################################################
python3 -m visualize.plot_motion \
    --input_path "${save_fig_dir}/gen.npy" \
    --output_path "${save_fig_dir}/gen${idx}_video.gif" \
    --selected_idx ${idx}


###############################################################################
# generate mesh
###############################################################################
python3 -m visualize.render_mesh_each --input_path "${save_fig_dir}/gen.npy" --selected_idx ${idx}

# exit 

###############################################################################
# render image/video
###############################################################################

# absolute path to this project
project_dir="/path/to/ProgMoGen"
# absolute path to blender app
blender_app="/path/to/blender-2.93.0/blender"

cd ../TEMOS-master
mesh_file="${project_dir}/ProgMoGen/${save_fig_dir}/gen_smpl/gen${idx}_smpl_params.npy"
echo ${mesh_file}

${blender_app} --background --python render_demo_hsi3.py -- npy=${mesh_file} canonicalize=true mode="sequence"
${blender_app} --background --python render_demo_hsi3.py -- npy=${mesh_file} canonicalize=true mode="video"
echo "[Results are saved in ${save_fig_dir}/gen_smpl]"
cd -

