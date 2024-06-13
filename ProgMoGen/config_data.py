import os 

# for mdm pretrained weight. Be sure to have args.json under the same directory.
MODEL_PATH="./save/humanml_trans_enc_512/model000475000.pt"


# Mean.npy/Std.npy, t2m_mean.npy/t2m_std.npy in ddim_*.py and DataTranform in run_demo_*.py 
# t2m checkpoint in progmogen/data_loaders/humanml/networks/evaluator_wrapper.py
# $ROOT_DIR/dataset/HumanML3D in dataloaders/humanml/data/dataset.py
# place HumanML3D dataset under $ROOT_DIR/dataset
ROOT_DIR="."


# for glove.
# Motion_all_dataset in run_demo_*.py, eval_task_*.py 
ABS_BASE_PATH=ROOT_DIR


# for smpl model
# visualize/joints2smpl/src/config.py
SMPL_MODEL_DIR_0 = os.path.join(ROOT_DIR, "body_models/")
# utils/config.py
SMPL_DATA_PATH_0 = os.path.join(SMPL_MODEL_DIR_0, "smpl")


# for IK skeleton template
# in eval_task_*.py DataTransform, ddim_*.py
SKEL_JOINTS_TEMPLATE_FILE_NAME = os.path.join(os.path.dirname(__file__), "my_data", "skeleton_template/gen.npy") 



# for loading gt samples in evaluation.
EVAL_SAMPLE32_FILE_NAME= os.path.join(os.path.dirname(__file__), "my_data", "n32_data.npy") 
EVAL_HOI1_FILE_NAME    = os.path.join(os.path.dirname(__file__), "my_data", "hoi1_eval_data.npy") 
EVAL_HSI1_FILE_NAME    = os.path.join(os.path.dirname(__file__), "my_data", "hsi1_n544_eval_data.npy") 


