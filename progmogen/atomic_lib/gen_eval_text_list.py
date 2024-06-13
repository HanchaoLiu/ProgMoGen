import os,sys 
import numpy as np 


def process_token(tokens):

    max_text_len=20
    # caption, tokens = text_data['caption'], text_data['tokens']

    if len(tokens) < max_text_len:
        # pad with "unk"
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
        tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
    else:
        # crop
        tokens = tokens[:max_text_len]
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)

    result = '_'.join(tokens)
    return result 


def load_txt(file_name):
    with open(file_name, "r") as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    return data 



def load_reference_gen_file(ref_gen_file):
    saved_data=np.load(ref_gen_file, allow_pickle=True).item()
    # print(saved_data.keys())
    # saved_prompt_list = list(zip( saved_data['text'], saved_data['lengths'] ))
    return saved_data['text'], saved_data['lengths']


def get_text_to_tokens_mapping(text_id_list_file, text_dir):

    text_id_list = load_txt(text_id_list_file)
    # text_id_list = [i for i in text_id_list if (not i.startswith("M"))]
    print(text_id_list)

    all_text_prompt_list = []
    for text_id in text_id_list:
        text_file = os.path.join( text_dir, f"{text_id}.txt")
        print(text_file)
        assert os.path.exists(text_file)

        text_prompt_list = load_txt(text_file)
        # print(text_prompt_list)
        all_text_prompt_list += text_prompt_list

    print("all_text_prompt_list = ", len(all_text_prompt_list))

    caption_list = []
    tokens_list = []
    dt = {}
    for line in all_text_prompt_list:
        line_split = line.split("#")
        caption = line_split[0]
        tokens  = line_split[1].split(' ')
        tokens  = process_token(tokens)
        print(caption, tokens)

        # assert caption not in dt.keys()
        dt[caption] = tokens

        caption_list.append(caption)
        tokens_list.append(tokens)

    # print(caption_list)
    # assert len(caption_list)==len(set(caption_list))
    return dt 


def save_npy(file_name, dt):
    np.save(file_name, dt)


def main_table2():

    ref_gen_file = "limited_ours_pos_npy/gen.npy"

    text_id_list_file = "dataset/HumanML3D/test_plane_v0_id.txt"
    text_dir = "dataset/HumanML3D/texts"

    
    text_to_tokens_dt = get_text_to_tokens_mapping(text_id_list_file, text_dir)
    

    caption_list, length_list = load_reference_gen_file(ref_gen_file)

    print(caption_list)
    print(length_list)

    tokens_list = [text_to_tokens_dt[i] for i in caption_list]

    ref_gen_data = list(zip(caption_list, tokens_list, length_list))
    for line in ref_gen_data:
        print(line)


    save_name = "ref_data/n32_data.npy"
    save_npy(save_name, ref_gen_data)

    x = np.load(save_name, allow_pickle=True)
    print("->")
    print(x[0])


def main_table1():

    ref_gen_file = "head_ours_gt_pos_npy/gen.npy"

    text_id_list_file = "dataset/HumanML3D/test_all_id.txt"
    text_dir = "dataset/HumanML3D/texts"

    
    text_to_tokens_dt = get_text_to_tokens_mapping(text_id_list_file, text_dir)
    

    caption_list, length_list = load_reference_gen_file(ref_gen_file)

    print(caption_list)
    print(length_list)

    tokens_list = [text_to_tokens_dt[i] for i in caption_list]

    ref_gen_data = list(zip(caption_list, tokens_list, length_list))
    for line in ref_gen_data:
        print(line)


    save_name = "ref_data/n544_data.npy"
    save_npy(save_name, ref_gen_data)

    x = np.load(save_name, allow_pickle=True)
    print("->")
    print(x[0])





def check_ref_file():

    EVAL_SAMPLE32_FILE_NAME = "ref_data/n32_data.npy"
    ref_n32_data = np.load(EVAL_SAMPLE32_FILE_NAME, allow_pickle=True)
            
    ref_text_prompt_list = [each_sample[0] for each_sample in ref_n32_data]
    ref_tokens_list      = [each_sample[1] for each_sample in ref_n32_data]
    ref_length_list      = [int(each_sample[2]) for each_sample in ref_n32_data]

    print(ref_n32_data[0])


# 0404: length_list for hoi1 is not fixed. fix this bug and regenerate ref_file
def save_ref_file_hoi1():

    EVAL_HOI1_FILE_NAME = "ref_data/hoi1_eval_data.npy"
    # ref_n32_data = np.load(EVAL_HOI1_FILE_NAME, allow_pickle=True)
            
    # ref_text_prompt_list = [each_sample[0] for each_sample in ref_n32_data]
    # ref_tokens_list      = [each_sample[1] for each_sample in ref_n32_data]
    # ref_length_list      = [int(each_sample[2]) for each_sample in ref_n32_data]

    # print(ref_n32_data[0])

    TEXT_PROMPT_LIST = [
    'a man picks something up, then puts it back.',
    'a person picks an object from a table and move it to another table.',
    'a person moves an object from a place to another place.',
    'a man picks something up puts it to another place while walking.',
    'a man picks something up, then walk and puts it back.',
    'a person picks an object from a table, and then walk and move it to another table.',
    'a person moves an object from a place to another place while walking.',
    'a person picks an object, and carry it to another place.'
    ]*4
    # LENGTH_LIST=[176]*32
    LENGTH_LIST=[196, 196, 120, 196, 196, 172, 144, 140, 196, 176, 88, 88, 84, 148, 196, 184, 196, 152, 140, 196, 184, 144, 196, 192, 120, 100, 196, 140, 176, 196, 132, 152]

    # model_kwargs['y']['tokens'] = [None]*32
    # (n,6)
    TARGET_LIST = [
        [[0, 0.5, 0.2], [1.0, 0.5, 0.2]],
        [[0, 0.5, 0.2], [2.0, 0.5, 0.2]],
        [[0, 0.5, 0.2], [3.0, 0.5, 0.2]],
        [[0, 0.5, 0.2], [5.0, 0.5, 0.2]],

        [[0, 0.2, 0.2], [1.0, 0.5, 0.2]],
        [[0, 0.5, 0.2], [2.0, 0.2, 0.2]],
        [[0, 0.5, 0.2], [3.0, 0.8, 0.2]],
        [[0, 0.8, 0.2], [4.0, 0.5, 0.2]],
    ]*4

    target=[]
    for (p1,p2) in TARGET_LIST:
        target.append(p1+p2)
    # (32,6)
    target=np.array(target).tolist()

    dt = []
    for i in range(32):
        each = [ TEXT_PROMPT_LIST[i], None, LENGTH_LIST[i], target[i] ]
        dt.append(each)


    print(dt)
    np.save(EVAL_HOI1_FILE_NAME, dt)





def check_load_hoi1():
    EVAL_HOI1_FILE_NAME = "ref_data/hoi1_eval_data.npy"
    
    ref_n32_data = np.load(EVAL_HOI1_FILE_NAME, allow_pickle=True)
            
    ref_text_prompt_list = [each_sample[0] for each_sample in ref_n32_data]
    ref_tokens_list      = [each_sample[1] for each_sample in ref_n32_data]
    ref_length_list      = [int(each_sample[2]) for each_sample in ref_n32_data]
    ref_target_list      = [each_sample[3] for each_sample in ref_n32_data]
    ref_target_list = np.array(ref_target_list)
    print(ref_target_list.shape, ref_target_list.dtype)
    print(ref_target_list)

    # target_list_ref = [ target_list_ref ]*32
    # target_list_ref = np.stack(target_list_ref, 0)
    # print(target_list_ref.shape)


def save_ref_file_hsi1():

    ref_data = np.load("ref_data/n544_data.npy", allow_pickle=True)
    ref_text_prompt_list = [each_sample[0] for each_sample in ref_data]
    ref_tokens_list      = [each_sample[1] for each_sample in ref_data]
    ref_length_list      = [int(each_sample[2]) for each_sample in ref_data]

    # (544,3)
    input_constraint_file = "head_raw_gt_pos_npy/gen.npy"
    ref_constraint = np.load(input_constraint_file, allow_pickle=True).item()['constraint']

    save_file_name = "ref_data/hsi1_n544_eval_data.npy"

    n_samples = len(ref_text_prompt_list)
    dt = []
    for i in range(n_samples):
        dt.append( [
            ref_text_prompt_list[i], ref_tokens_list[i], ref_length_list[i], ref_constraint[i].reshape(-1).tolist()
        ] )
    
    np.save(save_file_name, dt)
    print("save to ", save_file_name)





if __name__ == "__main__":
    # main_table2()

    main_table1()