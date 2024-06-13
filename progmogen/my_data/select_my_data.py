import os,sys
import numpy as np 


def select_list(old_file, new_file, s):
    with open(old_file, "r") as f:
        data = f.readlines()
    
    data_new = [i for i in data if s in i]
    with open(new_file, "w") as f:
        f.writelines(data_new)

def select_list_id_only(old_file, new_file, s):
    with open(old_file, "r") as f:
        data = f.readlines()
    
    data_new = [i for i in data if s in i]
    data_new = [i.split()[0]+"\n" for i in data_new]
    with open(new_file, "w") as f:
        f.writelines(data_new)


def main():
    old_file_name = "test_text_my.txt"
    new_file_name = "test_pick_v0.txt"
    select_list(old_file_name, new_file_name, "pick")

    new_file_name_id_only = "test_pick_v0_id.txt"
    select_list_id_only(old_file_name, new_file_name_id_only, "pick")



if __name__ == "__main__":
    main()