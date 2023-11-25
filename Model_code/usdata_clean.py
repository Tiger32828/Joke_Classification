import os


def main():
    path = "./Datasets/US"
    out_p = "./Datasets/US_clean"
    remove_set = {'Q:', 'A:'}
    for file in os.listdir(path):
        file_path = f"{path}/{file}"
        file_out = f"{out_p}/{file}"
        with open(file_path, 'r') as f:
            fout = open(file_out,'w')
            for line in f:
                while line[:2] in remove_set:
                    line = line[3:]
                fout.write(line)



if __name__ == '__main__':
    main()
