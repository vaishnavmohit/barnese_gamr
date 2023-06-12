import pandas as pd

import sys

def main():
    
    # print command line arguments
    for args in sys.argv[1:]:
        print(args)
    #setting training parameters
    
    #Load our dataset
    data_name = sys.argv[1]
    
    
    data = {'model_name': [],
            'ckpts_path': [],
            'ckpts_filename': [],
            'data_path': [],
            'batch_size': [],
            'image_type':[],
            'nclasses':[],
            'optimizer':[],
            'n_gpus':[],
            'epochs':[],
	        'lr':[],
            'wd': [],
            'val_split': [],
	}
    df = pd.DataFrame(data)
    df.to_pickle(data_name+'.pkl')

if __name__ == "__main__":
    main()
