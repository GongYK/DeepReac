import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import random
from sklearn.svm import SVR as svr
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

def arg_parse():
    parser = argparse.ArgumentParser(description="reactGAT arguments.")
    parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    parser.add_argument("--outdir", dest="outdir", help="result directory")
    parser.add_argument("--suffix", dest="name_suffix", help="suffix added to the output filename")
    parser.add_argument('--pre', dest='pre_ratio', type=float, help='Ratio of dataset for pre-training.')
    parser.add_argument("--select_mode", dest="select_mode", help="method to select data instances")
    parser.add_argument("--select_num", dest="num_selected", type=int, help="Number of data instances to select.")
    parser.add_argument("--sim_num", dest="simulation_num", type=int, help="Number of rounds for simulation.")
    parser.add_argument("--model", dest="model", help="rf or svm")
    parser.add_argument("--mode", dest="mode", help="descriptor or onehot")

    parser.set_defaults(
        dataset="DatasetA",
        outdir="results",
        name_suffix="1",
        pre_ratio=0.1,
        select_mode = "random",
        num_selected = 10,
        simulation_num = 10,
        model = "rf",
        mode = "descriptor"
    )
    return parser.parse_args()

def load_data(data):
    index = []
    feats = []
    labels = []
    for data_ in data:
        index_, feat, label = data_
        index.append(index_)
        feats.append(feat)
        labels.append(label)
    feats = np.array(feats)
    labels = np.array(labels)
    return index, feats, labels

def get_onehot(typelist,name):
    onehot = [0]*len(typelist)
    onehot[typelist.index(name)] = 1
    return onehot

def Rank(feats, index, predictions=None, feats_labeled=None, label=None, select_mode="random", num_selected=10):
    
    if select_mode == "random":
        random.shuffle(index)
        update_list = index[:num_selected]

    elif select_mode == "diversity":
        cos = cosine_similarity(feats,feats_labeled)
        similarity_list = np.max(cos,axis=-1)
        df = pd.DataFrame(zip(index,similarity_list),columns=['index','similarity'])
        df_sorted = df.sort_values(by=['similarity'],ascending=True)
        df_index = df_sorted['index'].values
        update_list = list(df_index[:num_selected])

    elif select_mode == "adversary":
        cos = cosine_similarity(feats,feats_labeled)
        label_list = []
        for i in range(len(feats)):
            idx_max = np.argmax(cos[i])
            label_list.append(label[idx_max])
            
        diff = abs(predictions - np.array(label_list))
        df = pd.DataFrame(zip(index,diff),columns=['index','diff'])
        df_sorted = df.sort_values(by=['diff'],ascending=False)
        df_index = df_sorted['index'].values
        update_list = list(df_index[:num_selected])

    return update_list

def load_dataset(dataset, mode = "descriptor"):
    if dataset == "DatasetA":
        raw_data = pd.read_csv('Data/DatasetA/Scaled_dataset.csv')
        y = (raw_data.values[:,-1])*0.01
        nan_list = [696, 741, 796, 797, 884]
        index_list = []
        for i in range(3960):
            if i not in nan_list:
                index_list.append(i)

        if mode == "descriptor":
            X = raw_data.values[:,:-1]
            data = []
            for i in index_list:
                data_ = (str(i),X[i],y[i])
                data.append(data_)

        elif mode == "onehot":
            from Data.DatasetA import main_test

            plate1 = main_test.plate1
            plate2 = main_test.plate2
            plate3 = main_test.plate3
            plates = [plate1,plate2,plate3]
            c1 = []
            c2 = []
            c3 = []
            c4 = []
            for plate in plates:
                for r in range(plate.rows):
                    for c in range(plate.cols):
                        cond = plate.layout[r][c].conditions
                        c1.append(cond['additive'])
                        c2.append(cond['ligand'])
                        c3.append(cond['aryl_halide'])
                        c4.append(cond['base'])
            components = [c1,c2,c3,c4]
            types = [list(set(names)) for names in components]
            data = []
            for i in index_list:
                onehot = []
                for j in range(len(components)):
                    onehot+=get_onehot(types[j],components[j][i])
                data_ = (str(i),onehot,y[i])
                data.append(data_)

    elif dataset == "DatasetC":
        raw_data = pd.read_csv("Data/DatasetC/dataset_D.csv")
        X = raw_data.values[:,0]
        y = (raw_data.values[:,-1])*0.01
        y = y.astype(float)
        label = np.log((1+y)/(1-y))*0.001987*298
        if mode == "descriptor":
            feat1 = pd.read_csv("Data/DatasetC/combined_ASO_bpas.csv",header=None)
            feat2 = pd.read_csv("Data/DatasetC/combined_ASO_products.csv",header=None)
            data = []
            for i in range(len(y)):
                name = X[i].split('_')
                cata = name[0]+'_'+name[1]
                react = name[2]+'_'+name[3]
                reaction = np.array(list(feat1[feat1[0] == cata].values[0,1:])+list(feat2[feat2[0] == react].values[0,1:-1]))
                data_ = (str(i),reaction,label[i])
                data.append(data_)
                
        elif mode == "onehot":
            components = np.array([x.split('_') for x in X]).T
            types = [list(set(names)) for names in components]
            data = []
            for i in range(len(y)):
                onehot = []
                for j in range(len(components)):
                    onehot+=get_onehot(types[j],components[j][i])
                data_ = (str(i),onehot,label[i])
                data.append(data_)

    elif dataset == "DatasetB":
        raw_data = pd.read_excel("Data/DatasetB/aap9112_Data_File_S1.xlsx")
        react1 = raw_data['Reactant_1_Short_Hand'].values
        react2 = raw_data['Reactant_2_Name'].values
        ligand = raw_data['Ligand_Short_Hand'].values
        reagent = raw_data['Reagent_1_Short_Hand'].values
        solvent = raw_data['Solvent_1_Short_Hand'].values
        y = raw_data['Product_Yield_PCT_Area_UV'].values

        components = [react1,react2,ligand,reagent,solvent]
        types = [list(set(names)) for names in components]

        data = []
        for i in range(len(y)):
            name2 = react2[i].split(',')
            if name2[0] == "2d":
                continue
            onehot = []
            for j in range(len(components)):
                onehot+=get_onehot(types[j],components[j][i])
            onehot = torch.tensor(onehot)
            label = torch.tensor([y[i]*0.01])
            data_ = (str(i),onehot,label)
            data.append(data_)

    return data

def main():

    args = arg_parse()
    data = load_dataset(args.dataset, args.mode)

    rmses = []
    maes = []
    r2s = []
    steps = []
    for num in range(args.simulation_num):
        t_start = time.time()
        random.shuffle(data)
        labeled = data[:int(args.pre_ratio*len(data))]
        unlabeled = data[int(args.pre_ratio*len(data)):]

        first_round = True
        rmse = []
        mae = []
        r2 = []
        step=[]
        if args.dataset == "DatasetA":
            clf = rf()
        elif args.dataset == "DatasetC":
            clf = svr(kernel='poly', degree=2)
        while len(unlabeled) > args.num_selected:
            if not first_round:
                sample_ = []
                sample_list = []
                for i,sample in enumerate(unlabeled):
                    if sample[0] in update_list:
                        sample_.append(i)
                sample_.sort(reverse=True)
                for i in sample_:
                    sample_list.append(unlabeled.pop(i))
                labeled += sample_list
                
            index, feats, labels = load_data(labeled)
            index_un, feats_un, labels_un = load_data(unlabeled)
            clf.fit(feats,labels)
            predictions = clf.predict(feats_un)
            score_rmse = np.sqrt(mean_squared_error(predictions,labels_un))
            score_mae = mean_absolute_error(predictions,labels_un)
            score_r2 = r2_score(predictions,labels_un)
            label_ratio = len(labeled)/len(data)
            rmse.append(score_rmse)
            mae.append(score_mae)
            r2.append(score_r2)
            step.append(label_ratio)
            print("Round",num+1,", label_ratio",label_ratio)
            update_list = Rank(feats_un,index_un,predictions,feats,labels,args.select_mode,args.num_selected)
            first_round = False
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        steps.append(step)
        print("The",num+1,"round", (time.time()-t_start)/3600, "小时")

    result = {"rmses":rmses, "maes":maes, "r2s":r2s, "steps":steps}
    result = json.dumps(result)
    filename = args.dataset+"_ML_"+args.select_mode+"_rounds"+str(args.simulation_num)+"_"+args.name_suffix+".json"
    with open(os.path.join(args.outdir,filename),'w') as f:
        f.write(result)


if __name__ == "__main__":
    main()