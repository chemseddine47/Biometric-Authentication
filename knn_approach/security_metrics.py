
import warnings
import pandas as pd
from database_manager import extract_from_dataset
from knn_engine import who_is_it
from processing_utils import row_to_list, print_inf

warnings.filterwarnings("ignore")

table, table2 = extract_from_dataset()

liste = row_to_list(table)
liste2 = row_to_list(table)


# d =4

def frr(dataset,listn):
    dtestfrr = pd.read_csv(dataset)

    dtestfrr= dtestfrr.drop(["Username", "Cap","Relatif minimum"], axis=1)
    totalfrr = 15
    frr = 0
    k=1
    while k <= 7:
        i=1
        frr = 0
        for index, row in dtestfrr.iterrows():
            values, seuil = who_is_it(row, listn, k,d)

            pred_user_r = print_inf(seuil, values, listn)

            pred_user_r = pred_user_r.replace("user", "")

            if pred_user_r == "null":
                frr +=1
            
            #print(i ,"is ",pred_user_r)  
            i += 1
        print("for k = ",k,"d = ",d,"frr = ","{:.2f}".format(frr/totalfrr))
        k+=1


def far(dataset,listen):
    
    dtest = pd.read_csv(dataset)

    dtest= dtest.drop(["Username", "Cap","Relatif minimum"], axis=1)
    total = 7
    far = 0
    k = 1

    while k <= 7:
        # i=16
        far = 0
        for index, row in dtest.iterrows():
            values, seuil = who_is_it(row, listen, k, d)

            pred_user_r = print_inf(seuil, values, listen)

            if pred_user_r != "null":
                far += 1

            # print(i ,"is ",pred_user_r)  
            # i += 1
        print("for k =", k, "d =", d, "far =","{:.2f}".format( far / total))
        k += 1



ds = [2.5,2.6,2.7,2.8,2.9,3,3.1]

for d in ds:
    
    print("right resulats")
    frr("testdt.csv",liste)
    far("far.csv",liste)

    print("left resulats")
    frr("testdt2.csv",liste2)
    far("far2.csv",liste2)
    print("*************************")


