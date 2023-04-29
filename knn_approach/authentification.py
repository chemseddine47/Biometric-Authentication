from database_manager import extract_from_dataset
from knn_engine import who_is_it
from processing_utils import splitting, car_extraction, sequence_video, row_to_list, print_inf

temps = 15
# seuil
threshold = 5
nb_user = 13
print('threshold', threshold)
print('/////////////////////////////////////////////////////////////////')


k = 1
while k < 7:
    print('k =', k)
    total_attempts = 0
    false_acc_r = 0
    false_acc_l = 0

    i = 1
    while i <= nb_user:
        
        total_attempts += 1

        path = 'caps/user' + str(i) + '_capcible.mp4'
        print('user:', path)

        user, cap = splitting(path)
        x, y, z = sequence_video(path, temps)

        cible, cible2 = car_extraction(y, z, user, x)

        table, table2 = extract_from_dataset()

        liste = row_to_list(table)
        liste2 = row_to_list(table2)


        
        #print('right eye:')
        values, seuil = who_is_it(cible, liste, k, threshold)

        pred_user_r = print_inf(seuil, values, liste)
        if pred_user_r != user:
            false_acc_r+=1
       
        #print('*********************************************************************')
        #print('left eye:')
        values2, seuil2 = who_is_it(cible2, liste2, k, threshold)

        pred_user_l = print_inf(seuil2, values2, liste2)
        if pred_user_l != user:
            false_acc_l+=1

       # print('#######################################################################')
        
        i += 1
    
    FAR_r = (false_acc_r/total_attempts)*100
    FAR_l = (false_acc_l/total_attempts)*100
    print("for k = ",k," right FAR = ",FAR_r)
    print("for k = ",k," left FAR = ",FAR_l)
    
    k += 1