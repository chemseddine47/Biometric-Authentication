from dataset_functions import extract_from_dataset
from knn_functions import who_is_it
from processing_utils import splitting, car_extraction, sequence_video, row_to_list, print_inf

temps = 15
# seuil
threshold = 5

nb_user = 13

print('threshold', threshold)
print('/////////////////////////////////////////////////////////////////')

i = 1
while i <= nb_user:
    path = 'caps/user' + str(i) + '_capcible.mp4'
    print('user:', path)
    user, cap = splitting(path)
    x, y, z = sequence_video(path, temps)

    cible, cible2 = car_extraction(y, z, user, x)

    table, table2 = extract_from_dataset()

    liste = row_to_list(table)
    liste2 = row_to_list(table2)

    k = 1
    #while k < 10:
    print('k =', k)
    print('right eye:')
    values, seuil = who_is_it(cible, liste, k, threshold)

    print_inf(seuil, values, liste)

    print('*********************************************************************')
    print('left eye:')
    values2, seuil2 = who_is_it(cible2, liste2, k, threshold)

    print_inf(seuil2, values2, liste2)

    print('#######################################################################')
    #k += 1
    i += 1
