from scipy.spatial import distance

#  algorithme knn
def who_is_it(cible, points, k, threshold):
    table_coor = []
    seuil = []
    i = 0
    while i < len(points):
        current = points[i]
        table_coor.append(current['Coordinates'])
        i += 1

    def k_plus_proches_voisins(table, cible, k):

        def distance_cible(donnee):
            dis = distance.euclidean(donnee, cible)
            if dis <= threshold:
                seuil.append((donnee, dis))
            return dis

        table_triee = sorted(table, key=distance_cible)

        proches_voisins = []

        for i in range(k):

            proches_voisins.append(table_triee[i])
            #if i == (len(seuil) - 1):
            #    break

        return proches_voisins

    return k_plus_proches_voisins(table_coor, cible, k), seuil

# ordonner les utilisateurs
def ord_users(nv, points):
    users_list = []
    for j in nv:
        for i in points:
            if i['Coordinates'] == j:
                users_list.append(i['Username'])

    return users_list
