from processing_utils import max_min, fft_calcul, relatif_avr, splitting, to_dict, sequence_video
from database_manager import add_to_dataset

nb_user = 15

i = 14

while i <= nb_user:
        
    j=1
    temps = 20
    while j < 11:
        
        # path de la vidéo
        path = 'caps/user' + str(i) + '_cap' + str(j) + '.mp4'

        #path = "4.mp4"
        print(path)
        # limiteur de la durée
        

        user, cap = splitting(path)
        #user = 1

        # extraction du signal des deux yeux
        x, y, z = sequence_video(path, temps)

        # extraction des caractéristiques
        _, _, max_quotient, _, min_quotient, avr_quotient = max_min(user, y)
        _, _, max_quotient2, _, min_quotient2, avr_quotient2 = max_min(user, z)
        y_fft = fft_calcul(y, temps)
        z_fft = fft_calcul(z, temps)
        _, _, max_fft, _, _, _ = max_min(user, y_fft)
        _, _, max_fft2, _, _, _ = max_min(user, z_fft)
        relatif_min = relatif_avr(y, 0.95, 0)
        relatif_max = relatif_avr(y, 1.04, 1)
        relatif_min2 = relatif_avr(z, 0.95, 0)
        relatif_max2 = relatif_avr(z, 1.04, 1)

        # plotter ses signaux
        #dessiner_graphe('Utilisateur 5: oeil droit et oeil gauche', 'oeil droit', x, y, 'oeil gauche', x, z)

        # enrollement des utisateurs (deux yeux, chaque 1 oeil = 1 dataset)
        row = to_dict(user, cap, min_quotient, max_quotient, avr_quotient, max_fft, relatif_min, relatif_max)
        row2 = to_dict(user, cap, min_quotient2, max_quotient2, avr_quotient2, max_fft2, relatif_min2, relatif_max2)

        add_to_dataset(row, row2)

        
        j += 1
    i += 1
    