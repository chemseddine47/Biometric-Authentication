import matplotlib.pyplot as plt


"""def dessiner_graphe(titre, p1, x1, y1, p2=None, x2=None, y2=None, p3=None, x3=None, y3=None):

    plt.title(titre)
    if (x1 and y1 is not None) and ((x2 and y2 and x3 and y3) is None):
        plt.plot(x1, y1, "b", label=p1)
    elif (x3 and y3 is None) and ((x2 and y2 and x1 and y1) is not None):
        plt.plot(x1, y1, "b", label=p1)
        plt.plot(x2, y2, "g", label=p2)
    elif (x3 and y3 is not None) and ((x2 and y2 and x1 and y1) is not None):
        plt.plot(x1, y1, "b", label=p1)
        plt.plot(x2, y2, "g", label=p2)
        plt.plot(x3, y3, "r", label=p3)
    else:
        print("Erreur!")
    plt.xlabel('Temps')
    plt.ylabel('Quotient')
    plt.axis([0, 20, 0, 60])
    plt.legend()
    plt.show()"""


def dessiner_graphe(titre, p1, x1, y1, p2=None, x2=None, y2=None):

    plt.title(titre)
    plt.plot(x1, y1, "b", label=p1)
    plt.plot(x2, y2, "g", label=p2)
    plt.xlabel('Temps')
    plt.ylabel('Quotient')
    plt.axis([0, 20, 0, 60])
    plt.legend()
    plt.show()
