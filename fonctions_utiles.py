# -*- coding: utf-8 -*-

""" 
Auteur : Marc-Antoine BUCHET

Fonctions utiles au traitement de données et à la génération de graphs 
scientifiques.
"""

import numpy as np
import matplotlib.pyplot as plt
import codecs
from scipy import optimize

################################################################################
# Fonctions pour lire les fichiers :
################################################################################        
def csv_reader(filename,sep=';',header=0):
    """ Lecteur de fichier CSV. Renvoie un tableau numpy.
    filename : nom du fichier à lire. Les données doivent être organisées en 
               colonne.
    sep : séparateur, point virgule par défaut
    header : nombre de lignes dans l'en-tête.
    
    Ne gère pas l'en-tête : passe simplement à la ligne suivante.
    """
    data=[]
    with codecs.open(filename,'r','utf-8') as f :
        l=-1
        for line in f :
            l+=1
            if l<header : continue
            temp=[float(elmt.strip()) for elmt in line.split(sep)]
            data.append(temp)
            l+=1
    return np.array(data)
    
################################################################################
# Fonctions pour tracer les graphs :
################################################################################   
#######################################
# Génération d'une figure et d'un axe :
#######################################
def start_fig(name,prefixe='') :
    """ Quelques lignes de code appelées systématiquement lors de la création
    d'une figure."""
    figname=prefixe+name
    fig=plt.figure(figname)
    ax=fig.add_subplot(111)
    return figname,fig,ax

###########################
# Sauvegarde de la figure :
###########################
def save_fig(figname,fig,figure_folder):
    """ Quelques lignes de code appelées systématiquement lors de la sauvegarde
    d'une figure."""
    plt.grid(True)
    plt.draw()
    plt.tight_layout()
    fig.savefig(figure_folder+figname+'.pdf')
    
################################
# Ajout d'un modèle à un graph :
################################
########################
def puissance_de_dix(x):
    """Donne la puissance de 10 du réel x"""
    return int(np.floor(np.log10(abs(x))))
    
#############################################################
def format_scientifique(x,dx,CS=2,fmt='std',fmt_puiss = 'e'):
    """Utilise la valeur de x et de son incertitude dx pour générer une chaine
    de caractères contenant x et dx avec le bon nombre de chiffres significatifs
    pour une notation scientifique.
    
    Dans tous les cas, on ne garde de dx que le nombre de chiffres significatifs
    choisi par CS (2 par défaut).
    
    Si x>dx (cas le plus usuel) alors on note x en écriture scientifique et 
    c'est dx qui impose le nombre de décimales à x. On a alors deux choix de 
    formats : 
    - le choix standard ('std') : on note l'incertitude avec le même nombre de
      chiffres significatifs que x et la même puissance, la puissance noté une
      fois à la fin en utilisant des parenthèses autout de x et dx.
      ex : x = 10,777777 et dx=0,33 alors on note x = (1,0778 \pm 0,033)x10^1
    - le choix métrologique ('NIST') : on note l'incertitude entre parenthèses
      à la fin des décimales de x. Cette notation n'est adaptée que si dx
      est suffisamment petit devant x.
     
    Si x<=dx alors on note dx en écriture scietifique et on ne garde de x
    que le nombre de chiffres significatifs correspondant. Ici, il n'y a pas 
    de choix de format."""
    def format_puissance(fmt_puiss) :
        """ Génère une chaine de caractère sur laquelle on pourra usiliser le
        formatage pour injecter la puissance"""
        # ATTENTION : format latex pas encore au point ! (pb avec la gestion des
        # accolades autour de la puissance)
        if fmt_puiss == 'latex' : 
            s = '\\times10^{2}'
        elif fmt_puiss == 'e' : 
            s = r'e{2:+03d}'
        else :
            raise ValueError('Format inconnu pour la puissance.')
        return s
        
    def format_standard(nbre_decimales,fmt_puiss) :
        """ Génère une chaine de caractère s au format standard sur laquelle
        on pourra usiliser le formatage suivant :
        s.format(m_x,m_dx,p)
        où : 
         - m_x est la "mantisse" de x
         - m_dx est la "mantisse" de dx
         - p une puissance.
        "mantisse" est utilisé avec des guillemets car ce ne sont pas des
        vraies mantisses, puisque l'une des deux commence par un ou plusieurs
        zéros (selon que x<dx ou x>dx)."""
        s = r'({0:.'+'{}'.format(nbre_decimales)+r'f} \pm {1:.'
        s += r'{}'.format(nbre_decimales)
        s += r'f})'+format_puissance(fmt_puiss)
        return s

    # On récupère les puissances de x et dx :
    p_x = puissance_de_dix(x)
    p_dx = puissance_de_dix(dx)
    # On gère la notation selon le cas x>dx ou x<=dx
    if x > dx : 
        # Le nombre de décimales dans l'écriture de x est ici imposé par la  
        # valeur de dx et le nombre de chiffres significatifs choisi :
        nbre_decimales = p_x-p_dx+CS-1
        # On récupère la mantisse de x :
        m_x=x/10**p_x
        if fmt in ['NIST']:
            # On récupère la mantisse de dx :
            m_dx = dx/10**p_dx
            # On récupère la valeur de l'incertitude sur les derniers chiffres 
            # de x avec le nombre de chiffres significatifs voulu (on ajoute 0.5 
            # pour arrondir à l'entier le plus proche et pas simplement à la
            # partie entière) :
            incertitude = int(m_dx*10**(CS-1)+0.5)
            # On formate la chaine de caractère :
            s = r'{0:.'+'{}'.format(nbre_decimales)+'f}'
            s = s.format(m_x)
            s += r'({0:2d})'.format(incertitude)
            s += format_puissance(fmt_puiss).format(p_x)
        elif fmt in ['std'] :
            # On récupère la "mantisse" de dx (à la même puissance que x) :
            m_dx = dx/10**p_x
            s = format_standard(nbre_decimales,fmt_puiss).format(m_x,m_dx,p_dx)
        else : 
            raise ValueError("Format choisi inconnu")
    elif x<=dx :
        # Le nombre de décimales dans l'écriture de dx est ici imposé par le
        # nombre de chiffres significatifs choisi :
        nbre_decimales = CS-1
        # On récupère la mantisse de dx :
        m_dx=dx/10**p_dx
        # On ramène la "mantisse" de x à la même puissance que dx :
        m_x = x/10**p_dx
        s = format_standard(nbre_decimales,fmt_puiss).format(m_x,m_dx,p_dx)
    return s

###############################################################################
def add_fit(ax,x_min,x_max,model,params,sigma_params,position=None,fontsize=11,
            N_points=1000,chisq=None,chisq_red=None,description=None):
    """ Ajoute un modele à une figure déjà existante.
    ax : instance de 'axis'
    x_min,x_max : valeurs extremes des abscisses
    modele : modèle de l'ajustement  model(x,param1,param2,...)
    params : liste contenant les paramètres du modèle
    sigma_params : liste contenant les incertitude des paramètres
                   même taille que params
    position : position de l'affichage de la description du modèle
               de la forme [abscisse,ordonnée]
    fontsize : taille de la police pour le texte décrivant le modèle
    N_points : nombre de points pour le modèle
    chisq : valeur du chi carré
    chisq_red : valeur du chi carré réduite
    description : liste de chaines de caractères contenant 
                  - la formule de "model"
                  - le nom des paramètres
                  de taille "size(params)+1" """
    # version du 29 aout 2018
    x=np.linspace(x_min,x_max,N_points)
    y=model(x,*params)
    ax.plot(x,y,label='Ajustement')
  
    string=r''
    if description==None :
        for i,elmt in enumerate(zip(params,sigma_params)):
            a,da=elmt
            string+=r'$param{0} = '.format(i)
            string+=format_scientifique(a,da)
            string+='$\n'
    else :
        string+=r'$'+description[0]+'$ \n'
        for a,da,name in zip(params,sigma_params,description[1:]):
            string+=name+r' = $'
            string+=format_scientifique(a,da)
            string+='$\n'
    if not chisq==None :
        string+=r'$\chi^2 = {0}$'.format(chisq)
        string+=' \n'
    if not chisq_red==None :
        string+=r'$\chi^2_{red} = '+'{0} $\n'.format(chisq_red)
    if position==None :
        ax.text(x_min+(x_max-x_min)/20.,3.*max(y)/4.,string,
                fontsize=fontsize)
    else :
        ax.text(position[0],position[1],string,
                fontsize=fontsize)
    return

################################################################################
# Moyenne glissante :
################################################################################ 
def smooth(y, box_pts):
    """ Moyenne glissante. """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

################################################################################
# Ajustement de données :
################################################################################
#######################
# Regression linéaire :
#######################
def reg_lin(x,y,sigma_y):
    """ Calcule les meilleurs paramètres de la droite y=ax+b et leurs incertitudes
    en tenant compte des incertitudes sur y.
    x,y,sigma : listes ou tableaux 1D """
    # version du 28_08_2019
    x=np.array(x)
    y=np.array(y)
    sigma_y=np.array(sigma_y)
    
    w=(1./sigma_y)**2
    Sw=sum(w)
    Sx=sum(w*x)
    Sy=sum(w*y)
    Sxy=sum(w*x*y)
    Sxx=sum(w*x*x)
    
    delta = Sw*Sxx-Sx**2
    a =(Sw*Sxy-Sx*Sy)/delta 
    b = (Sxx*Sy-Sx*Sxy)/delta
    da = np.sqrt(Sw/delta)
    db = np.sqrt(Sxx/delta)
    chisq = sum(w*((y-(a*x+b))**2))
    chisq_red = chisq/(np.size(x)-2.)
    
    return a,da,b,db,chisq,chisq_red

#################
# Moindres carrés
#################
def leastsq_wrapper(f, xdata, ydata, sigma, p0, ftol=1.49012e-8,
                    xtol=1.49012e-8, maxfev=0,full_output=0):
    """
    Wrapper for leastsq drawn from curve_fit but adapted a bit.
    Use non-linear least squares to fit a function, f, to data.
    Assumes ``ydata = f(xdata, *params) + eps``
    
    Parameters
    ----------
    f : callable
        The model function, f(x, ...). It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    xdata : An M-length sequence or an (k,M)-shaped array
        for functions with k predictors.
        The independent variable where the data is measured.
    ydata : M-length sequence
        The dependent data --- nominally f(xdata, ...)
    sigma : None or M-length sequence, these values are used as weights in the
        least-squares problem. `sigma` should describe one standard deviation
        errors of the input data points. The estimated covariance in `pcov` is
        based on these values.
    p0 : scalar, or N-length sequence
        Initial guess for the parameters.
        
    full_output : bool
        non-zero to return all optional outputs.
    ftol : float
        Relative error desired in the sum of squares.
    xtol : float
        Relative error desired in the approximate solution.
    maxfev : int
        The maximum number of calls to the function. If zero, then 100*(N+1) is
        the maximum where N is the number of elements in x0.
        
    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared error
        of ``f(xdata, *popt) - ydata`` is minimized
    pcov : 2d array
        The estimated covariance of popt. The diagonals provide the variance
        of the parameter estimate. To compute one standard deviation errors
        on the parameters use ``perr = np.sqrt(np.diag(pcov))``.
    chi_sq : scalar
        Value of the chi-square fonction evaluated with popt.
    infodict : dict
        a dictionary of optional outputs with the key s:
        
        ``nfev``
            The number of function calls
        ``fvec``
            The function evaluated at the output
        ``fjac``
            A permutation of the R matrix of a QR
            factorization of the final approximate
            Jacobian matrix, stored column wise.
            Together with ipvt, the covariance of the
            estimate can be approximated.
        ``ipvt``
            An integer array of length N which defines
            a permutation matrix, p, such that
            fjac*p = q*r, where r is upper triangular
            with diagonal elements of nonincreasing
            magnitude. Column j of p is column ipvt(j)
            of the identity matrix.
        ``qtf``
            The vector (transpose(q) * fvec).
            
    mesg : str
        A string message giving information about the cause of failure.
    ier : int
        An integer flag. If it is equal to 1, 2, 3 or 4, the solution was
        found. Otherwise, the solution was not found. In either case, the
        optional output variable 'mesg' gives more information.
    
    See Also
    --------
    leastsq
    
    Notes
    -----
    The algorithm uses the Levenberg-Marquardt algorithm through `leastsq`.
    Additional keyword arguments are passed directly to that algorithm.
    """
    # version du 28_08_2019
    def func(params, xdata, ydata, function, weights):
        return weights * (function(xdata, *params) - ydata)
    
    # Check input arguments
    if np.isscalar(p0):
        p0 = np.array([p0])
    
    xdata = np.array(xdata).flatten()
    ydata = np.array(ydata).flatten()
    
    
    args = (xdata, ydata, f,1.0 / np.asarray(sigma))
    res = optimize.leastsq(func, p0, args=args, full_output=1, ftol=ftol, xtol=xtol,
                           maxfev=maxfev)
    (popt, pcov, infodict, errmsg, ier) = res
    
    if ier not in [1, 2, 3, 4]:
        msg = "Optimal parameters not found: " + errmsg
        raise RuntimeError(msg)
    chisq = (np.asarray(func(popt, *args))**2).sum()
    if pcov is None:
        # indeterminate covariance
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(inf)
    perr = np.sqrt(np.diag(pcov))
    chisq_red=chisq/(len(xdata)-len(p0))
    if full_output :
        return popt, perr, chisq, chisq_red, (pcov, infodict, errmsg, ier)
    else : return popt, perr, chisq, chisq_red
