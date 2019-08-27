# -*- coding: utf-8 -*-

""" 
Auteur : Marc-Antoine BUCHET
Date : 30/05/2018

PTSI2 - Lycée Livet - Nantes

Fonctions utiles au traitement de données.
"""

from pylab import *

import codecs

################################################################################
# Fonctions pour lire les fichiers :
################################################################################        
def kill_caract_speciaux(x) :
    """Elimine les caractères spéciaux de tabulation et retour à la ligne
    en bord de chaines de caractères.
    Si x est une liste de chaine de caractères, fait l'élimination pour chaque
    élément, sinon sur la chaine x."""
    if type(x)==list : 
        return [kill_caract_speciaux(elmt) for elmt in x]
    elif type(x)==unicode or type(x)==str :
        caract_speciaux = ['\n','\r','\t',' ']
        temp=x
        while temp and temp[0] in caract_speciaux :
            try : temp = temp[1:]
            except IndexError : return '' 
        while temp and temp[-1] in caract_speciaux : temp = temp[:-1]
        return temp
    else : raise TypeError("Type of x should be string, unicode or list of such")

def csv_reader(filename,sep=';',header=0):
    """ Lecteur de fichier CSV.
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
            temp=[float(kill_caract_speciaux(elmt)) for elmt in line.split(sep)]
            data.append(temp)
            l+=1
    return array(data)
    
################################################################################
# Fonctions pour tracer les graphs :
################################################################################    
def start_fig(name,prefixe='') :
    figname=prefixe+name
    fig=figure(figname)
    ax=fig.add_subplot(111)
    return figname,fig,ax

def end_fig(figname,fig,figure_folder):
    grid()
    draw()
    tight_layout()
    fig.savefig(figure_folder+figname+'.pdf')

def elargir_graph(ax,coeff_x=5./100.,coeff_y=5./100.) :
    """
    Fonction pour élargir la fenêtre d'un graph.
    ax : instance d'axe
    coeff_x : taux d'élargissement selon l'axe des abscisses
    coeff_y : taux d'élargissement selon l'axe des ordonnées
    
    Pour élargir dans une seule direction, mettre 0. au coefficient de 
    l'autre direction 
    """
    x0,x1=ax.get_xlim()
    ax.set_xlim(x0-coeff_x*(x1-x0),x1+coeff_x*(x1-x0))
    y0,y1=ax.get_ylim()
    ax.set_ylim(y0-coeff_y*(y1-y0),y1+coeff_y*(y1-y0))
    
##############################
# Ajout d'un modèle à un graph
##############################
def add_fit(ax,x_min,x_max,model,params,sigma_params,position=None,
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
    N_points : nombre de points pour le modèle
    chisq : valeur du chi carré
    chisq_red : valeur du chi carré réduite
    description : liste de chaines de caractères contenant 
                  - la formule de "model"
                  - le nom des paramètres
                  de taille "size(params)+1" """
    # version customisée le 14 septmbre 2018
    def puissance_de_dix(x):
        """ donne la puissance de 10 du réel x"""
        return int(floor(log10(abs(x))))
    
    def format_generator(x):
        """ renvoie un chaine de caractère qui correspond au format voulu pour x"""
        dix=puissance_de_dix(x)
        return ur'{0:.'+ur'{}'.format(abs(dix)+1)+ur'f}'

    def string_generator(x,dx):
        """utilise la valeur de dx pour générer une chaine de caractères
        contenant x au format scientifique avec le bon nombre de chiffres
        significatifs"""
        dix=puissance_de_dix(x)
        
        y=x/(10**dix)
        temp_str=format_generator(dx)
        temp_str=temp_str.format(y)
        if dix==0 : temp_str=ur'$'+temp_str+ur'$'
        else :
            temp_str=ur'$'+temp_str+ur' \times\ 10^{'+str(dix)+'}$'
        temp_str2=format_generator(dx)
        temp_str2=temp_str2.format(dx/10**dix)
        if dix==0 : temp_str2=ur'$'+temp_str2+ur'$'
        else :
            temp_str2=ur'$'+temp_str2+ur' \times\ 10^{'+str(dix)+'}$'
        return temp_str,temp_str2
        
    x=linspace(x_min,x_max,N_points)
    y=model(x,*params)
    ax.plot(x,y,label='Ajustement')
  
    params_string=ur''
    if description==None :
        i=1
        for a,da in zip(params,sigma_params):
            params_string+=ur'$param{2} = {0} \pm {1} $'.format(a,da,i)
            params_string+='\n'
            i+=1
    else :
        params_string+=ur'$'+description[0]+'$ \n'
        for a,da,name in zip(params,sigma_params,description[1:]):
            temp_str1,temp_str2=string_generator(a,da)
            params_string+='$'+name+ur'=$ '+temp_str1
            params_string+=ur' $\pm$ '+temp_str2+' \n'
    if not chisq==None :
        params_string+=ur'$\chi^2 = {0}$'.format(chisq)
        params_string+=' \n'
    if not chisq_red==None :
        params_string+=ur'$\chi^2_{red} = '+'{0} $\n'.format(chisq_red)
    if position==None :
        ax.text(x_min+(x_max-x_min)/20.,3.*max(y)/4.,params_string)
    else :
        ax.text(position[0],position[1],params_string)
    return
    
################################################################################
# Regression linéaire :
################################################################################
def reg_lin(x,y,sigma_y):
    """ Calcule les meilleurs paramètres de la droite y=ax+b et leurs incertitudes
    en tenant compte des incertitudes sur y.
    x,y,sigma : listes ou tableaux 1D """
    # version du 25_09_2015
    x=array(x)
    y=array(y)
    sigma_y=array(sigma_y)
    
    w=(1./sigma_y)**2
    Sw=sum(w)
    Sx=sum(w*x)
    Sy=sum(w*y)
    Sxy=sum(w*x*y)
    Sxx=sum(w*x*x)
    
    delta = Sw*Sxx-Sx**2
    a =(Sw*Sxy-Sx*Sy)/delta 
    b = (Sxx*Sy-Sx*Sxy)/delta
    da = sqrt(Sw/delta)
    db = sqrt(Sxx/delta)
    chisq = sum(w*((y-(a*x+b))**2))
    chisq_red = chisq/(size(x)-2.)
    
    return a,da,b,db,chisq,chisq_red