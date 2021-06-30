import math
import random
import os
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers, reference_states #Convenient atomic information from reputable sources
from ase.md.andersen import Andersen #The NVT thermostat used in LoDiS
from ase.io import read, write #Input/Output manager for atoms
from ase.build import fcc111, fcc100
from asap3 import * #Just throw everything from asap3
from asap3 import Atoms, EMT, units, Trajectory #You can get specific utility from each of these sub-modules
from asap3.md.langevin import Langevin
from asap3.md.velocitydistribution import *

#Below are specific paramters for computing inter-atomic forces according to the RGL formula 
from GuptaParameters import AuCu_parameters as AuCu_Gupta
from GuptaParameters import AuPt_parameters as AuPt_Gupta
from GuptaParameters import Au_parameters as Au_Gupta







#This function is introduced for understanding the first column and giving the corresponding mass. For our code we will only use the Au mass, but other cases are included for further usage.
def mass (lista):
    #I introduce the values we read in the first column as a series of string.
    Au='Au'
    Ag='Ag'
    Al='Al'
    Cu='Cu'
    #We have written the values of the masses in terms of the mass of gold. That is to avoid divergences for very hig summation of masses when obtaining centers of masses.
    if lista==Au:
        m=float(196.966570/1000000)/float(196.966570/1000000)
    if lista==Ag:
        m=float(107.8683/1000000)/float(196.966570/1000000)
    if lista==Al:
        m=float(26.9815384/1000000)/float(196.966570/1000000)
    if lista==Cu:
        m=float(63.546/1000000)/float(196.966570/1000000)
    return m

#This function is introduced for understanding the first column and giving the corresponding mass.
def radiusatom (lista):
    #I introduce the values we read in the first column as a series of string.
    Au='Au'
    Ag='Ag'
    Al='Al'
    Cu='Cu'
    
    #Each possible variable is given its correspondent radius.
    if lista=='Au':
        r=1.44
        
    if lista=='Ag':
        r=6.
    if lista=='Al':
        r=3
    if lista=='Cu':
        r=3
    return r


#Here we will define the center of mass position:

def com(x,y,z,name):
    xcm=0
    ycm=0
    zcm=0
    Mt=0
    count=len(x)
    #The center of mass is obtained by computing the summation of al the positions within the position list times the mass of the atoms in the position and then dividing over the total mass.
    for j in range(0,count):
        xcm=xcm+(x[j]*mass(name[j]))
        ycm=ycm+(y[j]*mass(name[j]))
        zcm=zcm+(z[j]*mass(name[j]))
        Mt= Mt+mass(name[j])
    xcm=xcm/Mt
    ycm=ycm/Mt
    zcm=zcm/Mt
    return xcm, ycm, zcm


#The maxradius function find the furthest away atom from the center of mass of the NP and returns its distance from the center of mass.
def maxradius(x,y,z,xcm,ycm,zcm):
    r=0
    for j in range(0,len(x)):
        r0=((x[j]-xcm)**2+(y[j]-ycm)**2+(z[j]-zcm)**2)**(1/2)
        if r0>r:
            r=r0
    return r


#Here we will define the energy of a given foam by reading the xyz file in which the position of the atoms are displayed.

def RGL(c,lx,ly,lz):
        NP = read(c)
        #Set the dimensions of the cell
        Ico_Twin.set_cell((10,10,10))
        Ico_Twin.set_calculator(EMT())

        RGL_Calc = RGL(Au_Gupta) #The LoDiS type calculator

        Ico.set_cell((10,10,10))
        Ico.set_calculator(RGL_Calc)

        #Note the differences in setting calculators. This is due to the fact that we had to build our RGL
        #Whereas the EMT was pre-built and parameterised.

        print(Ico.get_potential_energy(), "ev \nUsing the RGL Calculator.\n")
        print(Ico_Twin.get_potential_energy(), "ev \nUsing the EMT model.\n")


#In continuation, we will show how to obtain a histogram representation of the distribution of atom probability depending on different variables.

#Radial distribution
def radiald(c,interval):
    NP= read(c)
    Distances = [ math.sqrt(a[0]**2 + a[1]**2 + a[2]**2) for a in NP.positions ]
    plt.hist(Distances, bins = interval, density = True)
    plt.title('Density probability with respact to the radial distance')
    plt.xlabel('X ($\AA$)' )
    plt.ylabel('Probability')
    plt.show()

#Horizontal distribution
def horizontald(c,interval):
    NP= read(c)
    Distances = [ a[0] for a in NP.positions ]
    plt.hist(Distances, bins = interval, density = True)
    plt.title('Density probability with respact to the x-distance')
    plt.xlabel('X ($\AA$)' )
    plt.ylabel('Probability')
    plt.show()

#Distribution on the z-axis
def azimutald(c,interval):
    NP= read(c)
    Distances = [ a[2] for a in NP.positions ]
    plt.hist(Distances, bins = interval, density = True)
    plt.title('Density probability with respact to the z-distance')
    plt.xlabel('Z ($\AA$) ')
    plt.ylabel('Probability')
    plt.show()


    
    
    
#The stirring function receives a certain nanofoam within the box and evaluates where to place the NP number i within it. The inputs are the dimensions of the box, the number i, the list of all other NP's com within the box xf, yf, zf, the number of recursions it has gone through. Moreover, r is the radius of the incoming NP, radius is the list of radii of the already displayed NP and interval is the number of places in the x-direction in which we will propose the incoming NP to be displayed.
def stirring(lx,ly,lz,i,xf,yf,zf,att,r,radius,interval):
        att=att+1
        print("Recursions: ",att)
        s=0
        t=0
        #We begin by proposing a random position in the yz-plane, that is where we will drop the NP.
        yo= r+radiusatom("Au")+(ly-2*r-2*radiusatom("Au"))*random.random()
        zo= r+radiusatom("Au")+(lz-2*r-2*radiusatom("Au"))*random.random()
        #We establish a while loop. We will evaluate all the possible interval number x-coordinates. For each position we will propose it as a possible x-position for the depositing NP. We will compare the NP in the suggested position with all the other NP in the system. If the distance between both NPs com is larger than the summation of the radius of both NP plus 2 times the radius of the gold atom we will consider the position to be valid. This will make t to increase by a unit of 1. If t becomes as large as the length of the list of com that will mean that the incoming NP is in an available coordinate and so it will remain there and the while-loop will break, returning the position. If the NP was not able to find any available position for all the interval x-positions the function will recall itself recursively and look for a new yz-position.
        while s!=interval:
            xo=-lx/2 +r+radiusatom("Au")+(lx-2*r-2*radiusatom("Au"))*(s/interval)
            s=s+1
            t=0
            for k in range(0,len(xf)):
                d=((xo-xf[k])**2+(yo-yf[k])**2+(zo-zf[k])**2)**(1/2)
                rd=r+radius[k]+2*radiusatom("Au")
                if rd<d:
                    t=t+1
                if t==len(xf):
                    xr=xo
                    yr=yo
                    zr=zo
                    s=interval
        if t!=len(xf):
            xr, yr, zr= stirring(lx,ly,lz,i,xf,yf,zf,att,r,radius,interval)
        return xr, yr, zr
        
        

        
        
        
        
        
def write_energy(a, step=[0]):
    #We have two commands which read the kinetic and potential energies of the system
    ekin = a.get_kinetic_energy()
    epot = a.get_potential_energy()
    #Then, we open a file in which we will write the energies
    with open('EnergyRGL.out', 'a') as outfile:
        outfile.write("%-10.5f  %-10.5f  %-10.5f  %10.1f\n" %
               (ekin, epot, ekin+epot, 2.0/3.0*ekin/units.kB/len(a)))
    step[0] += 1

    

def flour(lx,ly,lz,natoms):
    #In this function we will construct a slab with a given number of atoms "natoms" within a box of size lx, ly and lz.
    #We begin by computing both plates of the box.
    #In order to construct consistent gold plates we use the ase.build command for face centered cubic structures in the 111 direction. We also set a large enough number of atoms so they cover the whole range of the xy-plane.
    slab = fcc111('Au', size=(4,4,3), vacuum=0)
    
    #We want to know where to place this plate, it should be slightly outside the box so it does not overlap with the nanofoam within. The first thing we do is compute the maximum and minimum values of x, y and z of the slab to see their reach.
    
    xmax=0
    ymax=0
    zmax=0
    xmin=0
    ymin=0
    zmin=0

    x=[]
    y=[]
    z=[]
    xat=[]
    yat=[]
    zat=[]



    
    for i in range(0,len(slab.positions)):
        if slab.positions[i][0]>xmax:
            xmax=slab.positions[i][0]
        if slab.positions[i][0]<xmin:
            xmin=slab.positions[i][0]
        if slab.positions[i][1]>ymax:
            ymax=slab.positions[i][1]
        if slab.positions[i][1]<ymin:
            ymin=slab.positions[i][1]
        if slab.positions[i][2]>zmax:
            zmax=slab.positions[i][2]
        if slab.positions[i][2]<zmin:
            zmin=slab.positions[i][2]

    #We establish the lattice distances in our slab formation. Since we have set the slabs to be 4,4,3 atoms the necessary translation in the three direction to repeat itself will be:
    
    dx=8*radiusatom("Au")
    dy=8*radiusatom("Au")
    dz=6*radiusatom("Au")
    
    #We now establish how many of these slabs we will precise for our case. 
    
    a=(lx+20)/(dx)
    ax=int(a)
    a=(ly)/(dy)
    ay=int(a)+1
    a=(lz)/(dz)
    az=int(a)+1
    
    
    #We can now start building the nanofoam.




    for i in range (0,ay):
        for j in range (0,ax):
            x.append(-(lx/2)-10+(dx*j))
            y.append(dy*i)
            z.append(-zmax-radiusatom("Au"))
    for i in range (0,ay):
        for j in range (0,ax):
            x.append((-(lx/2)-10)+(dx)*j)
            y.append(dy*i)
            z.append(lz+zmin/2+radiusatom("Au"))

#Lastly, from the positions we have listed in x, y and z we save the positions of all the gold atoms.

    for i in range(0,len(x)):
        for j in range (0,len(slab.positions)):
            xat.append(slab.positions[j][0]+x[i])
            yat.append(slab.positions[j][1]+y[i])
            zat.append(slab.positions[j][2]+z[i])


        
 
    #We now proceed to compute the slab, that is, an arbitrary set of Au atoms displayed in form of fcc in the direction 100. Since the natoms will establish a cutoff in the formation of the slab we need to work with small fcc slabs which will be stacked. For our case, we use 12 Au atoms slabs. We will construct the slab by building yz layers. Once a layer is completed, we will compute another on top. We will continue with this method until all the natoms are displayed.
    slab = fcc100('Au', size=(2,2,3), vacuum=0)
    
    #The following steps are similar to the previous, we begin by computing the maximum and minimum positions of the slab.
    
    xmax=0
    ymax=0
    zmax=0
    xmin=0
    ymin=0
    zmin=0
    
    for i in range(0,len(slab.positions)):
        if slab.positions[i][0]>xmax:
            xmax=slab.positions[i][0]
        if slab.positions[i][0]<xmin:
            xmin=slab.positions[i][0]
        if slab.positions[i][1]>ymax:
            ymax=slab.positions[i][1]
        if slab.positions[i][1]<ymin:
            ymin=slab.positions[i][1]
        if slab.positions[i][2]>zmax:
            zmax=slab.positions[i][2]
        if slab.positions[i][2]<zmin:
            zmin=slab.positions[i][2]
            
    dx=(xmax-xmin)
    dy=(ymax-ymin)
    dz=(zmax-zmin)
    
    #Once again, we establish a cutoff, ax, ay and az are the number of atoms we will place in each direction. 
    a=((lx+20)/(2*radiusatom('Au')+dx))
    ax=int(a)-1
    a=((ly)/(radiusatom('Au')+dy))
    ay=int(a)-1
    a=((lz)/(2*radiusatom('Au')+dz))
    az=int(a)-1
    
    #We now establish the number of slabs which will be displayed. Since every of our slabs have 12 atoms we divide the number of atoms over 12 and we make it be an integer.
    numberslabs=int(natoms/12)
    
    #Afterwards we establish three controlling parameters. As we recall from earlier, we will construct the slab by making yz-layers. We begin a while-loop which will compute values until we run off slabs. For every step we will displace the following slab a step in the z-direction. Once we reach the limit, the value i resets to 0 and the slab is displaced one step into de y direction. After a number of slabs are displayed, i and j are reset to 0 and the slab moves one step in the x-direction. Moreover, for every step numberslabs decreases by a value of 1.
    i=0
    j=0
    k=0
    while numberslabs != 0:
        #We set the xyz-position of the slabs depending on the parameters i,j,k.
        xx=-lx/2+radiusatom('Au')-xmin+k*(dx+radiusatom('Au'))
        yy=radiusatom('Au')-ymin+j*(dy+radiusatom('Au'))
        zz=radiusatom('Au')-zmin+i*(dz+2*radiusatom('Au'))
        
        #We set the positions of the gold atoms.
        for l in range (0, len(slab.positions)):
            xat.append(slab.positions[l][0]+xx)
            yat.append(slab.positions[l][1]+yy)
            zat.append(slab.positions[l][2]+zz)
            
        #We establish the changes of parameters as we finish placing the gold positions.
        if i==az and j==ay:
            k=k+1
            i=0
            j=0
        elif i==az:
            j=j+1
            i=0
        else:
            i=i+1
            
        #The number of slabs decreases by one.
        numberslabs=numberslabs-1
        
        
    #We conclude the function by writing the "bulk.xyz" file with all the gold positions.
    with open('Bulk.xyz','w') as f:
        f.write(str(len(xat))+'\n')
        f.write('Au  Au \n')
        for i in range (0,len(xat)):
            f.write('Au \t  '+ str(xat[i])+'  \t  '+str(yat[i])+'  \t  '+str(zat[i]) +'\n')
    
    
    
               

###############################################################################################################################
        
        
                                                           #CASES       
        
        
############################################################################################################################### 
        
def llivia(c,M,lx,ly,lz,N):
    Vt=0 #Here we will write the occupied volume
    radius=[] #This list will contain the radius of all the NP within the box
    xfinal=[] #These next three values will contain the positions of all the atoms
    yfinal=[]
    zfinal=[]
    NPnumber=0 #This variable will keep track of the number of NP
    NPnumbercount=[] #We will use this variable to assign the corresponding NP to every atom.
    number=0 #These two variables will keep track of the number of atoms within the system
    numberini=0  
    names=[] #This last list will keep the name of the atoms so we can write them in the first column of the .xyz file.
    file=[]#We name all the possible
    nfile=[]#Number of atoms within each file
    matrix = np.zeros( (M+1, M+1) )#The adjacent NP matrix has 0 in all of its values
    
    
        #We begin by placing the first NP. The other NP will be displayed randomly but ensuring that the structure stays compact. 

                    
        #xf, yf and zf will be the places in which we will write the centers of masses of the NP, the check serves the purpose to observe the evolution of the program.
            
    xf=[0]
    yf=[0]
    zf=[0]
    check=0
        
        
    for i in range(0,M):
        NPnumber=NPnumber+1#We keep the number of NP updated.
        check=check+1#Simple check to follow the process.
        print('Check: ', check)
        #We pick a random file within our folder by creating a list of all the names of the files within the folder and picking a random slot within that list:
        nmlist=len(os.listdir(c))-1
        h=c+os.listdir(c)[random.randint(0,nmlist)]
        #We load the data and read the positions of all the gold atoms within the chosen NP.
        b=np.loadtxt(h, usecols=(1,2,3), skiprows=2)
        x=b[:,0]            
        y=b[:,1]
        z=b[:,2]
        #We also keep the name of the file in our file list and the number of gold atoms in the nfile list. We will use these two lists to construct the file: "NPs used in the configurations".
        file.append(h)
        nfile.append(len(x))
        #Lastly, we also keep the name of the elements. This was not strictly necessary for our case, since we are only using gold atoms, but in the case where multiatomic NP were to be considered, having a list with all the element names would be of use.
        #We also use the count number to have a parameter to keep in mind how many atoms we have in the NP.
        name=np.loadtxt(h, skiprows=2, usecols = (0),dtype='str')
        count=np.count_nonzero(x)
            

        #Here we read all the different types of atoms within the arbitrarily chosen NP and place it in the names list.
        for j in range (0, count):
            names.append(name[j])
                
            
            
        #We use the previously defined com function to obtain the center of mass of the NP.
        
        xcm, ycm, zcm = com(x,y,z,name)
            


       #In continuation we obtain the volume of the NP by observing the maximum com-atom distance and establishing as the radius of a supposidly speherical NP.
            
        r= maxradius(x,y,z,xcm,ycm,zcm)
        
        #We also add the obtained radius to a list, this list help us prevent NP from overlapping in the function stirring.
        radius.append(r)
        
        #We conclude by computing the volume of the NP as if it were a sphere with its radius being r. This ensures NP's from overlapping, since we stablish that all distance from the center of mass of the NP to a distance of r is occupied. For NP whose shape is signifficantly different to that of a sphere this assumption will derive in large interNP gold distances. More on that is discussed on the report.
        Vmol=(4./3.)*math.pi*r**3
        Vt=Vt+Vmol
       
            
            
            
            
            
        #We will now deposit the NPs in the box 1 by 1. 
        att=0 #This att number help us keep in track the number of recursions the overlap function is doing. If att appears to high numbers we can consider the box to be filled up, since the stirring function does not find xz-points for which the NP can be deposited.
        
        xr,yr,zr= stirring(lx,ly,lz,i,xf,yf,zf,att,r,radius,N)
        
        #After the overlap function we have succsessfylly depoisted the ith NP in the box. We now want to look if it is adjacent to other NP's. In order to do so, we use the  adjacent matrix. We compare the location of the center of mass of the new NP to the centers of mass of all the others. If the difference between the distance of an atom j into our recently displayed atom i is lower than the diameter of a gold atom we change the site (i,j) and (j,i) of the matrix from 0 to 1.
        for k in range (0,i):
            d=((xr-xf[k])**2+(yr-yf[k])**2+(zr-zf[k])**2)**(1/2)
            rd=r+radius[k]+2*radiusatom("Au")
            if d-rd<2*radiusatom("Au") :
                matrix[k,i]=1
                matrix[i,k]=1
            
        #Lastly we formally place the new NP com position to the list.
        xf.append(xr)
        yf.append(yr)
        zf.append(zr)


        
        #Now that we have succsessfully displayed the NP center we proceed to find the exact position of its atoms so that we can write them in the .xyz file. We define each atom position (xrot,yrot and zrot) as the position of the center of mass within the box + the position of the atom within the .xyz file - the position of the center of mass within the .xyz file.
        for k in range (0,count):
            xrot=xr+x[k]-xcm
            yrot=yr+y[k]-ycm
            zrot=zr+z[k]-zcm
                
                
            #Lastly, we write down the location of the atoms and the name of the located atoms.
            xfinal.append(xrot)
            yfinal.append(yrot)
            zfinal.append(zrot)
            number=number+1
            NPnumbercount.append(NPnumber)
            

    
    #Now we proceed to constructing the gold plates on both z-sides of the box. In order to construct consistent gold plates we use the ase.build command for face centered cubic structures in the 100 direction. We also set a large enough number of atoms so they cover the whole range of the xy-plane.
    slab = fcc111('Au', size=(4,4,3), vacuum=0)
    
    
    #We want to know where to place this plate, it should be slightly outside the box so it does not overlap with the nanofoam within. The first thing we do is compute the maximum and minimum values of x, y and z of the slab to see their reach.
    xmax=0
    ymax=0
    zmax=0
    xmin=0
    ymin=0
    zmin=0

    x=[]
    y=[]
    z=[]
    xat=[]
    yat=[]
    zat=[]

   
    for i in range(0,len(slab.positions)):
        if slab.positions[i][0]>xmax:
            xmax=slab.positions[i][0]
        if slab.positions[i][0]<xmin:
            xmin=slab.positions[i][0]
        if slab.positions[i][1]>ymax:
            ymax=slab.positions[i][1]
        if slab.positions[i][1]<ymin:
            ymin=slab.positions[i][1]
        if slab.positions[i][2]>zmax:
            zmax=slab.positions[i][2]
        if slab.positions[i][2]<zmin:
            zmin=slab.positions[i][2]

    #We establish the lattice distances in our slab formation. Since we have set the slabs to be 4,4,3 atoms the necessary translation in the three direction to repeat itself will be:
    
    dx=8*radiusatom("Au")
    dy=8*radiusatom("Au")
    dz=6*radiusatom("Au")
    
    #We now establish how many of these slabs we will precise for our case. 
    
    a=(lx+20)/(dx)
    ax=int(a)
    a=(ly)/(dy)
    ay=int(a)+1
    a=(lz)/(dz)
    az=int(a)+1
    
    

    #We can now start building the nanofoam.




    for i in range (0,ay):
        for j in range (0,ax):
            x.append(-(lx/2)-10+(dx*j))
            y.append(dy*i)
            z.append(-zmax-radiusatom("Au"))
    for i in range (0,ay):
        for j in range (0,ax):
            x.append((-(lx/2)-10)+(dx)*j)
            y.append(dy*i)
            z.append(lz+zmin/2+radiusatom("Au"))

    #Lastly, from the positions we have listed in x, y and z we save the positions of all the gold atoms.

    for i in range(0,len(x)):
        for j in range (0,len(slab.positions)):
            xat.append(slab.positions[j][0]+x[i])
            yat.append(slab.positions[j][1]+y[i])
            zat.append(slab.positions[j][2]+z[i])
        
     #We are finished with modeling, we can now proceed to writing our results.
        
        
 #######################################################################################################################       
    
    
    
    
    with open('Nanofoam.xyz', 'w') as f:
        #In order to satisfy the .xyz format, we must write in the first line the total number of gold atoms in our system. In order to represent this number we add the total number of atoms in all the NP (number) + the number of atoms in the total slab (through the length of the xat list).
        number1=str(number+len(xat))
        f.write(number1 + '\n')
        f.write('Au'+' '+'Au'+ '\n')
        
        #We begin by writing all the gold atoms of the plates. In order to differentiate them from the ones of the nanofoam we make that the fifth column indicates them as plate atoms with a value 0.
        for i in range (0,len(xat)):
            f.write('Au'+'   '+str(xat[i])+'   '+str(yat[i])+'   '+str(zat[i])+'   '+str(0)+' \n')
        #We write the atoms of the NP in the nanofoam while indicatin to which NP they correspond with the 5th column.
        for i in range(0,len(xfinal)): 
            f.write(names[i]+'    '+str(xfinal[i])+'    '+str(yfinal[i])+'    '+str(zfinal[i])+'    '+str(NPnumbercount[i])+'\n')
    
    

    #We finalize the code by writing the density of the desired foam within the box.
    Vbox=lx*ly*lz

    V_ratio=Vt/Vbox
    print('Occ Volume',Vt)
    print('Available Volume', Vbox)
    print('V_ratio:   ',V_ratio)
    
    
    
    if V_ratio<0.5:
        print('Low density system')

    if V_ratio>0.5 and V_ratio<2./3.:
        print('Medium density system')

    if V_ratio>2./3.:
        print('High density system')
        

    
    
    
    
    return V_ratio,radius, xf, yf, zf, file, nfile, matrix, number







################################################################################################################################
                                    #Second version case c
################################################################################################################################

#This function is just slightly different to the one of Llivia. It is adapted so we can define the probability proportions of the NP which will appear. It is specificaly made so that the cases A to E proposed for the project can be accomplished.

#The different probability for each particle is given by a list named "probs", which is now an input. This list has the normalized probability of each NP.


def Forn(c,M,lx,ly,lz,N,probs):
    Vt=0 #Here we will write the occupied volume
    radius=[] #This list will contain the radius of all the NP within the box
    xfinal=[] #These next three values will contain the positions of all the atoms
    yfinal=[]
    zfinal=[]
    NPnumber=0 #This variable will keep track of the number of NP
    NPnumbercount=[] #We will use this variable to assign the corresponding NP to every atom.
    number=0 #These two variables will keep track of the number of atoms within the system
    numberini=0  
    names=[] #This last list will keep the name of the atoms so we can write them in the first column of the .xyz file.
    file=[]#We name all the possible
    nfile=[]#Number of atoms within each file
    matrix = np.zeros( (M+1, M+1) )#The adjacent NP matrix has 0 in all of its values
    
    
        #We begin by placing the first NP. The other NP will be displayed randomly but ensuring that the structure stays compact. 

                    
        #xf, yf and zf will be the places in which we will write the centers of masses of the NP, the check serves the purpose to observe the evolution of the program.
            
    xf=[0]
    yf=[0]
    zf=[0]
    check=0
        
        
    for i in range(0,M):
        NPnumber=NPnumber+1#We keep the number of NP updated.
        check=check+1#Simple check to follow the process.
        print('Check: ', check)
        #We pick a random file within our folder by creating a list of all the names of the files within the folder and picking a random slot within that list:
        rndmnumber=random.random()
        controller=0
        for j in range(0,len(probs)):
            if controller<rndmnumber and rndmnumber< controller+probs[j]:
                h=c+os.listdir(c)[j]
            controller=controller+probs[j]
        #We load the data and read the positions of all the gold atoms within the chosen NP.
        b=np.loadtxt(h, usecols=(1,2,3), skiprows=2)
        x=b[:,0]            
        y=b[:,1]
        z=b[:,2]
        #We also keep the name of the file in our file list and the number of gold atoms in the nfile list. We will use these two lists to construct the file: "NPs used in the configurations".
        file.append(h)
        nfile.append(len(x))
        #Lastly, we also keep the name of the elements. This was not strictly necessary for our case, since we are only using gold atoms, but in the case where multiatomic NP were to be considered, having a list with all the element names would be of use.
        #We also use the count number to have a parameter to keep in mind how many atoms we have in the NP.
        name=np.loadtxt(h, skiprows=2, usecols = (0),dtype='str')
        count=np.count_nonzero(x)
            

        #Here we read all the different types of atoms within the arbitrarily chosen NP and place it in the names list.
        for j in range (0, count):
            names.append(name[j])
                
            
            
        #We use the previously defined com function to obtain the center of mass of the NP.
        
        xcm, ycm, zcm = com(x,y,z,name)
            


       #In continuation we obtain the volume of the NP by observing the maximum com-atom distance and establishing as the radius of a supposidly speherical NP.
            
        r= maxradius(x,y,z,xcm,ycm,zcm)
        
        #We also add the obtained radius to a list, this list help us prevent NP from overlapping in the function stirring.
        radius.append(r)
        
        #We conclude by computing the volume of the NP as if it were a sphere with its radius being r. This ensures NP's from overlapping, since we stablish that all distance from the center of mass of the NP to a distance of r is occupied. For NP whose shape is signifficantly different to that of a sphere this assumption will derive in large interNP gold distances. More on that is discussed on the report.
        Vmol=(4./3.)*math.pi*r**3
        Vt=Vt+Vmol
       
            
            
            
            
            
        #We will now deposit the NPs in the box 1 by 1. 
        att=0 #This att number help us keep in track the number of recursions the overlap function is doing. If att appears to high numbers we can consider the box to be filled up, since the stirring function does not find xz-points for which the NP can be deposited.
        
        xr,yr,zr= stirring(lx,ly,lz,i,xf,yf,zf,att,r,radius,N)
        
        #After the overlap function we have succsessfylly depoisted the ith NP in the box. We now want to look if it is adjacent to other NP's. In order to do so, we use the  adjacent matrix. We compare the location of the center of mass of the new NP to the centers of mass of all the others. If the difference between the distance of an atom j into our recently displayed atom i is lower than the diameter of a gold atom we change the site (i,j) and (j,i) of the matrix from 0 to 1.

        for k in range (0,i):
            d=((xr-xf[k])**2+(yr-yf[k])**2+(zr-zf[k])**2)**(1/2)
            rd=r+radius[k]+2*radiusatom("Au")
            if d-rd<2*radiusatom("Au") :
                matrix[k,i]=1
                matrix[i,k]=1
            
        #Lastly we formally place the new NP com position to the list.
        xf.append(xr)
        yf.append(yr)
        zf.append(zr)


        
        #Now that we have succsessfully displayed the NP center we proceed to find the exact position of its atoms so that we can write them in the .xyz file. We define each atom position (xrot,yrot and zrot) as the position of the center of mass within the box + the position of the atom within the .xyz file - the position of the center of mass within the .xyz file.
        for k in range (0,count):
            xrot=xr+x[k]-xcm
            yrot=yr+y[k]-ycm
            zrot=zr+z[k]-zcm
                
                
            #Lastly, we write down the location of the atoms and the name of the located atoms.
            xfinal.append(xrot)
            yfinal.append(yrot)
            zfinal.append(zrot)
            number=number+1
            NPnumbercount.append(NPnumber)
            

    
    #Now we proceed to constructing the gold plates on both z-sides of the box. In order to construct consistent gold plates we use the ase.build command for face centered cubic structures in the 100 direction. We also set a large enough number of atoms so they cover the whole range of the xy-plane.
    slab = fcc111('Au', size=(4,4,3), vacuum=0)
    
    
    #We want to know where to place this plate, it should be slightly outside the box so it does not overlap with the nanofoam within. The first thing we do is compute the maximum and minimum values of x, y and z of the slab to see their reach.
    xmax=0
    ymax=0
    zmax=0
    xmin=0
    ymin=0
    zmin=0

    x=[]
    y=[]
    z=[]
    xat=[]
    yat=[]
    zat=[]

   
    for i in range(0,len(slab.positions)):
        if slab.positions[i][0]>xmax:
            xmax=slab.positions[i][0]
        if slab.positions[i][0]<xmin:
            xmin=slab.positions[i][0]
        if slab.positions[i][1]>ymax:
            ymax=slab.positions[i][1]
        if slab.positions[i][1]<ymin:
            ymin=slab.positions[i][1]
        if slab.positions[i][2]>zmax:
            zmax=slab.positions[i][2]
        if slab.positions[i][2]<zmin:
            zmin=slab.positions[i][2]

    #We establish the lattice distances in our slab formation. Since we have set the slabs to be 4,4,3 atoms the necessary translation in the three direction to repeat itself will be:
    
    dx=8*radiusatom("Au")
    dy=8*radiusatom("Au")
    dz=6*radiusatom("Au")
    
    #We now establish how many of these slabs we will precise for our case. 
    
    a=(lx+20)/(dx)
    ax=int(a)
    a=(ly)/(dy)
    ay=int(a)+1
    a=(lz)/(dz)
    az=int(a)+1
    
    

    #We can now start building the nanofoam.




    for i in range (0,ay):
        for j in range (0,ax):
            x.append(-(lx/2)-10+(dx*j))
            y.append(dy*i)
            z.append(-zmax-radiusatom("Au"))
    for i in range (0,ay):
        for j in range (0,ax):
            x.append((-(lx/2)-10)+(dx)*j)
            y.append(dy*i)
            z.append(lz+zmin/2+radiusatom("Au"))

    #Lastly, from the positions we have listed in x, y and z we save the positions of all the gold atoms.

    for i in range(0,len(x)):
        for j in range (0,len(slab.positions)):
            xat.append(slab.positions[j][0]+x[i])
            yat.append(slab.positions[j][1]+y[i])
            zat.append(slab.positions[j][2]+z[i])
        
     #We are finished with modeling, we can now proceed to writing our results.
        
        
 #######################################################################################################################       
    
    
    
    
    with open('Nanofoam.xyz', 'w') as f:
        #In order to satisfy the .xyz format, we must write in the first line the total number of gold atoms in our system. In order to represent this number we add the total number of atoms in all the NP (number) + the number of atoms in the total slab (through the length of the xat list).
        number1=str(number+len(xat))
        f.write(number1 + '\n')
        f.write('Au'+' '+'Au'+ '\n')
        
        #We begin by writing all the gold atoms of the plates. In order to differentiate them from the ones of the nanofoam we make that the fifth column indicates them as plate atoms with a value 0.
        for i in range (0,len(xat)):
            f.write('Au'+'   '+str(xat[i])+'   '+str(yat[i])+'   '+str(zat[i])+'   '+str(0)+' \n')
        #We write the atoms of the NP in the nanofoam while indicatin to which NP they correspond with the 5th column.
        for i in range(0,len(xfinal)): 
            f.write(names[i]+'    '+str(xfinal[i])+'    '+str(yfinal[i])+'    '+str(zfinal[i])+'    '+str(NPnumbercount[i])+'\n')
    
    

    #We finalize the code by writing the density of the desired foam within the box.
    Vbox=lx*ly*lz

    V_ratio=Vt/Vbox
    print('Occ Volume',Vt)
    print('Available Volume', Vbox)
    print('V_ratio:   ',V_ratio)
    
    
    
    if V_ratio<0.5:
        print('Low density system')

    if V_ratio>0.5 and V_ratio<2./3.:
        print('Medium density system')

    if V_ratio>2./3.:
        print('High density system')
        

    
    
    
    
    return V_ratio,radius, xf, yf, zf, file, nfile, matrix, number










