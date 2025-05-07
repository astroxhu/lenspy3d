import matplotlib
import numpy as np
from scipy.optimize import fsolve
class spherical_surface():
  def __init__(self,r,x_c,y_c=0.):
    self.r=r
    self.x_c=x_c
    self.y_c=y_c
    
  def line_circle(self,A,B,C,r,x0,y0): # line Ax+By+C=0
    aq=A*A+B*B
    bq=2*A*C+2*A*B*y0-2*B*B*x0
    cq=C*C+2*B*C*y0-B*B*(r*r-x0*x0-y0*y0)
    delta=bq*bq-4*aq*cq
    if delta<=0:
      print('warning: no intersection:delta=',delta)
      return [x0,y0,x0,y0,0]
    else:
      x1=(-bq+np.sqrt(delta))/2/aq
      x2=(-bq-np.sqrt(delta))/2/aq
      y1=-(A*x1+C)/B
      y2=-(A*x2+C)/B
      
    return [x2,y2,x1,y1,1]

    
  def spherical(self,A,B,C,n1,n2,ymin,ymax,face='left'):
    [x1,y1,x2,y2,cross]=self.line_circle(A,B,C,self.r,self.x_c,self.y_c)
    if cross==0:
      if face=="left":
        return[A,B,C,x1-self.r,x1+self.r]
      else:
        return[A,B,C,x1+self.r,x1+2*self.r]
    else:
      if face=='left':
        xs=x1
        ys=y1
        theta0=np.arcsin((ys-self.y_c)/self.r)
        xface=self.x_c-self.r
 
      else:
        xs=x2
        ys=y2
        theta0=-np.arcsin((ys-self.y_c)/self.r)
        xface=self.x_c+self.r

      
      theta1=-np.arctan(-A/B)
      phi1=theta0-theta1
      if ys > ymin and ys<ymax:
        phi2=np.arcsin(n1/n2*np.sin(phi1))
      else:
        phi2=phi1
        xs=xface
        ys=-(A*xface+C)/B
      k1=np.tan(phi2-theta0)
      c1=ys-k1*xs
      A1=k1
      B1=-1
      C1=c1
      return[A1,B1,C1,xs,xs+self.r]

class ray2d():
  def __init__(self,A,B,C,xs,xe,color='k'):
    self.A=A
    self.B=B
    self.C=C
    self.xs=xs
    self.xe=xe
    self.color=color

class gen_optics():
  rad2din=ray2d(0,0,0,0,0)
  #rad2dinter=ray2d(0,0,0,0,0)
  #rad2dout=ray2d(0,0,0,0,0)
  endset=False
  x_out=100.
  def __init__(self,apertureL=20,apertureR=20,x_c=10,y_c=0,dl=3,Rl=30,dr=2,Rr=-40):
    self.apertureL=apertureL
    self.apertureR=apertureR
    self.x_c=x_c
    self.y_c=y_c
    self.dl=dl
    self.Rl=Rl
    self.dr=dr
    self.Rr=Rr

  def ray_in(self,x0,h0,x_c,aperture,Ray):#,N_out,x_out):
    Nray=len(Ray)
    phi0=np.arctan((h0-aperture/2.)/(x_c-x0))
    delta_phi=np.arctan((h0+aperture/2.)/(x_c-x0))-np.arctan((h0-aperture/2.)/(x_c-x0))
    dphi=delta_phi/(Nray-1)
    result=np.ndarray(shape=(Nray,5))
    for i in range(Nray):
      phi1=phi0+dphi*i
      #print(phi1,dphi,i)
      k=-np.tan(phi1)
      c=h0-x0*k
      #result[i,0]=k
      #result[i,1]=-1
      #result[i,2]=c
      #result[i,3]=x0
      #result[i,4]=x_c
      Ray[i].A=k
      Ray[i].B=-1
      Ray[i].C=c
      Ray[i].xs=x0
      Ray[i].xe=x_c

    return Ray
   
  
  def sphlens(self,n_l,n_in,n_r):
    left_xc=self.x_c-self.dl+self.Rl
    left_r=abs(self.Rl)
    left_yc=self.y_c
    if self.Rl>0.:
      face='left'
    else:
      face='right'
    sph_L=spherical_surface(left_r,left_xc,y_c=left_yc)
    ray_come=self.ray2din
    yminL=self.y_c-self.apertureL/2.
    ymaxL=self.y_c+self.apertureL/2.
    [A1,B1,C1,xs,xe]=sph_L.spherical(ray_come.A,ray_come.B,ray_come.C,\
                                     n_l,n_in,yminL,ymaxL,face=face)
    self.ray2dinter=ray2d(A1,B1,C1,xs,xe)
    ray_come.xe=xs
  
    right_xc=self.x_c+self.dr+self.Rr
    right_r=abs(self.Rr)
    right_yc=self.y_c
    if self.Rr>0.:
      face='left'
    else:
      face='right'
    sph_R=spherical_surface(right_r,right_xc,y_c=right_yc)
    yminR=self.y_c-self.apertureR/2.
    ymaxR=self.y_c+self.apertureR/2.
    [A2,B2,C2,xs2,xe2]=sph_R.spherical(self.ray2dinter.A,self.ray2dinter.B,\
                                       self.ray2dinter.C,n_in,n_r,\
                                       yminR,ymaxR,face=face)
    self.ray2dout=ray2d(A2,B2,C2,xs2,xe2)
    self.ray2dinter.xe=xs2
    return
  
  def plotray(self,ax,nstep,ray0,lw=0.2,color_in='k',color_out='k'): # plot rays, dashed lines for virtural image
    xarr=np.linspace(ray0.xs,ray0.xe,nstep)
    k=-ray0.A/ray0.B
    c=-ray0.C/ray0.B
    if(ray0.xs<ray0.xe):
      ax.plot(xarr,k*xarr+c,c=color_out,lw=lw)
    else:
      ax.plot(xarr,k*xarr+c,c=color_out,lw=lw,ls='--')
      xarrf=np.linspace(ray0.xe,ray0.xe+abs(ray0.xe-ray0.xe),nstep)
      ax.plot(xarrf,k*xarrf+c,c=color_out,lw=1)
    return
  
  def plotallray(self,ax,nstep,lw=0.2,color_in='k',color_out='k',\
                 plotin=False,plotinter=True,plotout=True):
    if plotin:
      self.plotray(ax,nstep,self.ray2din,lw=lw,color_in=color_in,color_out=color_out)
    if plotinter:
      self.plotray(ax,nstep,self.ray2dinter,lw=lw,color_in=color_in,color_out=color_out)
    if self.endset:
      self.ray2dout.xe=self.x_out
    if plotout:
      self.plotray(ax,nstep,self.ray2dout,lw=lw,color_in=color_in,color_out=color_out)
  
  def plotlens(self,ax,nstep,color='C0'):
    rad=self.apertureL/2.
    left_xc=self.x_c-self.dl+self.Rl
    left_r=abs(self.Rl)
    xmin=self.x_c-self.dl
    yarr=np.linspace(-rad,rad,nstep)

    if self.Rl>0.:
      xmax=-np.sqrt(left_r**2-rad**2)+left_xc
      xplot=-np.sqrt(left_r**2-yarr**2)+left_xc
    else:
      xmax=np.sqrt(left_r**2-rad**2)+left_xc
      xplot=+np.sqrt(left_r**2-yarr**2)+left_xc
    xarr=np.linspace(xmin,xmax,nstep)
    #ax.plot(xarr,np.sqrt(left_r**2-(xarr-left_xc)**2)+self.y_c,c=color)
    #ax.plot(xarr,-np.sqrt(left_r**2-(xarr-left_xc)**2)+self.y_c,c=color)
    ax.plot(xplot,yarr+self.y_c,c=color)
    #ax.scatter(xmax,rad)
    rad1=self.apertureR/2.
    right_xc=self.x_c+self.dr+self.Rr
    right_r=abs(self.Rr)
    xmin1=self.x_c+self.dr
    yarr1=np.linspace(-rad1,rad1,nstep)
    if self.Rr>0.:
      xmax1=-np.sqrt(right_r**2-rad1**2)+right_xc
      xplot1=-np.sqrt(right_r**2-yarr1**2)+right_xc
    else:
      xmax1=np.sqrt(right_r**2-rad1**2)+right_xc
      xplot1=np.sqrt(right_r**2-yarr1**2)+right_xc

    
    xarr=np.linspace(xmin1,xmax1,nstep)
    #ax.plot(xarr,np.sqrt(right_r**2-(xarr-right_xc)**2)+self.y_c,c=color)
    #ax.plot(xarr,-np.sqrt(right_r**2-(xarr-right_xc)**2)+self.y_c,c=color)
    ax.plot(xplot1,yarr1+self.y_c,c=color)
    #ax.scatter(xmax1,rad1)
    xarr2=np.linspace(xmax,xmax1,10)
    yarr2=np.linspace(rad,rad1,10)
    rad2=rad
    xlow=xmax1
    if rad1>rad:
      rad2=rad1
      xlow=xmax
    ax.plot(xarr2,xarr2/xarr2*rad2+self.y_c,c=color)
    ax.plot(xarr2,-xarr2/xarr2*rad2+self.y_c,c=color)
    ax.plot(yarr2/yarr2*xlow,yarr2+self.y_c,c=color)
    ax.plot(yarr2/yarr2*xlow,-yarr2+self.y_c,c=color)

def readzmx(filename):
  i=0
  lens=dict()
  with open(filename,encoding='utf-16') as file:
    #file0=open(filename, 'rb').read()
    #mytext = file0.decode('utf-16')
    #mytext = mytext.encode('ascii', 'ignore')
    #lines = file.readlines()
    print((file))
    n_inarr=[]
    Rarr=[]
    diszarr=[]
    diamarr=[]
    for line in file:
      line0=line.rstrip()
      #print(line0)#,i,type(line0))
      i+=1
      if "CURV" in line0 or "GLAS" in line0 or "SURF" in line0 or "DISZ" in line0 or "DIAM" in line0:
        #print(line0,i)
        if "CURV" in line0 and len(line0)>30:
          print("CURV",line0)
          #print(1/float(line0[7:33]),len(line0))
          curv0=line0[7:].strip()
          curv=curv0.split(" ", 1)[0]
          R=1./float(curv)
          print(R)
          Rarr.append(R)
        if "DISZ" in line0:
          #print(line0[7:])
          disz=float(line0[7:])
          diszarr.append(disz)
        if "GLAS" in line0:
          print('GLAS',line0[20:])
          #glas=line0[20:-19]
          glas0=line0[20:].strip()
          glas=glas0.split(" ", 1)[0]
          #n_in=float(line0[20:-19])
          print('glas',glas)
          n_in=float(glas)
          #disz=float(line0[7:])
          n_inarr.append(n_in)
        if "DIAM" in line0:
          #print("DIAM",line0[7:-10])
          diam=float(line0[7:-10])
          diamarr.append(diam)
    lens["diam"]=diamarr
    lens["R"]=Rarr
    lens["disz"]=diszarr
    lens["n_in"]=n_inarr

  return lens

class lens():
  def __init__(self,n_elements,diaphram,glued,offset=1):
    self.n_elements=n_elements
    self.x_c=np.zeros(n_elements)
    self.y_c=np.zeros(n_elements)
    self.dl=np.zeros(n_elements)
    self.dr=np.zeros(n_elements)
    self.nl=np.zeros(n_elements)
    self.nr=np.zeros(n_elements)
    self.nin=np.zeros(n_elements)
    self.apl=np.zeros(n_elements)
    self.apr=np.zeros(n_elements)
    self.Rl=np.zeros(n_elements)
    self.Rr=np.zeros(n_elements)
    self.diaphram=diaphram
    self.glued=glued
    self.offset=offset

  def get_params(self,lens_params):
    self.disz=lens_params['disz']
    self.R=lens_params['R']
    self.nglass=lens_params['n_in']
    self.diam=lens_params['diam']
    #i_glued=0
    for i in range(self.n_elements):
      self.dr[i]=0.
      self.nin[i]=self.nglass[i]
      xcend=i*2+2+self.offset
      i_ap=i*2+1+self.offset
      i_Rl=i*2
      i_Rr=i*2+1
      self.nl[i]=1.
      self.nr[i]=1.
      if i >= self.diaphram:
        xcend+=1
        i_ap+=1
      for i_glued in self.glued:
        self.nl[i_glued]=self.nglass[i_glued-1]
        self.nr[i_glued-1]=self.nglass[i_glued]
        if i>i_glued-1:
          xcend-=1
          i_Rl-=1
          i_Rr-=1
          i_ap-=1
        #if i>i_glued-2:

      self.x_c[i]=sum(self.disz[self.offset:xcend])
      self.apl[i]=self.diam[i_ap]
      self.apr[i]=self.diam[i_ap+1]
      i_dl=xcend-1
      self.dl[i]=self.disz[i_dl]
      self.Rl[i]=self.R[i_Rl]
      self.Rr[i]=self.R[i_Rr]

    #### temper####
    #self.nin=[1.8, 1.773, 1.673, 1.74, 1.773, 1.788, 1.788]
    #self.nl=[1.,1.,1.,1.,self.nglass[3],1.,1.]
    #self.nr=[1.,1.,1.,self.nglass[4],1.,1.,1.]
    #self.ap=[38.0, 32.0, 28.0, 26.5, 26.5, 33.4,30.4]
    #self.x_c=[sum(self.disz[1:3]),sum(self.disz[1:5]),\
    #                                  sum(self.disz[1:7]),\
    #                                  sum(self.disz[1:10]),\
    #                                  sum(self.disz[1:11]),\
    #         sum(self.disz[1:13]),sum(self.disz[1:15])]
    #self.dl=[9.884, 9.108, 2.326, 1.938, 12.403, 8.333, 5.039]
    print("surface",self.dl)
    #self.dr=[0,0,0,0,0,0,0]
    #self.Rl=[78.687, 50.297, 138.143,  -34.407, -2906.977, -150.021, 284.63]
    #self.Rr=[471.434, 74.376, 34.326, -2906.977,-59.04699999999999, -57.89, -253.21700000000004]
  def addlens(self,apertureL=20,apertureR=20,x_c=10,y_c=0,dl=3,Rl=30,dr=2,Rr=-40,nl=1.,nin=1.7,nr=1.):
    self.n_elements+=1
    self.apl=np.append(self.apl,apertureL)
    self.apr=np.append(self.apr,apertureR)
    self.x_c=np.append(self.x_c,x_c)
    self.y_c=np.append(self.y_c,y_c)
    self.dl=np.append(self.dl,dl)
    self.Rl=np.append(self.Rl,Rl)
    self.dr=np.append(self.dr,dr)
    self.Rr=np.append(self.Rr,Rr)
    self.nl=np.append(self.nl,nl)
    self.nr=np.append(self.nr,nr)
    self.nin=np.append(self.nin,nin)



  def assemble(self,ax,Ray,lencolor='C0'):
    #Ray=[ray2d((-i*0+5)*0.02,-1,-10+(i*1-5)*2,-5,0) for i in range(Nray)]
    #Ray=[ray2d((-i*0+30)*0.02,-1,-70+(i*1-5)*2,-5,0) for i in range(Nray)]
    Nray=len(Ray)
    lensarr=[gen_optics(2*self.apl[i],2*self.apr[i],self.x_c[i],self.y_c[i],self.dl[i],\
                       self.Rl[i],self.dr[i],self.Rr[i]) for i in range(self.n_elements)]
    for i in range(self.n_elements):
      lensarr[i].plotlens(ax1,200,lenscolor=lenscolor)
    rayout=[]
    rayout1=[]
    for j in range(Nray):
      lensarr[0].ray2din=Ray[j]
      #print(self.nl[0])
      lensarr[0].sphlens(self.nl[0],self.nin[0],self.nr[0])
      for i in range(1,self.n_elements-1):
        lensarr[i].ray2din=lensarr[i-1].ray2dout
         #lens2.endset=True
        #lens2.x_out=45
        lensarr[i].sphlens(self.nl[i],self.nin[i],self.nr[i])
      rayout1.append(lensarr[self.n_elements-2].ray2dout)
      lensarr[self.n_elements-1].ray2din=lensarr[self.n_elements-2].ray2dout
      lensarr[self.n_elements-1].endset=True
      lensarr[self.n_elements-1].x_out=390
      lensarr[self.n_elements-1].sphlens(self.nl[self.n_elements-1],\
                                        self.nin[self.n_elements-1],\
                                        self.nr[self.n_elements-1])
      rayout.append(lensarr[self.n_elements-1].ray2dout)
      for i in range(self.n_elements):
        if i==0:
          plotin=True
        else:
          plotin=False
        lensarr[i].plotallray(ax,100,plotin=plotin)
    return [rayout,rayout1]

  def evaluate(self,arr):
    sy0=-36/2.
    sy1=36./2.
    Ny=8256
    pixel=(sy1-sy0)/Ny
    Nin=len(arr)
    sensor=np.linspace(sy0,sy1,Ny+1)
    count=np.zeros(Ny)
    for i in range(Nin):
      j=int(np.floor((arr[i]-sy0)/pixel))
      if j>0 and j<Ny:
        count[j]+=1
      else:
        count[0]+=0
    #return np.var(arr) #1./(max(count)-min(count))
    return 1./(max(count)-min(count))


  def findfocus(self,raylist,step=1.,dx=0.01,x0=150.,x1=500.):
    Nray=len(raylist)
    print('Nray',Nray,raylist)
    y0arr=np.zeros(Nray)
    y1arr=np.zeros(Nray)
    dy0=1e5
    rate=1.
    n_iter=0
    while(abs(x0-x1)>dx):
      x1=x0+step
      print("try",x1,step,n_iter)
      n_iter+=1
      if n_iter%100==0:
        print('n_iter',n_iter)
      if n_iter>1000:
        break
      for i in range(Nray):
        y0arr[i]=-raylist[i].A/raylist[i].B*x0-raylist[i].C/raylist[i].B
        y1arr[i]=-raylist[i].A/raylist[i].B*x1-raylist[i].C/raylist[i].B
        print('ABC',raylist[i].A,raylist[i].B,raylist[i].C)
      #dy0=max(y0arr)-min(y0arr)
      #dy1=max(y1arr)-min(y1arr)
      #dy0=np.var(y0arr)
      #dy1=np.var(y1arr)
      dy0=self.evaluate(y0arr)
      dy1=self.evaluate(y1arr)
      #print('dy0',dy0,'dy1',dy1,'n_iter',n_iter,max(y1arr),min(y1arr))
      if dy0>=dy1:
        step*=1.25
        x0=x1
        x1=x1+step
        rate=dy1/dy0
      else:
        step*=-0.8
        #if dy0/dy1<rate:
        #  step*=0.5
        #x1=x0+step


    return [x1,max(y1arr),min(y1arr)]

class aspherical_surface():
  def __init__(self,curv,conic,Asph,x_c,diam,norm=1.,y_c=0.):
    self.curv=curv
    self.conic=conic
    self.Asph=Asph
    self.norm=norm
    self.x_c=x_c
    self.y_c=y_c
    self.diam=diam
  def asph_func(self,r):
    A=self.Asph
    curv=self.curv
    conic=self.conic
    norm=self.norm
    m=len(A)
    #print('curv',curv,'conic',conic,'r',r)
    z=curv*r**2/(1+np.sqrt(1-(1+conic)*curv**2*r**2))
    for i in range(m):
      z+=A[i]*(r/norm)**(i*2+2)
    return z

  def dasph_dr(self,r):
    A=self.Asph
    curv=self.curv
    conic=self.conic
    norm=self.norm
    m=len(A)
    dz=2*curv*r/(1+np.sqrt(1-(1+conic)*curv**2*r**2))\
    +(conic+1)*curv**3*r**3/(np.sqrt(1-(1+conic)*curv**2*r**2)*\
     (np.sqrt(1-(1+conic)*curv**2*r**2)+1)**2)
    for i in range(m):
      dz+=A[i]*(r/norm)**(i*2+1)*(i*2+2)/norm
    return dz

  def line_asph(self,A,B,C,ymin,ymax): # line Ax+By+C=0
    if A==0:
      y1=-C/B
      r=y1-self.y_c
      x1=self.asph_func(r)+self.x_c
      return[x1,y1]
    else:
      y1=0.
      r1=y1-self.y_c
      y2=-(A*self.x_c+C)/B
      r2=y2-self.y_c
      func = lambda y: self.asph_func(y-self.y_c)+self.x_c+(B*y+C)/A
      #print(func(2),y2)
      y_sol = fsolve(func, y2)
      x_sol = -(B*y_sol+C)/A

    return [x_sol,y_sol]

  def plot_asph(self,ax,lw=1.5,color='C0'):
    y=np.linspace(-self.diam/2,self.diam/2,100)
    x=np.zeros(len(y))

    for i in range(100):
      x[i]=self.asph_func(y[i])
    ax.plot(x+self.x_c,y+self.y_c,lw=lw,c=color)


  def aspherical(self,A,B,C,n1,n2,ymin,ymax):
    [x1,y1]=self.line_asph(A,B,C,ymin,ymax)
    cross=1
    if cross==0:
      if face=="left":
        return[A,B,C,x1-self.r,x1+self.r]
      else:
        return[A,B,C,x1+self.r,x1+2*self.r]
    else:
      xs=x1
      ys=y1
      k_asph=self.dasph_dr(y1-self.y_c)
      theta0=np.arcsin(k_asph/np.sqrt(1+k_asph**2))
      xface=self.x_c


      theta1=-np.arctan(-A/B)
      phi1=theta0-theta1
      if ys > ymin and ys<ymax:
        phi2=np.arcsin(n1/n2*np.sin(phi1))
      else:
        phi2=phi1
        xs=xface
        ys=-(A*xface+C)/B
      k1=np.tan(phi2-theta0)
      c1=ys-k1*xs
      A1=k1
      B1=-1
      C1=c1
      if self.curv>0.:
        R0=1./self.curv
      else:
        R0=10.
      return[A1,B1,C1,xs,xs+R0]

class asph_optics(gen_optics):
  def __init__(self,apertureL=20,apertureR=20,x_c=10,y_c=0,dL=3,\
               curvL=0.033,dR=2,curvR=-0.025,AsphL=[1.,1.,1.],AsphR=[1.,1.,1.],conicL=-1,conicR=-1,normL=1.,normR=1.):
    self.ray2din=ray2d(0,0,0,0,0)
    self.endset=False
    self.x_out=100.
    self.apertureL=apertureL
    self.apertureR=apertureR
    self.x_c=x_c
    self.y_c=y_c
    self.dL=dL
    self.curvL=curvL
    self.dR=dR
    self.curvR=curvR
    self.conicL=conicL
    self.conicR=conicR
    self.AsphL=AsphL
    self.AsphR=AsphR
    self.normL=normL
    self.normR=normR
    left_xc=self.x_c-self.dL
    right_xc=self.x_c+self.dR
    self.asph_L=aspherical_surface(self.curvL,self.conicL,self.AsphL,left_xc,self.apertureL)
    self.asph_R=aspherical_surface(self.curvR,self.conicR,self.AsphR,right_xc,self.apertureR)
  def asphlens(self,n_l,n_in,n_r):
    left_xc=self.x_c-self.dL
    left_yc=self.y_c
    self.asph_L=aspherical_surface(self.curvL,self.conicL,self.AsphL,left_xc,self.apertureL)
    ray_come=self.ray2din
    yminL=self.y_c-self.apertureL/2.
    ymaxL=self.y_c+self.apertureL/2.
    [A1,B1,C1,xs,xe]=self.asph_L.aspherical(ray_come.A,ray_come.B,ray_come.C,\
                                     n_l,n_in,yminL,ymaxL)
    if (n_l <= 1.01):
        self.ray2dinter=ray2d(A1,B1,C1,xs,xe,ray_come.color)
    else:
        self.ray2dinter=ray_come
    ray_come.xe=xs
  
    right_xc=self.x_c+self.dR
    right_yc=self.y_c

    self.asph_R=aspherical_surface(self.curvR,self.conicR,self.AsphR,right_xc,self.apertureR)
    yminR=self.y_c-self.apertureR/2.
    ymaxR=self.y_c+self.apertureR/2.
    [A2,B2,C2,xs2,xe2]=self.asph_R.aspherical(self.ray2dinter.A,self.ray2dinter.B,\
                                       self.ray2dinter.C,n_in,n_r,\
                                       yminR,ymaxR)
    self.ray2dout=ray2d(A2,B2,C2,xs2,xe2,self.ray2dinter.color)
    self.ray2dinter.xe=xs2
    return
  def plotalens(self,ax,nstep,lw=1.5,color='C0'):
    rad=self.apertureL/2.
    left_xc=self.x_c-self.dL

    xmin=self.x_c-self.dL
    yarr=np.linspace(-rad,rad,nstep)

    rad1=self.apertureR/2.
    right_xc=self.x_c+self.dR
    xmin1=self.x_c+self.dR
    yarr1=np.linspace(-rad1,rad1,nstep)

    self.asph_L.plot_asph(ax,lw=lw,color=color)
    self.asph_R.plot_asph(ax,lw=lw,color=color)
    xmax=self.asph_L.asph_func(self.apertureL/2.)+left_xc
    xmax1=self.asph_R.asph_func(self.apertureR/2.)+right_xc

    xarr2=np.linspace(xmax,xmax1,10)
    yarr2=np.linspace(rad,rad1,10)
    rad2=rad
    xlow=xmax1
    if rad1>rad:
      rad2=rad1
      xlow=xmax
    ax.plot(xarr2,xarr2/xarr2*rad2+self.y_c,lw=lw,c=color)
    ax.plot(xarr2,-xarr2/xarr2*rad2+self.y_c,lw=lw,c=color)
    ax.plot(yarr2/yarr2*xlow,yarr2+self.y_c,lw=lw,c=color)
    ax.plot(yarr2/yarr2*xlow,-yarr2+self.y_c,lw=lw,c=color)

def readzmx_asph(filename,n_element=10):
  i=0
  lens=[dict() for x in range(n_element)]
  surf=[dict() for x in range(2*n_element)]
  #glass=dict()
  with open(filename,encoding='utf-16') as file:
    #file0=open(filename, 'rb').read()
    #mytext = file0.decode('utf-16')
    #mytext = mytext.encode('ascii', 'ignore')
    lines = file.readlines()
    print(len(lines))
    surfidx=[]
    glassidx=[]
    surftype=[]
    n_inarr=[]
    Rarr=[]
    diszarr=[]
    diamarr=[]
    for idx in range(len(lines)):
      line0=lines[idx].strip()
      #print('idx',idx)
      #print(lines[idx])
      #print(line0)#,i,type(line0))
      if "SURF" in line0:# or "GLAS" in line0 or "SURF" in line0 or "DISZ" in line0 or "DIAM" in line0:
        #print(lines[idx+1].strip())
        surfidx.append(idx)

        if "CURV" in line0 and len(line0)>30:
          #print("CURV",line0)
          #print(1/float(line0[7:33]),len(line0))
          curv0=line0[7:].strip()
          curv=curv0.split(" ", 1)[0]
          R=1./float(curv)
          #print(R)
          Rarr.append(R)
        if "DISZ" in line0:
          #print(line0[7:])
          disz=float(line0[7:])
          diszarr.append(disz)
        if "GLAS" in line0:
          #print('GLAS',line0[20:])
          #glas=line0[20:-19]
          glas0=line0[20:].strip()
          glas=glas0.split(" ", 1)[0]
          #n_in=float(line0[20:-19])
          #print('glas',glas)
          n_in=float(glas)
          #disz=float(line0[7:])
          n_inarr.append(n_in)
        if "DIAM" in line0:
          #print("DIAM",line0[7:-10])
          diam=float(line0[7:-10])
          diamarr.append(diam)
    surfidx.append(len(lines)-1)
    print('num of surface',len(surfidx))
    for i in range(len(surfidx)-1):
      #print(lines[idx].strip())
      for j in range(surfidx[i],surfidx[i+1]):
        line0=lines[j].strip()
        #print('b',line0)
        if line0[0:4]=='GLAS':
          print(line0)
          glassidx.append(i)
          break
    idx0=glassidx[0]
    for i in range(0,len(surfidx)-1):
      surf[i-idx0]['num']=i-idx0
      for j in range(surfidx[i],surfidx[i]+3):
        line0=lines[j].strip()
        if line0[0:4]=='TYPE':
          type0=line0[4:].strip()
          surftype.append(type0)
          #print(type0)
          break

      isglass=False
      isconic=False
      print(i,type0)
      surf[i]['type']=type0
      if type0=='EVENASPH' or type0=='STANDARD' or type0=='XASPHERE':
        parm_arr=[]
        norm_arr=[]
        xdat_arr=[]
        for jj in range(j,surfidx[i+1]):
          line0=lines[jj].strip()
          if line0[0:4]=='CURV':
            curv0=line0[4:].strip()
            curv=curv0.split(" ", 1)[0]
            #print(curv)
            surf[i]['curv']=float(curv)
          if type0=='EVENASPH':
            if line0[0:4]=='PARM':
              parm0=line0[4:].strip()
              parm=parm0.split(" ", 1)[1]
              #print(parm)
              parm_arr.append(float(parm))
              surf[i]['parm']=parm_arr
          if type0=='XASPHERE':
            if line0[0:4]=='XDAT':
              xdat0=line0[4:].strip()
              xdat=xdat0.split(" ")[1]
              #norm=xdat0.split(" ")[4]
              xdat_arr.append(float(xdat))
              #norm_arr.append(float(norm))
              surf[i]['xdat']=xdat_arr
              #surf[i-idx0]['norm']=xdat_arr[1]
          if type0=='STANDARD':
            surf[i]['parm']=[0.]
          if line0[0:4]=='DISZ':
            disz0=line0[4:].strip()
            disz=disz0.split(" ", 1)[0]
            #print(disz)
            surf[i]['disz']=float(disz)
          if line0[0:4]=='DIAM':
            diam0=line0[4:].strip()
            diam=diam0.split(" ", 1)[0]
            #print(diam)
            surf[i]['diam']=float(diam)
          if line0[0:4]=='GLAS':
            isglass=True
            n_in0=line0[4:].strip()
            #print(n_in0)
            n_in=n_in0.split(" ")[3]
            #print('n_in',n_in)
            surf[i]['n_in']=float(n_in)
          if line0[0:4]=='CONI':
            isconic=True
            coni0=line0[4:].strip()
            coni=coni0.split(" ", 1)[0]
            #print(diam)
            surf[i]['conic']=float(coni)

        if isglass==False:
          surf[i]['n_in']=1.
        if isconic==False:
          surf[i]['conic']=0.

    print(surf)
## assmemble lens elements
    i_lens=0
    for i in range(idx0,len(surfidx)-2):
      print(i)
      if surf[i]['n_in']>1.01:
        lens[i_lens]['num']=i_lens+1
        xc=0.
        lens[i_lens]['curv_L']=surf[i]['curv']
        lens[i_lens]['curv_R']=surf[i+1]['curv']
        lens[i_lens]['type_L']=surf[i]['type']
        lens[i_lens]['type_R']=surf[i+1]['type']

        if i>0:
          lens[i_lens]['n_L']=surf[i-1]['n_in']
        else:
          lens[i_lens]['n_L']=1.
        lens[i_lens]['n_in']=surf[i]['n_in']
        lens[i_lens]['n_R']=surf[i+1]['n_in']
        lens[i_lens]['diam_L']=2*surf[i]['diam']
        lens[i_lens]['diam_R']=2*surf[i+1]['diam']

        for ii in range(1,i):
          xc+=surf[ii]['disz']
        lens[i_lens]['xc']=xc
        lens[i_lens]['thick']=surf[i]['disz']

        if surf[i]['type']=='EVENASPH' or surf[i]['type']=='STANDARD':
          lens[i_lens]['parm_L']=surf[i]['parm']
          lens[i_lens]['parm_R']=surf[i+1]['parm']
          lens[i_lens]['conic_L']=surf[i]['conic']
          lens[i_lens]['conic_R']=surf[i+1]['conic']

        if surf[i]['type']=='XASPHERE':
          lens[i_lens]['xdat_L']=surf[i]['xdat'][2:]
          lens[i_lens]['xdat_R']=surf[i+1]['xdat'][2:]
          lens[i_lens]['norm_L']=surf[i]['xdat'][1]
          lens[i_lens]['norm_R']=surf[i+1]['xdat'][1]
          lens[i_lens]['conic_L']=surf[i]['conic']
          lens[i_lens]['conic_R']=surf[i+1]['conic']

        i_lens+=1


  return lens[0:i_lens]

class lensnew():
  def __init__(self, lens_params):
    self.lens_params=lens_params
    self.n_elements=len(lens_params)
    print('num of elements',len(lens_params))
    print(self.lens_params)
    self.lensarr=[]
  def assemble(self,ax,lw=1.5,lenscolor='k'):
    lensarr=[]
    lensarr=[asph_optics(apertureL=1,apertureR=1\
                  ,x_c=1,y_c=0,dL=0.,curvL=1,\
                  dR=1,curvR=1,AsphL=[0],\
                  AsphR=[0],conicL=0,conicR=0) for i in range(self.n_elements)]

    for i in range(self.n_elements):
      if self.lens_params[i]['type_L']=="EVENASPH" or self.lens_params[i]['type_L']=="STANDARD":
        print(i,self.lens_params[i]['num'])
        diamL=self.lens_params[i]['diam_L']
        diamR=self.lens_params[i]['diam_R']
        xc=self.lens_params[i]['xc']
        thick=self.lens_params[i]['thick']
        curvL=self.lens_params[i]['curv_L']
        curvR=self.lens_params[i]['curv_R']
        conicL=self.lens_params[i]['conic_L']
        conicR=self.lens_params[i]['conic_R']
        AL=self.lens_params[i]['parm_L']
        AR=self.lens_params[i]['parm_R']
        lensarr[i]=asph_optics(apertureL=diamL,apertureR=diamR\
                  ,x_c=xc,y_c=0,dL=0.,curvL=curvL,\
                  dR=thick,curvR=curvR,AsphL=AL,\
                  AsphR=AR,conicL=conicL,conicR=conicR)
        lensarr[i].plotalens(ax,1000,lw=lw,color=lenscolor)
      if self.lens_params[i]['type_L']=="XASPHERE":
        print(i,self.lens_params[i]['num'])
        diamL=self.lens_params[i]['diam_L']
        diamR=self.lens_params[i]['diam_R']
        xc=self.lens_params[i]['xc']
        thick=self.lens_params[i]['thick']
        curvL=self.lens_params[i]['curv_L']
        curvR=self.lens_params[i]['curv_R']
        conicL=self.lens_params[i]['conic_L']
        conicR=self.lens_params[i]['conic_R']
        AL=self.lens_params[i]['xdat_L']
        AR=self.lens_params[i]['xdat_R']
        lensarr[i]=asph_optics(apertureL=diamL,apertureR=diamR\
                  ,x_c=xc,y_c=0,dL=0.,curvL=curvL,\
                  dR=thick,curvR=curvR,AsphL=AL,\
                  AsphR=AR,conicL=conicL,conicR=conicR)
        lensarr[i].plotalens(ax,1000,color=lenscolor)

        #lensarr.append(lens1)

    self.lensarr=lensarr
    #for i in range(self.n_elements-1):
      #self.lensarr[i].plotalens(ax,1000)
  def raytrace(self,ax,Ray,raywidth=0.2,rayend=1000):
    Nray=len(Ray)
    rayout=[]
    rayout1=[]
    for j in range(Nray):
      self.lensarr[0].ray2din=Ray[j]
      #print(self.nl[0])
      self.lensarr[0].asphlens(self.lens_params[0]['n_L'],self.lens_params[0]['n_in'],self.lens_params[0]['n_R'])
      for i in range(1,self.n_elements-1):
        self.lensarr[i].ray2din=self.lensarr[i-1].ray2dout
         #lens2.endset=True
        #lens2.x_out=45
        self.lensarr[i].asphlens(self.lens_params[i]['n_L'],self.lens_params[i]['n_in'],self.lens_params[i]['n_R'])
      rayout1.append(self.lensarr[self.n_elements-2].ray2dout)
      self.lensarr[self.n_elements-1].ray2din=self.lensarr[self.n_elements-2].ray2dout
      self.lensarr[self.n_elements-1].endset=True
      self.lensarr[self.n_elements-1].x_out=rayend
      self.lensarr[self.n_elements-1].asphlens(self.lens_params[self.n_elements-1]['n_L'],\
                                        self.lens_params[self.n_elements-1]['n_in'],\
                                        self.lens_params[self.n_elements-1]['n_R'])
      rayout.append(self.lensarr[self.n_elements-1].ray2dout)
      for i in range(self.n_elements):
        if i==0:
          plotin=True
        else:
          plotin=False
        if Nray > 20:
          raystep = Nray//3
          #raystep = 1
          if j % raystep == 0:# or j == Nray-1:
            self.lensarr[i].plotallray(ax,200,lw=raywidth,plotin=plotin,color_in=Ray[j].color,color_out=Ray[j].color)
        else:
            self.lensarr[i].plotallray(ax,200,lw=raywidth,plotin=plotin,color_in=Ray[j].color,color_out=Ray[j].color)
    return [rayout,rayout1]


  def evaluate(self,arr):
    sy0=-36/2.
    sy1=36./2.
    Ny=8256
    pixel=(sy1-sy0)/Ny
    Nin=len(arr)
    sensor=np.linspace(sy0,sy1,Ny+1)
    count=np.zeros(Ny)
    for i in range(Nin):
      j=int(np.floor((arr[i]-sy0)/pixel))
      if j>0 and j<Ny:
        count[j]+=1
      else:
        count[0]+=0
    #return np.var(arr) #1./(max(count)-min(count))
    return 1./(max(count)-min(count))

  def findfocus(self,raylist,step=1.,dx=0.01,x0=150.,x1=500.):
    Nray=len(raylist)
    print('Nray',Nray)
    y0arr=np.zeros(Nray)
    y1arr=np.zeros(Nray)
    dy0=1e5
    dy1=2*dy0
    rate=1.
    n_iter=0
    while(abs(x0-x1)>dx):
      x1=x0+step
      #print("try",x1,step,n_iter)
      n_iter+=1
      if n_iter%100==0:
        print('n_iter',n_iter, x1, step,dy0,dy1)
      if n_iter>1000:
        break
      for i in range(Nray):
        y0arr[i]=-raylist[i].A/raylist[i].B*x0-raylist[i].C/raylist[i].B
        y1arr[i]=-raylist[i].A/raylist[i].B*x1-raylist[i].C/raylist[i].B
        #print('ABC',raylist[i].A,raylist[i].B,raylist[i].C)
      #dy0=max(y0arr)-min(y0arr)
      #dy1=max(y1arr)-min(y1arr)
      #dy0=np.var(y0arr)
      #dy1=np.var(y1arr)
      dy0=self.evaluate(y0arr)
      dy1=self.evaluate(y1arr)
      #print('dy0',dy0,'dy1',dy1,'n_iter',n_iter,max(y1arr),min(y1arr))
      if dy0>=dy1:
        step*=1.25
        x0=x1
        x1=x1+step
        rate=dy1/dy0
      else:
        step*=-0.8
        #if dy0/dy1<rate:
        #  step*=0.5
        #x1=x0+step


    return [x1,max(y1arr),min(y1arr)]
