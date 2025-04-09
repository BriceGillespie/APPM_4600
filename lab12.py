import numpy as np
import matplotlib.pyplot as plt

def lgwt(N,a,b):
    x=np.zeros(N)
    w=np.zeros(N)
    e=3e-14
    M=(N+1)//2
    xm=0.5*(b+a)
    xl=0.5*(b-a)
    for i in range(M):
        z=np.cos(np.pi*(i+0.75)/(N+0.5))
        while True:
            p1=1.0
            p2=0.0
            for j in range(1,N+1):
                p3=p2
                p2=p1
                p1=((2*j-1)*z*p2-(j-1)*p3)/j
            pp=N*(z*p1-p2)/(z*z-1)
            z1=z
            z=z1-p1/pp
            if abs(z-z1)<e:
                break
        x[i]=xm-xl*z
        x[N-1-i]=xm+xl*z
        w[i]=2*xl/((1-z*z)*pp*pp)
        w[N-1-i]=w[i]
    return x,w

def compTrap(n,a,b,f):
    h=(b-a)/(n-1)
    x=np.linspace(a,b,n)
    y=f(x)
    I=0.5*h*(y[0]+2*np.sum(y[1:-1])+y[-1])
    return I,x,None

def compSimp(n,a,b,f):
    if n%2==0:
        n+=1
    h=(b-a)/(n-1)
    x=np.linspace(a,b,n)
    y=f(x)
    I=(h/3)*(y[0]+4*np.sum(y[1:-1:2])+2*np.sum(y[2:-2:2])+y[-1])
    return I,x,None

def compGauss(n,a,b,f):
    x,w=lgwt(n,a,b)
    I=np.sum(f(x)*w)
    return I,x,w

def adaptQuad(a,b,f,tol,n,method):
    m=50
    L=np.zeros(m)
    R=np.zeros(m)
    S=np.zeros(m)
    L[0]=a
    R[0]=b
    S[0],xInit,_=method(n,a,b,f)
    X=[xInit]
    i=1
    totalI=0
    splits=1
    while i<m:
        mid=0.5*(L[i-1]+R[i-1])
        v1,xTemp,_=method(n,L[i-1],mid,f)
        X.append(xTemp)
        v2,xTemp,_=method(n,mid,R[i-1],f)
        X.append(xTemp)
        if abs(v1+v2-S[i-1])>tol:
            L[i]=L[i-1]
            R[i]=mid
            S[i]=v1
            L[i-1]=mid
            S[i-1]=v2
            i+=1
            splits+=1
        else:
            totalI+=v1+v2
            i-=1
            if i==0:
                i=m
    return totalI,np.unique(np.concatenate(X)),splits

def adaptTrap(a,b,f,tol,n):
    return adaptQuad(a,b,f,tol,n,compTrap)

def adaptSimp(a,b,f,tol,n):
    return adaptQuad(a,b,f,tol,n,compSimp)

def adaptGauss(a,b,f,tol,n):
    return adaptQuad(a,b,f,tol,n,compGauss)

# We will do two things below:
# 1) Print the results for n=5 (already in the code)
# 2) Show "non-adaptive vs adaptive" error plots for each method for M in [2..10] (same style as in the lab test code)
# 3) Finally, show a single figure with the function and the adaptive meshes for n=5

a=0.1
b=2
f=lambda x:np.sin(1/x)
tol=1e-3
n=5

I1,X1,s1=adaptTrap(a,b,f,tol,n)
I2,X2,s2=adaptSimp(a,b,f,tol,n)
I3,X3,s3=adaptGauss(a,b,f,tol,n)

print("trap intervals:",s1,"Integral:",I1)
print("simpson intervals:",s2,"Integral:",I2)
print("gauss intervals:",s3,"Integral:",I3)

# For reference, approximate known value:
I_true=1.1455808341  

Ms=np.arange(2,11)
methods=[(compTrap,"Composite Trap"),(compSimp,"Composite Simpson"),(compGauss,"Gauss-Legendre")]
for method,method_name in methods:
    err_non=np.zeros(len(Ms))
    err_adapt=np.zeros(len(Ms))
    stored_mesh=None
    for i,mval in enumerate(Ms):
        nonI,_,_=method(mval,a,b,f)
        adaptI,mesh,_=adaptQuad(a,b,f,tol,mval,method)
        err_non[i]=abs(nonI-I_true)/abs(I_true)
        err_adapt[i]=abs(adaptI-I_true)/abs(I_true)
        if mval==2:
            stored_mesh=mesh
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].semilogy(Ms,err_non,'ro--')
    ax[0].set_ylim([1e-16,2])
    ax[0].set_xlabel('M')
    ax[0].set_title('nonadaptive '+method_name)
    ax[0].set_ylabel('relative Error')
    ax[1].semilogy(Ms,err_adapt,'ro--')
    ax[1].set_ylim([1e-16,2])
    ax[1].set_xlabel('M')
    ax[1].set_title('adaptive '+method_name)
    plt.suptitle(method_name+" vs. M")
    plt.tight_layout()
    plt.show()

    if stored_mesh is not None:
        fig2,ax2=plt.subplots()
        ax2.semilogy(stored_mesh,f(stored_mesh),'ro')
        ax2.set_title('Adaptive mesh for '+method_name+', M=2')
        plt.show()

