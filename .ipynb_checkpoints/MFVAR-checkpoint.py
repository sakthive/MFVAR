# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:54:57 2019

@author: Aubrey Poon
Bayesian Mixed Frequency State Space VAR model with DL priors
"""

# From line 9 to 113 do not touch, this is all basic setup stuff
def clear_all():
    """ clears variables """
    import os
    os.system('cls')
    import IPython as ip
    ip.get_ipython().magic('reset -sf')

clear_all()

import pandas as pd
import numpy as np
import math
import scipy.linalg 
from scipy.linalg import block_diag
from scipy.stats import gamma 
from scipy.stats import norm
from scipy.stats import invgauss
from scipy.stats import invgamma
import scipy as sp
from scipy.sparse import coo_matrix
import time
import warnings
warnings.filterwarnings("ignore")
from numpy import asarray
from numpy import savetxt

## define random draws from a gen
def psi(x,alpha,lam):
    return -alpha*(scipy.cosh(x)-1)-lam*(scipy.exp(x) - x - 1)

def dpsi(x,alpha,lam):
    return -alpha*scipy.sinh(x) - lam*(scipy.exp(x) - 1)

def g(x,sd,td,f1,f2):
    a = 0
    b = 0
    c = 0
    if x >= -sd and x <= td:
        a=1
    elif x > td:
        b = f1
    elif x < -sd:
        c = f2
    return a + b + c    

def GIGrnd(p,a,b,sam):
    lam = p
    omega = math.sqrt(a*b)
    swap = 0
    if lam < 0:
        lam = lam*-1
        swap = 1
    alpha = scipy.sqrt(omega**2 + lam**2) - lam
    # Find t
    x = -psi(1,alpha,lam)
    if x >= .5 and x <= 2:
        t=1
    elif x > 2:
        t = scipy.sqrt(2/(alpha + lam))
    elif x < 0.5:
        t = scipy.log(4/(alpha + 2*lam))
    # Find s
    x = -psi(1,alpha,lam)
    if x >= .5 and x <= 2:
        s=1
    elif x > 2:
        s = scipy.sqrt(4/(alpha*scipy.cosh(1) + lam))
    elif x < 0.5:
        s = np.min(np.array([1/lam,scipy.log(1 + 1/alpha + scipy.sqrt(1/alpha**2 + 2/alpha))]).reshape(1,2))
    # Generation
    eta = -psi(t,alpha,lam)
    zeta = -dpsi(t,alpha,lam)
    theta = -psi(-s,alpha,lam)
    xi = dpsi(-s,alpha,lam)
    p = 1/xi
    r = 1/zeta
    td = t - r*eta
    sd = s - p*theta
    q = td + sd
    bigX = np.zeros((sam,1))
    for i in range (sam):
        dd = 1
        while dd > 0:
            U = np.random.rand(1)
            V = np.random.rand(1)
            W = np.random.rand(1)
            if U < (q/(p+q+r)):
                bigX[i,:] = -sd + q*V
            elif U < ((q+r)/(p+q+r)):
                bigX[i,:] = td - r*scipy.log(V)
            else:
                bigX[i,:] = -sd + p*scipy.log(V)
        
            f1 = scipy.exp(-eta - zeta*(bigX[i,:] -t))
            f2 = scipy.exp(-theta + xi*(bigX[i,:] + s))
            
            if W*g(bigX[i,:],sd,td,f1,f2) <= scipy.exp(psi(bigX[i,:],alpha,lam)):
                break
    bigX = scipy.exp(bigX)*(lam/omega + scipy.sqrt(1 + (lam/omega)**2))
    if swap == 1:
        bigX = 1/bigX
    return bigX/scipy.sqrt(a/b)

#Here are the number of draws and burnin
nsim = 100 # number of draws
burnin = 10 # number of burnin draws

ts = pd.read_csv("quarterly.csv",header=None) # Load quarterly dataset
dataq = ts.values

ts = pd.read_csv("annual.csv",header=None) # Load annual dataset
dataa = ts.values

tm = dataq[:,-1]
ya = np.kron(dataa[:,1:],np.ones((4,1)))
yq = dataq[:,1:-1]
Tnew = len(yq) - len(ya)
if Tnew > 0:
     ya = np.ma.row_stack((ya,np.ones((Tnew,1))*ya[-1,:]))    
     
ydata = np.column_stack((yq,ya))
p = 7 # no. of lags
quarters = tm[p:]
yact0 = ydata[0:p,:]
yact = ydata[p:,:]
y = ydata[p:,:]
ystore = y

nq = 5 # no. of quartely variables
na = 12 # no. of annual variables
n = nq + na
T = len(yact) # no. of periods
k = n+p*n**2 # no. of parameters
m = int((n*(n-1))*.5) # no. of covariance
Lid = np.nonzero(np.tril(np.ones((n,n)),-1))
L = np.eye(n)

# Construct X
X = np.zeros((T,1))
for i in range (p):
     X = np.column_stack((X,ydata[p-(1+i):len(ydata)-(i+1),:]))
X[:,0] = 1

for i in range(n):
    if i == 0:
        idn = 0
    else:
        idn = np.ma.row_stack((idn,np.ones((i,1)).reshape(i,1)*i))
idn = idn[1:,:]        
       

# Define few things
# prior of sig
nu = 5
S0 = (nu-1)*.01

# cross-sectional restriction
cserr = 0.0001
S0cs = .0001
nucs = 10000

err1 = np.ma.row_stack((np.zeros((n,1)),cserr))
err2 = np.ma.row_stack((np.zeros((nq,1)),cserr))

beta = np.zeros((n,n*p+1))
a = np.zeros((m,1))
sig2 = np.ones((n,1))*.01
invVbeta = np.ones((n,n*p+1))*.01
invVa = np.ones((m,1))*.01


varthetabeta = 0.1
zetabeta = 0.1
varthetaa= 0.1
zetaa = 0.1
alpa = .5 # Hyperparameter controls shrinkage on covariances a's
alpb = .5 # Hyperparameter controls shrinkage on beta's

# do not modify 188-208
store_reg1 = np.zeros((nsim,T))
store_reg2 = np.zeros((nsim,T))
store_reg3 = np.zeros((nsim,T))
store_reg4 = np.zeros((nsim,T))
store_reg5 = np.zeros((nsim,T))
store_reg6 = np.zeros((nsim,T))
store_reg7 = np.zeros((nsim,T))
store_reg8 = np.zeros((nsim,T))
store_reg9 = np.zeros((nsim,T))
store_reg10 = np.zeros((nsim,T))
store_reg11 = np.zeros((nsim,T))
store_reg12 = np.zeros((nsim,T))
store_beta = np.zeros((nsim,k))
store_sig2 = np.zeros((nsim,n))

m11 = np.column_stack((np.eye(nq),np.zeros((nq,n*p-nq))))
m12 = np.column_stack((np.zeros((na,nq)),np.eye(na)*.25,np.zeros((na,nq)),np.eye(na)*.5,np.zeros((na,nq)),np.eye(na)*.75,np.zeros((na,nq)),np.eye(na),np.zeros((na,nq)),np.eye(na)*.75,np.zeros((na,nq)),np.eye(na)*.5,np.zeros((na,nq)),np.eye(na)*.25))
m13 = np.column_stack((np.zeros((1,nq)),np.ones((1,na))*1/na,np.zeros((1,n*p-n))))
M1 = np.ma.row_stack((m11,m12,m13))
M2 = np.ma.row_stack((m11,m13))

start = time.time()

# Start MCMC
for i in range((nsim + burnin)):
    for ii in range (n):
        if ii == 0:
            # Sample beta
            zt = X
            yhat = y[:,ii].reshape(T,1)
            km = int(k/n)
            Ktheta = np.diag(invVbeta[ii,:]) + np.dot(zt.T,zt)/sig2[ii,:]
            Ctheta = scipy.linalg.cholesky(Ktheta, lower=True)
            theta_hat = scipy.linalg.solve(Ctheta.T,scipy.linalg.solve(Ctheta,np.asarray(np.dot(zt.T,yhat)*(1/sig2[ii,:]))))
            beta[ii,:] = (theta_hat + scipy.linalg.solve(Ctheta.T,np.random.standard_normal(km).reshape(km,1))).T
          
            # sample sig2
            e = yhat - np.dot(zt,beta[ii,:].reshape(n*p+1,1));
            newnu = nu + T/2;
            S0new = S0 + np.sum(e**2)*.5 
            sig2[ii,:] = invgamma.rvs(newnu,scale = S0new)        
        else:
            # Sample beta
            zt = X
            yhat = y[:,ii].reshape(T,1) - np.dot(-y[:,0:ii],a[np.where(idn == ii)].reshape(ii,1))
            km = int(k/n)
            Ktheta = np.diag(invVbeta[ii,:]) + np.dot(zt.T,zt)/sig2[ii,:]
            Ctheta = scipy.linalg.cholesky(Ktheta, lower=True)
            theta_hat = scipy.linalg.solve(Ctheta.T,scipy.linalg.solve(Ctheta,np.asarray(np.dot(zt.T,yhat)*(1/sig2[ii,:]))))
            beta[ii,:] = (theta_hat + scipy.linalg.solve(Ctheta.T,np.random.standard_normal(km).reshape(km,1))).T
            
            # Sample a
            Xa = -y[:,0:ii]
            yaerr = y[:,ii].reshape(T,1) - np.dot(zt,beta[ii,:].reshape(n*p+1,1))
            Ka = np.diag(invVa[np.where(idn == ii)].reshape(ii,1)) + np.dot(Xa.T,Xa)/sig2[ii,:]
            Ca = scipy.linalg.cholesky(Ka, lower=True)
            ahat = scipy.linalg.solve(Ca.T,scipy.linalg.solve(Ca,np.asarray(np.dot(Xa.T,yaerr)*(1/sig2[ii,:]))))
            atilde = ahat + scipy.linalg.solve(Ca.T,np.random.standard_normal(ii).reshape(ii,1))
            a[np.where(idn == ii)] = atilde.T
           
            
            # sample sig2
            e = y[:,ii].reshape(T,1) - np.dot(zt,beta[ii,:].reshape(n*p+1,1)) - np.dot(Xa,atilde.reshape(ii,1))
            newnu = nu + T/2;
            S0new = S0 + np.sum(e**2)*.5 
            sig2[ii,:] = invgamma.rvs(newnu, scale = S0new)    
    
    bet = np.reshape(beta.T,k,1).reshape(k,1) 
    L[Lid] = a.T
    
    # Sample psibeta
    nutaubet = varthetabeta*(zetabeta/np.absolute(bet))
    invtaubet = invgauss.rvs(nutaubet, scale =1)
    taubeta = 1/(invtaubet + 10**-6)  
    
    # sample zetabeta
    zetabeta = GIGrnd(k*(alpb-1),1,2*np.sum(np.absolute(bet)/varthetabeta),1) + 10**-6
    
    # sample varthetabeta
    bigL = np.zeros((k,1))
    for v in range(k):
        bigL[v,:] = GIGrnd(alpb-1,1,2*np.absolute(bet[v,:]),1)
    varthetabeta = bigL/np.sum(bigL) + 10**-6
    invVbeta = (np.reshape(1/(taubeta*(varthetabeta**2)*zetabeta**2),(n*p+1,n))).T 
    
    # Sample psia
    nutaua = varthetaa*(zetaa/np.absolute(a))
    invtaua = invgauss.rvs(nutaua, scale =1)
    taua = 1/(invtaua + 10**-6)  
    
     # sample zetaa
    zetaa = GIGrnd(m*(alpa-1),1,2*np.sum(np.absolute(a)/varthetaa),1) + 10**-6
    
    # sample varthetaa
    bigL = np.zeros((m,1))
    for v in range(m):
        bigL[v,:] = GIGrnd(alpa-1,1,2*np.absolute(a[v,:]),1)
    varthetaa = bigL/np.sum(bigL) + 10**-6
    invVa = 1/(taua*(varthetaa**2)*zetaa**2) 
    
    # Convert structural parameters to reduce form parameters

    invSigma = np.dot(np.dot(L.T,np.eye(n)*(1/sig2)),L)
    Sigma = scipy.linalg.inv(invSigma)
    
    B1 = scipy.linalg.solve(L,beta[:,0]).reshape(n,1)
    B2 = scipy.linalg.solve(L,beta[:,1:n+1])
    B3 = scipy.linalg.solve(L,beta[:,n+1:2*n+1])
    B4 = scipy.linalg.solve(L,beta[:,2*n+1:3*n+1])
    B5 = scipy.linalg.solve(L,beta[:,3*n+1:4*n+1])
    B6 = scipy.linalg.solve(L,beta[:,4*n+1:5*n+1])
    B7 = scipy.linalg.solve(L,beta[:,5*n+1:6*n+1])
    B8 = scipy.linalg.solve(L,beta[:,6*n+1:7*n+1])
    betacoeff = np.column_stack((B1,B2,B3,B4,B5,B6,B7,B8))
    
    # Draw mixed frequency variable
    state0 = np.zeros((n*p,1))
    P0 = np.eye(n*p)*.01
    F0 = np.ma.row_stack((B1,np.zeros((n*p-n,1))))
    F1 = np.ma.row_stack((betacoeff[:,1:],np.column_stack((np.eye(n*p-n),np.zeros((n*p-n,n))))))
    bigGam = block_diag(Sigma,np.zeros((n*p-n,n*p-n))).reshape(n*p,n*p)
    
    Phat_store = np.zeros((T,), dtype=np.object)
    Phat = np.zeros((T,), dtype=np.object)
    state_store = np.zeros((n*p,T))
    
    # Kalman filter
    if quarters[0] == 4:
        shat = F0 + np.dot(F1,state0)
        Phat[0] = np.dot(np.dot(F1,P0),F1.T) + bigGam
        K1 = np.dot(Phat[0],M1.T)
        K2 = np.dot(np.dot(M1,Phat[0]),M1.T) + err1
        K = np.dot(K1,scipy.linalg.pinv(np.asarray(K2)))
        kerr = np.ma.row_stack((yact[0,:].reshape(n,1),yact[0,0])) - np.dot(M1,shat)
        state_store[:,0] = (shat + np.dot(K,kerr)).T
        Phat_store[0] = np.dot(np.eye(n*p) - np.dot(K,M1),Phat[0])
    else:
        shat = F0 + np.dot(F1,state0)
        Phat[0] = np.dot(np.dot(F1,P0),F1.T) + bigGam
        K1 = np.dot(Phat[0],M2.T)
        K2 = np.dot(np.dot(M2,Phat[0]),M2.T) + err2
        K = np.dot(K1,scipy.linalg.pinv(np.asarray(K2)))
        kerr = np.ma.row_stack((yact[0,0:nq].reshape(nq,1),yact[0,0])) - np.dot(M2,shat)
        state_store[:,0] = (shat + np.dot(K,kerr)).T
        Phat_store[0] = np.dot(np.eye(n*p) - np.dot(K,M2),Phat[0])
        
    for t in range(1,T):
        if quarters[t] == 4:
            shat = F0 + np.dot(F1,state_store[:,t-1].reshape(n*p,1))
            Phat[t] = np.dot(np.dot(F1,Phat_store[t-1]),F1.T) + bigGam
            K1 = np.dot(Phat[t],M1.T)
            K2 = np.dot(np.dot(M1,Phat[t]),M1.T) + err1
            K = np.dot(K1,scipy.linalg.pinv(np.asarray(K2)))
            kerr = np.ma.row_stack((yact[t,:].reshape(n,1),yact[t,0])) - np.dot(M1,shat)
            state_store[:,t] = (shat + np.dot(K,kerr)).T
            Phat_store[t] = np.dot(np.eye(n*p) - np.dot(K,M1),Phat[t])
        else:
            shat = F0 + np.dot(F1,state_store[:,t-1].reshape(n*p,1))
            Phat[t] = np.dot(np.dot(F1,Phat_store[t-1]),F1.T) + bigGam
            K1 = np.dot(Phat[t],M2.T)
            K2 = np.dot(np.dot(M2,Phat[t]),M2.T) + err2
            K = np.dot(K1,scipy.linalg.pinv(np.asarray(K2)))
            kerr = np.ma.row_stack((yact[t,0:nq].reshape(nq,1),yact[t,0])) - np.dot(M2,shat)
            state_store[:,t] = (shat + np.dot(K,kerr)).T
            Phat_store[t] = np.dot(np.eye(n*p) - np.dot(K,M2),Phat[t])
    
    # Kalman smoother
    post_state = np.zeros((n*p,T))
    post_Phat = np.zeros((T,), dtype=np.object)
    post_state[:,-1] = state_store[:,-1] 
    post_Phat[-1] = Phat_store[-1]
    
    for tt in range(T-2,-1,-1):
        C = np.dot(np.dot(Phat_store[tt],F1.T),np.linalg.pinv(np.asarray(Phat[tt+1])))
        Cerr = post_state[:,tt+1].reshape(n*p,1) - F0 - np.dot(F1,state_store[:,tt].reshape(n*p,1))
        post_state[:,tt] = (state_store[:,tt].reshape(n*p,1) + np.dot(C,Cerr)).T
        tmp = np.eye(n*p) - np.dot(C,F1)
        g1 = np.dot(np.dot(tmp,Phat_store[tt]),tmp.T)
        g2 = np.dot(np.dot(C,bigGam),C.T)
        g3 = np.dot(np.dot(C,post_Phat[tt+1]),C.T)
        post_Phat[tt] = np.dot(np.dot(g1,g2),g3)
        
    # Cross-sectional error
    err3 = np.mean(state_store[nq:n,:],axis=0) - yact[:,0]
    S0err = S0cs + np.sum(err3**2)
    cserr = invgamma.rvs(nucs + T/2, scale = S0err) 
    err1 = np.ma.row_stack((np.zeros((n,1)),cserr))
    err2 = np.ma.row_stack((np.zeros((nq,1)),cserr))   
    
    # reconfigure dataset
    Y1 = np.ma.row_stack((yact0[:,nq:n],post_state[nq:n,:].T))     
    yy = np.column_stack((ydata[:,0:nq],Y1))
    
    X = np.zeros((T,1))
    for iv in range (p):
        X = np.column_stack((X,yy[p-(1+iv):len(yy)-(iv+1),:]))
    X[:,0] = 1
    y = yy[p:,:]
      

    if i > burnin:
        ic = i - burnin
        store_reg1[ic,:] = y[:,5].T
        store_reg2[ic,:] = y[:,6].T
        store_reg3[ic,:] = y[:,7].T
        store_reg4[ic,:] = y[:,8].T
        store_reg5[ic,:] = y[:,9].T
        store_reg6[ic,:] = y[:,10].T
        store_reg7[ic,:] = y[:,11].T
        store_reg8[ic,:] = y[:,12].T
        store_reg9[ic,:] = y[:,13].T
        store_reg10[ic,:] = y[:,14].T
        store_reg11[ic,:] = y[:,15].T
        store_reg12[ic,:] = y[:,16].T
        store_beta[ic,:] = bet.T
        store_sig2[ic,:] = sig2.T
        
    if np.mod(i,500) == 0:
        mloop = str(i)
        print ( mloop + " Iterations " )

end = time.time()
print(str(end - start) +  " seconds ")

# annualised the UK regions GVA
store_ann1 = np.zeros((nsim,T-6))
store_ann2 = np.zeros((nsim,T-6))
store_ann3 = np.zeros((nsim,T-6))
store_ann4 = np.zeros((nsim,T-6))
store_ann5 = np.zeros((nsim,T-6))
store_ann6 = np.zeros((nsim,T-6))
store_ann7 = np.zeros((nsim,T-6))
store_ann8 = np.zeros((nsim,T-6))
store_ann9 = np.zeros((nsim,T-6))
store_ann10 = np.zeros((nsim,T-6))
store_ann11 = np.zeros((nsim,T-6))
store_ann12 = np.zeros((nsim,T-6))


for tv in range(1,nsim):
    for tg in range(6,T):
        store_ann1[tv,tg-6] = (.25*store_reg1[tv,tg] + .5*store_reg1[tv,tg-1] + .75*store_reg1[tv,tg-2] + store_reg1[tv,tg-3] + .75*store_reg1[tv,tg-4] + .5*store_reg1[tv,tg-5] + .25*store_reg1[tv,tg-6])
        store_ann2[tv,tg-6] = (.25*store_reg2[tv,tg] + .5*store_reg2[tv,tg-1] + .75*store_reg2[tv,tg-2] + store_reg2[tv,tg-3] + .75*store_reg2[tv,tg-4] + .5*store_reg2[tv,tg-5] + .25*store_reg2[tv,tg-6])
        store_ann3[tv,tg-6] = (.25*store_reg3[tv,tg] + .5*store_reg3[tv,tg-1] + .75*store_reg3[tv,tg-2] + store_reg3[tv,tg-3] + .75*store_reg3[tv,tg-4] + .5*store_reg3[tv,tg-5] + .25*store_reg3[tv,tg-6])
        store_ann4[tv,tg-6] = (.25*store_reg4[tv,tg] + .5*store_reg4[tv,tg-1] + .75*store_reg4[tv,tg-2] + store_reg4[tv,tg-3] + .75*store_reg4[tv,tg-4] + .5*store_reg4[tv,tg-5] + .25*store_reg4[tv,tg-6])
        store_ann5[tv,tg-6] = (.25*store_reg5[tv,tg] + .5*store_reg5[tv,tg-1] + .75*store_reg5[tv,tg-2] + store_reg5[tv,tg-3] + .75*store_reg5[tv,tg-4] + .5*store_reg5[tv,tg-5] + .25*store_reg5[tv,tg-6])
        store_ann6[tv,tg-6] = (.25*store_reg6[tv,tg] + .5*store_reg6[tv,tg-1] + .75*store_reg6[tv,tg-2] + store_reg6[tv,tg-3] + .75*store_reg6[tv,tg-4] + .5*store_reg6[tv,tg-5] + .25*store_reg6[tv,tg-6])
        store_ann7[tv,tg-6] = (.25*store_reg7[tv,tg] + .5*store_reg7[tv,tg-1] + .75*store_reg7[tv,tg-2] + store_reg7[tv,tg-3] + .75*store_reg7[tv,tg-4] + .5*store_reg7[tv,tg-5] + .25*store_reg7[tv,tg-6])
        store_ann8[tv,tg-6] = (.25*store_reg8[tv,tg] + .5*store_reg8[tv,tg-1] + .75*store_reg8[tv,tg-2] + store_reg8[tv,tg-3] + .75*store_reg8[tv,tg-4] + .5*store_reg8[tv,tg-5] + .25*store_reg8[tv,tg-6])
        store_ann9[tv,tg-6] = (.25*store_reg9[tv,tg] + .5*store_reg9[tv,tg-1] + .75*store_reg9[tv,tg-2] + store_reg9[tv,tg-3] + .75*store_reg9[tv,tg-4] + .5*store_reg9[tv,tg-5] + .25*store_reg9[tv,tg-6])
        store_ann10[tv,tg-6] = (.25*store_reg10[tv,tg] + .5*store_reg10[tv,tg-1] + .75*store_reg10[tv,tg-2] + store_reg10[tv,tg-3] + .75*store_reg10[tv,tg-4] + .5*store_reg10[tv,tg-5] + .25*store_reg10[tv,tg-6])
        store_ann11[tv,tg-6] = (.25*store_reg11[tv,tg] + .5*store_reg11[tv,tg-1] + .75*store_reg11[tv,tg-2] + store_reg11[tv,tg-3] + .75*store_reg11[tv,tg-4] + .5*store_reg11[tv,tg-5] + .25*store_reg11[tv,tg-6])
        store_ann12[tv,tg-6] = (.25*store_reg12[tv,tg] + .5*store_reg12[tv,tg-1] + .75*store_reg12[tv,tg-2] + store_reg12[tv,tg-3] + .75*store_reg12[tv,tg-4] + .5*store_reg12[tv,tg-5] + .25*store_reg12[tv,tg-6])


post_reg1 = np.mean(store_ann1[1:,:],axis =0).T
post_reg2 = np.mean(store_ann2[1:,:],axis =0).T
post_reg3 = np.mean(store_ann3[1:,:],axis =0).T
post_reg4 = np.mean(store_ann4[1:,:],axis =0).T
post_reg5 = np.mean(store_ann5[1:,:],axis =0).T
post_reg6 = np.mean(store_ann6[1:,:],axis =0).T
post_reg7 = np.mean(store_ann7[1:,:],axis =0).T
post_reg8 = np.mean(store_ann8[1:,:],axis =0).T
post_reg9 = np.mean(store_ann9[1:,:],axis =0).T
post_reg10 = np.mean(store_ann10[1:,:],axis =0).T
post_reg11 = np.mean(store_ann11[1:,:],axis =0).T
post_reg12 = np.mean(store_ann12[1:,:],axis =0).T

post_reg = np.column_stack((post_reg1,post_reg2,post_reg3,post_reg4,post_reg5,post_reg6,post_reg7,post_reg8,post_reg9,post_reg10,post_reg11,post_reg12))

dataoutput = np.column_stack((dataq[13:,0],post_reg))

# Save annualised regional UK GVA into a csv file
savetxt('regionalGVA.csv', dataoutput, delimiter=',')

post_beta = np.mean(store_beta[1:,:],axis =0).T
post_sig2 = np.mean(store_sig2[1:,:],axis =0).T

# Save parameters
savetxt('betaNGVA.csv', post_beta, delimiter=',')
savetxt('sig2NGVA.csv', post_sig2, delimiter=',')

