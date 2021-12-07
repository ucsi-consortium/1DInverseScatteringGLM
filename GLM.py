# +
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import scipy as sp
import scipy.sparse as sps
from scipy import interpolate
from scipy.sparse.linalg import lsqr
from scipy import fft,ifft
from scipy.optimize import minimize_scalar, minimize

#The following wave solver is used in Example 1 (PLASMA WAVE EQUATION)
def SolveWE(q,f,T,L,h):
    """
    Solve plasma wave-equation 
    
        u_tt(t,x) = u_xx(t,x) - q(x) u(t,x) + f(t,x) on [0,T] x [0,L]
    
    Uses second-order FD scheme with absorbing boundary conditions
    
    input:
        q - scattering potential q(x) (callable)
        f - source function f(t,x) (callable)
        T,L - domain (floats)
        h - gridsize (float)
        
    output:
        t - temporal grid (1d array)
        x - spatial grid (1d array)
        u - solution (2d array)
    """
    
    Nt = int(round(T/h))
    t = np.linspace(0, Nt*h, Nt+1)   
    Nx = int(round(L/h))
    x = np.linspace(0, Nx*h, Nx+1) 
    
    u  = np.zeros((Nt+1,Nx+1)) 
    
    for n in range(1,Nt):
        u[n+1,0] = ( u[n,0] + (u[n,1] - u[n,0]) )
        u[n+1,1:-1] = -u[n-1,1:-1] + 2*u[n,1:-1] + (u[n,2:] - 2*u[n,1:-1] + u[n,0:-2]) + h**2*(-q(x[1:-1])*u[n,1:-1] + f(t[n],x[1:-1]))
        u[n+1,Nx] = ( u[n,Nx] + (u[n,Nx-1] - u[n,Nx]) )
        
    return t,x,u

# The following is used in Example 2 (ACOUSTIC WAVE EQUATION WITH VARIABLE DENSITY)
def SolveWE2(rho,f,T,L,dx):
    
    Le = 2.1*T
    
    nx = int(2*Le/dx - 1)
    dt = 0.8*dx
    nt = int(T/dt)
    
    x = np.linspace(-Le+dx,Le-dx,nx)
    xm = np.linspace(-Le+dx/2,Le-dx/2,nx+1)
    t = np.linspace(0,T,nt)
    
    mu = rho(xm)
    A = sps.diags(1/rho(x),0)
    D = sps.diags([-np.ones(nx), np.ones(nx)],[-1,0],shape=(nx+1,nx))/dx
    M = sps.diags([mu],[0])
    I = sps.identity(nx)
    B = -D.T@M@D
    
    u = np.zeros((nt,nx))
    for k in range(2,nt-1):
        u[k+1] = (2*I + (dt**2)*A@B)@u[k] - u[k-1] + (dt**2)*A@f((k+1)*dt,x)
        
    a = nx//2
    b = np.where(x>L)[0][0]
    return t,x[a:b],u[:,a:b]

def H(x,f0):
    # smooth Heaviside
    return 0.5 + 0.5*sp.special.erf(np.pi*f0*x)

def delta(x,f0):
    # smooth approximation of \delta(x)
    return f0*np.sqrt(np.pi)*np.exp(-(np.pi*f0*x)**2)

def deltap(x,f0):
    # smooth approximation of \delta'(x)
    epsilon = 0.5/(np.pi*f0)**2
    return f0*np.sqrt(np.pi)*(-2*x*(np.pi*f0)**2)*np.exp(-(np.pi*f0*x)**2)

def getScatteringData(q,T,L,h,x0,t0,f0):
    """
    Generate scattering data by solving the wave-equation with source term
        
        f(t,x) = \delta(x-x0)\delta'(t-t0)
    
    input:
        q,T,h - paramters for SolveWE
        x0,t0,f0 - source parameters
        
    output:
        x - spatial grid (1d array)
        t - temporal grid (1d array)
        r - scattering data (1d array)
    """
    # source
    f = lambda t,x : delta(x-x0,f0)*deltap(t-t0,f0)
    
    # solve wave-equation
    t,x,u = SolveWE(q,f,T,L,h)
    t,x,u0 = SolveWE(lambda x : 0*x,f,T,L,h)
    
    # interpolate scattering data
    r = interpolate.interp1d(x,u - u0,axis=1)(x0)
    
    return x,t,r

def getScatteringData2(rho,T,L,h,x0,t0,f0):
    """
    Generate scattering data by solving the wave-equation with source term
        
        f(t,x) = \delta(x-x0)\delta'(t-t0)
    
    input:
        rho,T,h - paramters for SolveWE2
        x0,t0,f0 - source parameters
        
    output:
        x - spatial grid (1d array)
        t - temporal grid (1d array)
        r - scattering data (1d array)
    """
    # source
    f = lambda t,x : delta(x-x0,f0)*deltap(t-t0,f0)
    
    # solve wave-equation
    t,x,u = SolveWE2(rho,f,T,L,h)
    t,x,u0 = SolveWE2(lambda x : 0*x + 1,f,T,L,h)
    
    # interpolate scattering data
    r = interpolate.interp1d(x,u - u0,axis=1)(x0)
    
    return x,t,r,u,u0

def Solve_GLM_LS(r,t,alpha=1e-16):
    """
    Solve the GLM equation: 
        
        r(x + y) + \int_0^T B(x,z)r(x + y + z)dz + B(x,y) = 0
    
    for B(x,z) by posing it as a regularised least-squares problem
    
        min_b \|A_x b + r_x\|_2^2 + \alpha \|L b\|_2^2,
        
    with L the second derivative operator.
    
    input:
        r - scattering data (1d array)
        t - temporal grid at which r is given (1d array)
        alpha - regularization parameter (float), default=1e-16
        method - {'tsvd','lsqr'}, default='lsqr'
    
    output:
        B - solution (2d array)
        t - temporal grid at which B is defined (1d array)
        phi - value of objective
    """
    # grid
    nt = len(t)
    dt = t[2]-t[1]
    mt = int(nt/3)
    
    # Regularisation operator
    L = sps.spdiags(np.outer(np.array([1,-2,1]),np.ones(mt)),np.array([-1,0,1]),mt,mt).toarray()
    
    # initialize
    B = np.zeros((mt,mt))
    phi = 0
    
    # solve for every column
    for j in range(mt):
        # generate operator (Hankel matrix)
        I = np.identity(mt)
        R = la.hankel(r[j:j+mt],r[j+mt-1:j+2*mt-1])
        A = I + dt*R
        B[:,j] = lsqr(np.concatenate((A,np.sqrt(alpha)*L)),np.concatenate((-r[j:j+mt],np.zeros(mt))))[0]
        phi += np.linalg.norm(B[:,j] + dt*R@B[:,j] + r[j:j+mt])**2 + alpha*np.linalg.norm(L@B[:,j])**2
    return B,t[:mt],phi

def Solve_GLM_TLS(r,t,alpha=[1e-16,1e-3],maxit=(10,10)):
    """
    Solve the GLM equation: 
        
        (r + e)(x + y) + \int_0^T B(x,z)(r + e)(x + y + z)dz + B(x,y) = 0.
    
    for e and B in a Total Least-Squares sense using altermating minimisation.
    
    input:
        r - scattering data (1d array)
        t - temporal grid at which r is given (1d array)
        alpha - regularization parameters for \|B\|^2 and \|e\|^2  (float, float), default=(1e-16,1e-3)
        maxit - maximum (inner,outer) iterations (int,int), default=(10,10)
    
    output:
        B - estimated kernel (2d array)
        e - estimated error
        t - temporal grid at which B is defined (1d array)
        hist - objective at each iteration (1d array)
    """
    
    # grid
    nt = len(t)
    dt = t[2]-t[1]
    mt = int(nt/3)
    
    # initialize
    B = np.zeros((mt,mt))
    e = np.zeros(nt)
    hist = np.zeros(maxit[1])
    
    # Regularisation operator
    L = sps.spdiags(np.outer(np.array([1,-2,1]),np.ones(mt)),np.array([-1,0,1]),mt,mt).toarray()
    
    # main loop
    for it in range(maxit[1]):
        # estimate kernel
        B,ts,phi= Solve_GLM_LS(r + e,t,alpha[0])
        
        # estimate error
        
        ## first, setup system of m^2 x n equations for e
        A = np.zeros((mt**2,nt))
        b = np.zeros(mt**2)
        k = 0
        for i in range(mt):
            for j in range(mt):
                A[k,i+j] = 1
                A[k,i+j:i+j+mt] += dt*B[:,i]
                b[k] = -(r[i + j] + dt*B[:,i].dot(r[i+j:i+j+mt]) + B[j,i])             
                k += 1
        ## now solve it
        e = lsqr(A,b,damp=np.sqrt(alpha[1]),iter_lim=maxit[0])[0]
        
        ## objective
        hist[it] = phi + alpha[1]*np.linalg.norm(e)**2
    return B,e,t[:mt],hist

def reconstruct(B,ts,t0,x0):
    """
    Reconstruct scattering potential from Kernel
    """
    ns = len(ts)-2
    xs = ts[1:-1]/2-(t0/2-x0)
    h = (xs[1] - xs[0])
    
    q_hat = 2*(B[0,2:] - B[0,:-2])/(h)
    return q_hat, xs

def get_parameters(r_delta,B0,t,ts,t0,x0):  
    """
    Get regularisation parameters for LS and TLS by minimising the error w.r.t. a reference solution.
    """   
    # optimal parameter LS
    def Jerror_LS(alpha):
        B_LS,tsz,phi = Solve_GLM_LS(r_delta,t,alpha=alpha)
        return np.linalg.norm(B_LS - B0,ord='fro')/np.linalg.norm(B0,ord='fro')

    result_LS = minimize_scalar(Jerror_LS, bounds=(1e-16,1e2), method='bounded')
    alpha_LS = result_LS.x
    
    # optimal parameter TLS
    def Jerror_TLS(alpha):
        B_TLS,ts,ek,res = Solve_GLM_TLS(r_delta,t,alpha=(alpha,1e-16))
        return np.linalg.norm(B_TLS - B0,ord='fro')/np.linalg.norm(B0,ord='fro')
    
    result_TLS = minimize_scalar(Jerror_TLS, bounds=(1e-16,1e2), method='bounded')
    alpha_TLS = result_TLS.x
    
    return alpha_LS, alpha_TLS
    
# -


