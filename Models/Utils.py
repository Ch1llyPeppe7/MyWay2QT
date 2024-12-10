import torch
import cvxpy as cp
def cp_minimize_sigma(CovMatrix,u,ustar):
    w=cp.Variable(u.shape[0])

    objective=cp.Minimize(cp.quad_form(w,CovMatrix))

    constrains=[cp.sum(w)==1,w>=0,w@u>=ustar]

    problem=cp.Problem(objective,constrains)
    problem.solve()
    if problem.status not in ["infeasible", "unbounded"]:
        return w.value,cp.quad_form(w,CovMatrix).value
    else:
        raise ValueError("minimize Optimization did not succeed. Status: " + problem.status)
def cp_maximize_u(CovMatrix,u,sig):
    w=cp.Variable(u.shape[0])

    objective=cp.Maximize(w@u)

    constrains=[cp.sum(w)==1,w>=0,cp.quad_form(w,CovMatrix)<=sig]

    problem=cp.Problem(objective,constrains)
    problem.solve()
    if problem.status not in ["infeasible", "unbounded"]:
        return w.value,cp.quad_form(w,CovMatrix).value
    else:
        raise ValueError("maximize Optimization did not succeed. Status: " + problem.status)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def SelectStks(df,infostar=0.05):
    m=torch.tensor(df.fillna(df.mean()).to_numpy())
    m[torch.isnan(m)]=-1
    m=m.to(device)
    Yield=torch.log(m[1:,:]/m[:-1,:])*100#(%)
    var=torch.var(Yield,dim=0,unbiased=True)

    U=Yield.mean(0)

    info=U/(torch.sqrt(var)+(var==0).float())

    _,indices=torch.topk(info,30)
    filtered_idxs=indices[info[indices]>infostar]

    return filtered_idxs.cpu(),Yield.cpu()

def Estimate(Yield,idxs):
    cov=torch.cov(Yield[:,idxs].to(device).T)
    u=Yield[:,idxs].mean(0)
    return cov.cpu(),u.cpu()
