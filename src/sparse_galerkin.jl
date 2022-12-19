
function sparse_galerkin(y,p,t)
    yPool = evaluate(y, p.basis)
    return yPool * p.Ξ
end
