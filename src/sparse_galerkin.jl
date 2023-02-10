
function sparse_galerkin!(dy,y,p,t)
    yPool = evaluate(y, p.basis)
    return_mat_size = size(yPool * p.Ξ, 2)
    for index = 1:return_mat_size
        dy[index] = (yPool * p.Ξ)[1, index]
    end
end
