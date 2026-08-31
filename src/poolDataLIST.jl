using DelimitedFiles

function poolDataLIST(yin, ahat, nVars, polyorder, usesine)
    #n: number of iterations/samples i.e. rows of yin
    n = size(yin, 1)

    ind = 1

    yout = Matrix{String}(undef, nVars+1, 1)
    #poly order 0
    yout[ind, 1] = "1"
    ind = ind+1

    #poly order 1
    for i in 1:nVars
        yout[ind, 1] = yin[i]
        ind = ind+1
    end

    if (polyorder>=2)
        #poly order 2
        for i in 1:nVars
            for j in i:nVars
                yout_temp = [yin[i]*yin[j]]
                yout = reduce(vcat, (yout, yout_temp))
                ind = ind+1
            end
        end
    end

    if (polyorder>=3)
        #poly order 3
        for i in 1:nVars
            for j in i:nVars
                for k in j:nVars
                    yout_temp = [yin[i]*yin[j]*yin[k]]
                    yout = reduce(vcat, (yout, yout_temp))
                    ind = ind+1
                end
            end
        end
    end

    if (polyorder>=4)
        #poly order 4
        for i in 1:nVars
            for j in i:nVars
                for k in j:nVars
                    for l in k:nVars
                        yout_temp = [yin[i]*yin[j]*yin[k]*yin[l]]
                        yout = reduce(vcat, (yout, yout_temp))
                        ind = ind+1
                    end
                end
            end
        end
    end

    if (polyorder>=5)
        #poly order 5
        for i in 1:nVars
            for j in i:nVars
                for k in j:nVars
                    for l in k:nVars
                        for m in l:nVars
                            yout_temp = [yin[i]*yin[j]*yin[k]*yin[l]*yin[m]]
                            yout = reduce(vcat, (yout, yout_temp))
                            ind = ind+1
                        end
                    end
                end
            end
        end
    end

    if (usesine)
        for k in 1:10
            yout_temp = ["sin("*string(k)*"*yin)"]
            yout = reduce(vcat, (yout, yout_temp))
            ind = ind + 1

            yout_temp = ["cos("*string(k)*"*yin)"]
            yout = reduce(vcat, (yout, yout_temp))
            ind = ind + 1
        end
    end

    output = yout

    newout = Matrix{String}(undef, size(ahat, 1)+1, length(yin)+1)
    newout[1, 1] = " "

    for k in 1:size(yin, 1)
        newout[1, 1 + k] = string(yin[k])*"dot"
    end

    #we iterate over size of states in sparsified system
    for k in 1:size(ahat, 1)
        newout[k + 1, 1] = output[k]
        for j in 1:length(yin)
            newout[k + 1, 1 + j] = string(ahat[k, j])
        end
    end

    writedlm(stdout, newout)
    return yout
end
