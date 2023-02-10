using DelimitedFiles

function poolDataLIST(yin,ahat,nVars,polyorder,usesine)
    #n: number of iterations/samples i.e. rows of yin
    n = size(yin,1)
    
    ind = 1
    
    yout = Matrix{String}(undef,nVars+1,1) 
    #poly order 0
    yout[ind,1] = "1"
    ind = ind+1
    
    #poly order 1
    for i=1:nVars
        yout[ind,1] = yin[i]
        ind = ind+1
    end
    
    if(polyorder>=2)
        #poly order 2
        for i=1:nVars
            for j=i:nVars
                yout_temp = [yin[i]*yin[j]]
                yout = reduce(vcat, (yout, yout_temp))
                ind = ind+1
            end
        end
    end
    
    if(polyorder>=3)
        #poly order 3
        for i=1:nVars
            for j=i:nVars
                for k=j:nVars
                    yout_temp = [yin[i]*yin[j]*yin[k]]
                    yout = reduce(vcat, (yout, yout_temp))
                    ind = ind+1
                end
            end
        end
    end
    
    if(polyorder>=4)
        #poly order 4
        for i=1:nVars
            for j=i:nVars
                for k=j:nVars
                    for l=k:nVars
                        yout_temp = [yin[i]*yin[j]*yin[k]*yin[l]]
                        yout = reduce(vcat, (yout, yout_temp))
                        ind = ind+1
                    end
                end
            end
        end
    end
    
    if(polyorder>=5)
        #poly order 5
        for i=1:nVars
            for j=i:nVars
                for k=j:nVars
                    for l=k:nVars
                        for m=l:nVars
                            yout_temp = [yin[i]*yin[j]*yin[k]*yin[l]*yin[m]]
                            yout = reduce(vcat, (yout, yout_temp))
                            ind = ind+1
                        end
                    end
                end
            end
        end
    end
    
    if(usesine)
        for k=1:10
            yout_temp = ["sin("*string(k)*"*yin)"]
            yout = reduce(vcat, (yout, yout_temp))
            ind = ind + 1
            
            yout_temp = ["cos("*string(k)*"*yin)"]
            yout = reduce(vcat, (yout, yout_temp))
            ind = ind + 1
        end
    end

    output = yout

    newout = Matrix{String}(undef, size(ahat,1)+1, length(yin)+1) 
    newout[1,1] = " "

    for k = 1:size(yin,1)
        newout[1,1+k] = string(yin[k])*"dot"
    end

    #we iterate over size of states in sparsified system
    for k = 1:size(ahat,1)
        newout[k+1,1] = output[k]
        for j = 1:length(yin)
            newout[k+1,1+j] = string(ahat[k,j])
        end
    end

    writedlm(stdout, newout)
    return yout
end