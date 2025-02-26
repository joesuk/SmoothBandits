#=  =## Algorithms for Regret Minimization in two-armed non-stationary bandit problems
# julia 1.0.5

# faster implementation of META for one-armed bandit problem given tape 'prob' of gaps of arm 2 to arm 1 where arm 1 has constant reward
function METAone(prob,amp,params=[1,1,1],verbose="off")
    
    # set parameters
    C=params[1] # eviction threshold
    P=params[2] # replay probability multiplier
    maxp=params[3] # max replay probability
    
    # prob contains a table of rewards for arm 2 up to horizon T
    T = length(prob)

    # record indicators of replay
    repTime = 0
    
    # index of the episode
    episode = 1
    ChosenArms = zeros(Int,T)
    ChangePoints = []
    t = 0 # round timer
    
    # replay setup
    M = trunc(Int,round(log2(T)))
    outcomes = zeros(M,T)
    lengths = zeros(Int,M)
    
    # set eviction thresholds
    thresholds = zeros(T)
    for s in 1:T
        thresholds[s] = C*(ℯ-1)*(sqrt((4*log2(T))*(s))+(4*log2(T)))
    end
    
    # total reward
    ReceivedRewards = zeros(T)
    
    while (t < T)
        epStart = t+1
        t=t+1

        # set up master arm set
        Master_Arms = collect(1:2) # master arm set

        # set up replay schedule for this episode
        for i in 1:M # length of replay
            lengths[i] = 2^i
            for j in 1:T # start time of replay
                outcomes[i,j] = rand(Binomial(1,min(maxp,(P*1/sqrt((lengths[i])*(j))))))
            end
        end
        startTimes = findall(!iszero, sum(outcomes,dims=1)) # get rounds with new replays

        # get first elimination round of master and update till then
        elimArm, elimRound = updateData(epStart,Master_Arms,prob,thresholds,amp,ChosenArms,ReceivedRewards,t,T)

         if (verbose=="on")
            println("first elim round $(elimRound): elim arm $(elimArm)")
        end

        # update current round
        t = elimRound
        
        # if no replays, end
        if isempty(startTimes)
            t = T
        end
        
        # run the replays
        newEpisode = false # should we start a new episode?
        while (!newEpisode)&&(t < T)&&(!isempty(startTimes))&&(t > epStart)
            # get first next replay round
            nextReplay = T
            for i=1:length(startTimes)
                if startTimes[i][2] > t
                    nextReplay = startTimes[i][2]
                    break
                end
            end
            t = nextReplay
            replayLength = lengths[argmax(outcomes[:,nextReplay])] # get length of longest replay scheduled on nextReplay
            elimArm, elimRound = updateData(epStart,Master_Arms,prob,thresholds,amp,ChosenArms,ReceivedRewards,t,min(t+replayLength,T))

            # update replay indicators
            repTime += elimRound - t + 1

            # update current round
            t = elimRound
            
            if (verbose=="on")&&(elimArm!=0)&&(elimArm in Master_Arms)
                println("replay elim round $(elimRound): elim arm $(elimArm)")
                display(Master_Arms)
            end
            
            # perform restart tests
            if (length(Master_Arms)==0)
                if (verbose=="on")
                    println("new ep")
                end
                episode+=1
                ChangePoints=append!(ChangePoints,t)
                newEpisode = true
                break
            end
        end
    end
    return ChosenArms,ChangePoints,ReceivedRewards,repTime
end

# skip ahead and update bandit data for two-armed problem
function updateData(epStart,Master_Arms,prob,thresholds,amp,ChosenArms,ReceivedRewards,t,next)
    duration = next - t + 1
    draws = rand([1,2],duration)
    ChosenArms[t:next] = draws
    ReceivedRewards[t:next] = (draws .- 1).*(rand.(Normal.(prob[t:next],0.01)))
    gaps21 = (draws .- 1) .* (amp .+ ReceivedRewards[t:next]) - amp*(2 .- draws)
    elimRound = next+1
    elimArm = 0
    sum21 = 0
    sum12 = 0
    for i=1:duration
        # set up dyadic indices
        inds = collect(0:trunc(Int,floor(log2(i))))
        inds = 2 .^ inds
        back_inds = i .- inds .+ 1
        temp = cumsum(reverse(gaps21[1:i]))
        if any(x -> x > 0, temp[inds] .- thresholds[inds])
            elimRound = t+i-1
            elimArm = 1
            deleteat!(Master_Arms, findall(x->x==1, Master_Arms))
            ChosenArms[t+i-1:next] .= 2
            ReceivedRewards[t+i-1:next] = rand.(Normal.(prob[t+i-1:next],0.01))
            break
        elseif any(x -> x > 0, -1 .* temp[inds] .- thresholds[inds])
            elimRound = t+i-1
            elimArm = 2
            deleteat!(Master_Arms, findall(x->x==2, Master_Arms))
            ChosenArms[t+i-1:next] .= 1
            ReceivedRewards[t+i-1:next] .= 0
            break
        end
    end
    return elimArm, elimRound
end

# baseline which plays a random arm
function RANDone(prob)
   T = length(prob)
   ChosenArms = rand(collect(1:2),T)
    noise = rand(Normal(0,0.01),T)
    ReceivedRewards = (prob .+ noise) .* (ChosenArms .- 1)
   return ChosenArms,ReceivedRewards
end

# budgeted exploration
function BE(T, params, prob)
    # Budgeted Exploration Algorithm 
    # input: time horizon T, exploration budget, epoch size, reward prob
    # output: algorithm rewards
    budget = params[1]
    delta = params[2]
    rew = zeros(T)
    ChosenArms = zeros(T)
    t = 0
    
    while t < T
        # Get the epoch slice
        epoch_tape = rand.(Normal.(prob[(t+1):min(t+delta, T)],0.01))

        # Cumulative sum of rewards in the epoch
        cum_reward = cumsum(epoch_tape)
        
        # Find where cumulative reward drops below -budget
        index = findfirst(x -> x <= -budget, cum_reward)

        t=t+1        
        # Calculate epoch reward
        if isnothing(index)  # Budget not run out
            epoch_reward = cum_reward[end]
            ChosenArms[t:min(t+delta,T)] .= 2
        else  # Budget run out
            epoch_reward = cum_reward[index]
            ChosenArms[t:min(t+delta,T)] .= 2
            ChosenArms[(index+1):min(t+delta,T)] .= 1
            # println("budget ran out")
        end
        
        # Move to next epoch
        t += delta
    end
    return ChosenArms
end


###########################################################################################
# work-in-progress code for non-stationary K-armed bandits
###########################################################################################


# META for multiple arms
function META(Table,params=[1,1,1],verbose="off")
    
    # set parameters
    C=params[1] # eviction threshold
    P=params[3] # replay probability multiplier
    maxp=params[4] # max replay probability
    
    # table contains a table of rewards for the K arms up to horizon T
    (K,T)=size(Table)
    
    # index of the episode
    episode = 1
    ChosenArms = zeros(Int,T)
    ChangePoints = []
    t = 0 # round timer
    
    # replay setup
    M = trunc(Int,round(log2(T)))
    outcomes = zeros(Bool,M,T)
    lengths = zeros(Int,M)
    
    # set eviction thresholds
    thresholds = zeros(T)
    for s in 1:T
        thresholds[s] = C*(ℯ-1)*(sqrt(K*(4*log2(T)+log2(K))*(s))+K*(4*log2(T)+log2(K)))
    end

    # storage of aggregate gaps and base algorithms
    SUMS = zeros(K,T) # array containing the S[a,s,t_current]
    Bases = Stack{Vector}() # stack of base algorithms
    
    while (t < T)

        # set up replay schedule for this episode
        for i in 1:M # length of replay
            lengths[i] = 2^i
            for j in 1:T # start time of replay
                outcomes[i,j] = rand(Binomial(1,min(maxp,(P*1/sqrt((lengths[i])*(j))))))
            end
        end
        
        # initialize master set of arms for episode
        Master_Arms = collect(1:K) # master arm set
        push!(Bases,[(t+1),(t+1),(T+1),collect(1:K)]) # push first base alg
        newEpisode = false # should we start a new episode?
        epStart = t+1
        
        # start the episode
        while (!newEpisode)&&(t < T)
            t=t+1
            # form the set of candidate arms
            recent = first(Bases)
            recent_start = recent[1]
            recent_last = recent[2]
            recent_end = recent[3]
            recent_arms = recent[4]
            Active = copy(recent_arms)
            
            # draw a random arm
            I = rand(Active)
            rew = Table[I,t]
            ChosenArms[t] = I
            
            # update gap estimates
            SUMS[I,epStart:t] = SUMS[I,epStart:t] .+ (rew)*length(Active)
            
            # update current active arm set
            if (t>maximum([epStart,recent_start]))

                # set grid of round indices over which to do elimination
                inds = collect(recent_start:t)
                back_inds = collect(1:(t-recent_start+1))

                # update current base-alg active set
                if (rew>0) && (I in Active) && (sum(SUMS[I,inds] .- thresholds[back_inds]) > 0)
                    # remove arm from active set
                    deleteat!(Active, findfirst(isequal(I), Active))
                
                    # remove from master set
                    if I in Master_Arms
                        deleteat!(Master_Arms, findfirst(isequal(I), Master_Arms))
                    end
                end

                # update master arm set over data outside current active alg
                if (rew>0) && (recent_start > epStart) && (I in Master_Arms)
                    # set grid of round indices over which to do elimination
                    inds_e = collect(epStart:(recent_start-1))
                    back_inds_e =  collect((t-recent_start+2):(t-epStart+1))
                    
                    if (sum(SUMS[I,inds_e].-thresholds[back_inds_e]) > 0)
                        # remove arm from master
                        deleteat!(Master_Arms, findfirst(isequal(I), Master_Arms))
                    end
                end
            end

            # perform restart tests
            if (length(Master_Arms)==0)
                episode+=1
                ChangePoints=append!(ChangePoints,t)
                newEpisode = true
                break
            end
            # clean base algorithm stack
            cleanBases(Bases,Active,Master_Arms,t,epStart,outcomes,M,lengths,K,"META")
        end
    end
    return ChosenArms,ChangePoints 
end

# clean stack of base algorithms in meta-algorithm
function cleanBases(Bases,Active,Master_Arms,t,epStart,outcomes,M,lengths,K,which)
    recent = first(Bases)
    # update last active round of current base alg
    recent[2] = t
    recent_end = recent[3]
    recent_arms = recent[4]
    
    # clean and update base algs (in Bases stack)
    clean=true
    prune_arms = copy(Active)
    if (length(Bases)>1) 
        while clean
            recent = first(Bases)
            recent_start = recent[1]
            recent_end = recent[3]
            recent_arms = recent[4]
            prune_arms = intersect(prune_arms,recent_arms)
            if (length(setdiff(Master_Arms,recent_arms))>0)
                println("ERROR: master arm set not subset of recent armset")
            end
            if (t >= recent_end) || ((length(recent_arms)==length(Master_Arms)) & (recent_start != epStart))
                pop!(Bases)
            else
                clean=false
            end
        end
    end

    recent = first(Bases)
    recent[4] = intersect(recent[4],prune_arms)

    # randomly add some new replay
    if (t>epStart)&&(length(Active)<K)
        approxtime=t-epStart
        min_ind = 2^(trunc(Int,ceil(log2(2))))
        if which=="ANACONDA"
            min_ind = 2^(trunc(Int,ceil(log2(2))))
        end
        inds = [i for i in min_ind:M if outcomes[i,approxtime] > 0]
        append!(inds,0)
        if (maximum(inds) > 0)
            m = maximum(inds)
            push!(Bases,[(t),(t),(t+1+lengths[m]),collect(1:K)])
        end
    end
end