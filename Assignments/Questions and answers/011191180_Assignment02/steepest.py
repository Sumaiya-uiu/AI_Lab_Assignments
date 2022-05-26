def init():
    l=[2, 1, 5, 0, 8, 4, 10, 0, 20, 10]
    return l
def calc_cost(l):
    la = l
    cost = 0
    for i in range(len(la)):
        l1=l[(i+1) : len(l)]
        ls = [j for j in l1 if la[i]>j ]
        cost = cost + len(ls)
    return cost


def state_generation(current_state, current_state_cost):
    min_cost = current_state_cost
    sli_state = current_state.copy()
    for i in range(len(current_state)-1):
        for j in range(i+1,len(current_state)):
            sli_state[i],sli_state[j] = sli_state[j],sli_state[i]
            eitar_cost = calc_cost(sli_state)
            if(eitar_cost < min_cost):
                min_cost = eitar_cost
                min_state = sli_state.copy()
            sli_state = current_state.copy()

    if(min_cost < current_state_cost):
        return min_state,min_cost
    else:
        return current_state,None

def goal_test(state):
    if calc_cost(state) == 0 :
        return True
    else:
        return False





def main():
    # state = [int(s) for s in input().split()]
    state=init()
    cost = calc_cost(state)
    while(not goal_test(state)):
        state, cost = state_generation(state, cost)
        if cost is None:
            print(state)
            return
    print(state)
    return

if __name__ == '__main__':
    main()