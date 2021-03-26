import numpy as np

def path_finder(game_map,start,target,blocked):
    '''
    This function finds teh shortest, unblocked Path to the object at position target.
    Parameters:
        start:      current position of the agent
        target:     position of the opponent / coin
        blocked:    contains values for blocked fields
        game_map:   current game map

    Returns: 
        path:       Coorddinates of the fields contained in the path 
                    None if there is no path to the target
    '''

    q = [start]
    parents = {start:start}
    found = False

    while len(q) > 0:
        #get the current position
        pos = q.pop(0)
        
        #visit all neighbors
        for n in [(pos[0]+1,pos[1]),(pos[0]-1,pos[1]),(pos[0],pos[1]+1),(pos[0],pos[1]-1)]:
            
            #Already visited?
            if (n in parents.keys()): continue

            #Did we reach the target
            if n == target: 
                found = True
                parents[n] = pos
                break

            #Is the neighbor blocked?
            if (game_map[n[0],n[1]] in blocked): continue

            #Add the current neighbor to the queue
            q.append(n)
            parents[n] = pos
        
        if found: break

    if found:
        #reconstruct the path
        path = []
        while n != parents[n]:
            path.append(n)
            n = parents[n]
        return np.flip(np.array(path))
    else:
        return None

def danger_close(pos,times,bombs,explosion_map):
    #This function determines, if the given pos is a deadly place

    #If there is no explosion there can be bombs harming the spot if they explode in the next step
    if  (0 in times) or (1 in times) or (2 in times) or (3 in times) or (4 in times):
        
        mask = (times == 0)+(times == 1)+(times == 2)+(times == 3)+(times == 4)

        #Is the position reachable for the bombs exploding in the next step
        for bomb in np.array(bombs)[mask]:
            if bomb[0] == pos[0] and pos[0] % 2 != 0 and abs(pos[0] - bomb[0]) < 4:
                return 1
            elif bomb[1] == pos[1] and pos[1] % 2 != 0 and abs(pos[1] - bomb[1]) < 4: 
                return 1

    return 0

def features_V2(gam_state):
    #######################################################################################################################
    #Own agent
    #######################################################################################################################
    own_position = np.array(list(gam_state["self"][3]))
    own_bombing = int(gam_state["self"][2])

    ######################################################################################################################
    #Bombs and other agents block the path too
    ######################################################################################################################
    
    
    for bomb in gam_state["bombs"]:
        gam_state["field"][bomb[0][0]][bomb[0][1]] = 1

    for other in gam_state["others"]:
        gam_state["field"][other[3][0]][other[3][1]] = 1
    

    #######################################################################################################################
    #coins 
    #######################################################################################################################
    coin_pos = np.array([list(coin) for coin in gam_state["coins"]])

    dist_coins = []
    best_field_coins = []
    
    for i in range(len(coin_pos)):
        path = path_finder(gam_state["field"],(own_position[0],own_position[1]),(coin_pos[i][0],coin_pos[i][1]),[-1,1])
        #Not reachable
        if path is None:
            dist_coins.append(np.inf)
            best_field_coins.append([-1,-1])

        #Is reachable
        else:
            dist_coins.append(len(path))
            best_field_coins.append([path[0][1],path[0][0]])

    #######################################################################################################################
    #bombs 
    #######################################################################################################################
    #Which of the reachable fields is dangerous?
    deadly = []
    times = np.array([bomb[1] for bomb in gam_state["bombs"]])
    bomb_position = [list(bomb[0]) for bomb in gam_state["bombs"]]

    deadly += [danger_close(own_position,times,bomb_position,gam_state["explosion_map"])]

    neighbors = [
        [own_position[0]+1,own_position[1]],
        [own_position[0]-1,own_position[1]],
        [own_position[0],own_position[1]+1],
        [own_position[0],own_position[1]-1],
    ]

    pos_dict = {
        (own_position[0]+1,own_position[1]): 1,
        (own_position[0]-1,own_position[1]): 2,
        (own_position[0],own_position[1]+1): 3,
        (own_position[0],own_position[1]-1): 4,
    }

    for n in neighbors:
        deadly.append(danger_close(n,times,bomb_position,gam_state["explosion_map"]))

    
    #-----------------------------------------------------------------------------------------------------
    #Handle the case, that all reachable fields are marked as deadly
    reachable = [True] #The current field is always reachable
    for n in neighbors:
        if gam_state["field"][n[0]][n[1]] != 0: reachable.append(False)
        else: reachable.append(True)

    if np.array(reachable).sum() == np.array(deadly)[reachable].sum():
        #Get the closest 25 fields
        mask = np.zeros((17,17)).astype(np.bool)
        mask[max(0,own_position[0]-2):min(16,own_position[0]+3),max(0,own_position[1]-2):min(16,own_position[1]+3)] = (gam_state["field"][max(0,own_position[0]-2):min(16,own_position[0]+3),max(0,own_position[1]-2):min(16,own_position[1]+3)] == 0)

        grid = np.indices((17,17))
        
        positions = np.concatenate((grid[0].reshape(17,17,1),grid[1].reshape(17,17,1)),axis=2)
        positions = np.array(positions[mask].reshape(-1,2))
        
        #Sort by distance
        dist_free = np.abs(positions[:,0] - own_position[0]) + np.abs(positions[:,1] - own_position[1])
        indices = np.argsort(dist_free)

        positions = positions[indices]

        #Which of them is save?
        mask = []
        for p in positions:
            mask.append(bool(danger_close(p,times,bomb_position,gam_state["explosion_map"])))

        positions = positions[mask]
        
        #Which path to go?
        dists = []
        best_step = []
        
        for p in positions:
            path = path_finder(gam_state["field"],(own_position[0],own_position[1]),(p[0],p[1]),[-1,1])
            if path is None: continue
            dists.append(len(path))
            best_step.append((path[0][1],path[0][0]))

            if len(path) < 3: break

        if len(dists) > 0:
            i = np.argmin(dists)
            q = pos_dict[best_step[i]]

            deadly[q] = 0

    #######################################################################################################################
    #Other players
    #######################################################################################################################
    other_pos = np.array([list(other[3]) for other in gam_state["others"]])

    other_pos = other_pos[:min(3,len(other_pos))]

    dist_others = []
    best_field_others = []
    bombable = []
    

    for i in range(len(other_pos)):
        
        path = path_finder(gam_state["field"],(own_position[0],own_position[1]),(other_pos[i][0],other_pos[i][1]),[-1,1])
        

        #Not reachable
        if path is None:
            dist_others.append(np.inf)
            best_field_others.append([-1,-1])
            bombable.append(0)

        #Is reachable
        else:
            dist_others.append(len(path))
            best_field_others.append([path[0][1],path[0][0]])

            if other_pos[i][0] == own_position[0] and other_pos[i][0] % 2 != 0 and abs(own_position[0] - other_pos[i][0]) < 4: bombable.append(1)
            elif other_pos[i][1] == own_position[1] and other_pos[i][1] % 2 != 0 and abs(own_position[1] - other_pos[i][1]) < 4: bombable.append(1)
            else: bombable.append(0)

    #######################################################################################################################
    #Crates
    #######################################################################################################################
    crates = []
    indices = np.indices((17,17))
    coords = np.concatenate((indices[0].reshape(17,17,1),indices[1].reshape(17,17,1)),axis = 2)
    mask = (gam_state["field"] == 1).reshape(17*17)
    coords = coords.reshape(17*17,2)
    pos_crates = coords[mask]

    if mask.sum() != 0:
        dist_crates = np.sqrt((pos_crates[:,0] - own_position[0])**2 + (pos_crates[:,1] - own_position[1])**2)
        indices = np.argsort(dist_crates)

        pos_crates = pos_crates[indices]

        #check reachability for the closest eight crates
        crates = []

        for i in range(min(len(pos_crates),16)):
            feature_map = gam_state["field"]
            feature_map[pos_crates[i][0]][pos_crates[i][1]] = 0

            path = path_finder(feature_map,(own_position[0],own_position[1]),(pos_crates[i][0],pos_crates[i][1]),[-1,1])
            feature_map[pos_crates[i][0]][pos_crates[i][1]] = 1
            if path is not None:
                crates.append(len(path))
                crates.append(pos_dict[(path[0][1],path[0][0])])

            if len(crates) == 8: 
                break

        #Fill missing entries
        for i in range(4- int(len(crates) / 2)):
            #erreichbar
            crates += [0] #Go nowhere
            #distance
            crates += [np.inf]#Infinitly far away

    #No crates
    else:
        for i in range(4):
            #erreichbar
            crates += [0] #Go nowhere
            #distance
            crates += [np.inf]#Infinitly far away
    

    #######################################################################################################################
    #Putting things together
    #######################################################################################################################
    feature_vector = []

    #The four clost coins
    if len(coin_pos) > 0:
        mask = (np.array(dist_coins) < np.inf)
        
        if sum(mask) > 0:
            dist_coins = np.array(dist_coins)[mask]
            best_field_coins = np.array(best_field_coins)[mask]

            indices = np.argsort(dist_coins)

            dist_coins = dist_coins[indices]
            best_field_coins = best_field_coins[indices]

            for i in range(min(4,len(indices))):
                feature_vector += [pos_dict[(best_field_coins[i][0],best_field_coins[i][1])]]
                feature_vector += [dist_coins[i]]

    #Handle missing coins
    for i in range(4 - int(len(feature_vector)/2)):
        feature_vector += [0] #Go nowhere
        feature_vector += [np.inf]#Infinitly far away

    #Danger of the fields reachable in the next step
    feature_vector += deadly

    #Own position and bombing ability
    feature_vector += [own_position[0],own_position[1]]
    feature_vector += [own_bombing]

    #other players
    other_players = []

    if len(other_pos) > 0:
        mask = (np.array(dist_others) < np.inf)

        if sum(mask) > 0:
            dist_others = np.array(dist_others)[mask]
            best_field_others = np.array(best_field_others)[mask]
            bombable = np.array(bombable)[mask]

            for i in range(len(dist_others)):
                #rel position
                other_players += [other_pos[i][0] - own_position[0],other_pos[i][1] - own_position[1]]
                #Best field
                other_players += [pos_dict[(best_field_others[i][0],best_field_others[i][1])]]
                #distance
                other_players += [dist_others[i]]
                #can be killed
                other_players += [bombable[i]]

    for i in range(int((15 - len(other_players))/5)):
        #rel pos
        other_players += [np.inf,np.inf]
        #erreichbar
        other_players += [0]
        #Distance
        other_players += [np.inf]
        #In Bombenreichweite
        other_players += [0]

    feature_vector += other_players

    #crates
    feature_vector += crates

    return np.array(feature_vector)
            

    


