import numpy as np                                                                                                                        
from numba import njit                                                                                                                    
                                                                                                                                          
@njit(cache=True)                                                                                                                         
def _recycle_and_draw(hands, draw_piles, draw_pile_pos, discard_counts,                                                                   
                      discard_sizes, top_card, e, p, count):                                                                              
    # recycle discards (except top card) into the BOTTOM of the draw pile                                                                 
    if draw_pile_pos[e] < count and discard_sizes[e] > 1:                                                                                 
        top = top_card[e]                                                                                                                 
        n_rec = discard_sizes[e] - 1                                                                                                      
        start = draw_pile_pos[e]                                                                                                          
        for i in range(start - 1, -1, -1):                 # shift existing up                                                            
            draw_piles[e, n_rec + i] = draw_piles[e, i]                                                                                   
        idx = 0                                                                                                                           
        for c in range(54):                                # write recycled at bottom                                                     
            cnt = discard_counts[e, c]                                                                                                    
            if c == top:                                                                                                                  
                cnt -= 1                                                                                                                  
            for _ in range(cnt):                                                                                                          
                draw_piles[e, idx] = c                                                                                                    
                idx += 1                                                                                                                  
        for i in range(n_rec - 1, 0, -1):                  # Fisher-Yates shuffle                                                         
            j = np.random.randint(0, i + 1)                                                                                               
            tmp = draw_piles[e, i]                                                                                                        
            draw_piles[e, i] = draw_piles[e, j]                                                                                           
            draw_piles[e, j] = tmp                                                                                                        
        draw_pile_pos[e] = start + n_rec                                                                                                  
        for c in range(54):                                                                                                               
            discard_counts[e, c] = 0                                                                                                      
        discard_counts[e, top] = 1                                                                                                        
        discard_sizes[e] = 1                                                                                                              
                                                                                                                                          
    actual = count                                                                                                                        
    if draw_pile_pos[e] < actual:                                                                                                         
        actual = draw_pile_pos[e]                                                                                                         
    last = -1                                                                                                                             
    for _ in range(actual):                                # draw from the TOP (high indices)                                             
        c = draw_piles[e, draw_pile_pos[e] - 1]                                                                                           
        draw_pile_pos[e] -= 1                                                                                                             
        hands[e, p, c] += np.int16(1)                                                                                                     
        last = c                                                                                                                          
    return actual, last                                                                                                                   
                                                                                                                                          
                                                                                                                                          
@njit(cache=True)                                                                                                                         
def _exec_play(hands, draw_piles, draw_pile_pos, discard_counts, discard_sizes,                                                           
               top_card, active_color, drew_flags, e, p, opp, act):                                                                       
    grant_extra = False                                                                                                                   
    declared = -1                                                                                                                         
    for c4 in range(4):                                                                                                                   
        drew_flags[e, p, c4] = np.float32(0.0)                                                                                            
    if act < 52:                                                                                                                          
        hands[e, p, act] -= np.int16(1)                                                                                                   
        top_card[e] = act                                                                                                                 
        discard_counts[e, act] += np.int16(1)                                                                                             
        discard_sizes[e] += 1                                                                                                             
        active_color[e] = act // 13                                                                                                       
        ctype = act % 13                                                                                                                  
        if ctype == 10 or ctype == 11:                     # skip / reverse                                                               
            grant_extra = True                                                                                                            
        elif ctype == 12:                                  # +2                                                                           
            _recycle_and_draw(hands, draw_piles, draw_pile_pos, discard_counts,                                                           
                              discard_sizes, top_card, e, opp, 2)                                                                         
            grant_extra = True                                                                                                            
    elif act < 56:                                         # wild                                                                         
        hands[e, p, 52] -= np.int16(1)                                                                                                    
        top_card[e] = 52                                                                                                                  
        discard_counts[e, 52] += np.int16(1)                                                                                              
        discard_sizes[e] += 1                                                                                                             
        declared = act - 52                                                                                                               
        active_color[e] = declared                                                                                                        
    else:                                                  # wild +4                                                                      
        hands[e, p, 53] -= np.int16(1)                                                                                                    
        top_card[e] = 53                                                                                                                  
        discard_counts[e, 53] += np.int16(1)                                                                                              
        discard_sizes[e] += 1                                                                                                             
        declared = act - 56                                                                                                               
        active_color[e] = declared                                                                                                        
        _recycle_and_draw(hands, draw_piles, draw_pile_pos, discard_counts,                                                               
                          discard_sizes, top_card, e, opp, 4)                                                                             
        grant_extra = True                                                                                                                
    return grant_extra, declared                                                                                                          
                                                                                                                                          
                                                                                                                                          
@njit(cache=True)                                                                                                                         
def step_kernel(hands, draw_piles, draw_pile_pos, discard_counts, discard_sizes,                                                          
                current_player, current_phase, last_drawn, top_card, active_color,                                                        
                turn_counts, drew_flags, move_history, actions, env_mask,                                                                 
                rewards, dones, winners, out_turns, max_turns):                                                                           
    N = actions.shape[0]                                                                                                                  
    L = move_history.shape[1]                                                                                                             
    for e in range(N):                                                                                                                    
        out_turns[e] = turn_counts[e]                                                                                                     
        if not env_mask[e]:                                                                                                               
            continue                                                                                                                      
        p = current_player[e]                                                                                                             
        opp = 1 - p                                                                                                                       
        act = actions[e]                                                                                                                  
        phase = current_phase[e]                                                                                                          
        drawn_count = 0                                                                                                                   
        declared_color = -1                                                                                                               
        grant_extra = False                                                                                                               
        switch = False                                                                                                                    
                                                                                                                                          
        if phase == 0:                                                                                                                    
            if act == 60:                                  # DRAW                                                                         
                actual, drawn_card = _recycle_and_draw(                                                                                   
                    hands, draw_piles, draw_pile_pos, discard_counts,                                                                     
                    discard_sizes, top_card, e, p, 1)                                                                                     
                turn_counts[e] += 1                                                                                                       
                drew_flags[e, p, active_color[e]] = np.float32(1.0)                                                                       
                drawn_count = actual                                                                                                      
                legal = False                                                                                                             
                if drawn_card >= 52:                                                                                                      
                    legal = True                                                                                                          
                elif drawn_card >= 0:                                                                                                     
                    top = top_card[e]                                                                                                     
                    tt = top % 13                                                                                                         
                    if top >= 52:                                                                                                         
                        tt = top                                                                                                          
                    if drawn_card // 13 == active_color[e] or drawn_card % 13 == tt:                                                      
                        legal = True                                                                                                      
                if legal:                                                                                                                 
                    current_phase[e] = 1                                                                                                  
                    last_drawn[e] = drawn_card                                                                                            
                else:                                                                                                                     
                    current_phase[e] = 0                                                                                                  
                    last_drawn[e] = -1                                                                                                    
                    switch = True                                                                                                         
            else:                                          # play from hand                                                               
                turn_counts[e] += 1                                                                                                       
                grant_extra, declared_color = _exec_play(                                                                                 
                    hands, draw_piles, draw_pile_pos, discard_counts, discard_sizes,                                                      
                    top_card, active_color, drew_flags, e, p, opp, act)                                                                   
                switch = not grant_extra                                                                                                  
        else:                                              # post-draw phase                                                              
            if act == 61:                                  # PASS                                                                         
                current_phase[e] = 0                                                                                                      
                last_drawn[e] = -1                                                                                                        
                switch = True                                                                                                             
            else:                                          # play drawn card                                                              
                grant_extra, declared_color = _exec_play(                                                                                 
                    hands, draw_piles, draw_pile_pos, discard_counts, discard_sizes,                                                      
                    top_card, active_color, drew_flags, e, p, opp, act)                                                                   
                current_phase[e] = 0                                                                                                      
                last_drawn[e] = -1                                                                                                        
                switch = not grant_extra                                                                                                  
                                                                                                                                          
        # history (in-place shift, no np.roll)                                                                                            
        dc = declared_color if declared_color != -1 else active_color[e]                                                                  
        for i in range(L - 1):                                                                                                            
            move_history[e, i, 0] = move_history[e, i + 1, 0]                                                                             
            move_history[e, i, 1] = move_history[e, i + 1, 1]                                                                             
            move_history[e, i, 2] = move_history[e, i + 1, 2]                                                                             
            move_history[e, i, 3] = move_history[e, i + 1, 3]                                                                             
        move_history[e, L - 1, 0] = np.float32(p)                                                                                         
        move_history[e, L - 1, 1] = np.float32(act) / np.float32(61.0)                                                                    
        move_history[e, L - 1, 2] = (np.float32(dc) + np.float32(1.0)) / np.float32(5.0)                                                  
        move_history[e, L - 1, 3] = np.float32(drawn_count) / np.float32(4.0)                                                             
                                                                                                                                          
        # termination                                                                                                                     
        my_total = 0                                                                                                                      
        opp_total = 0                                                                                                                     
        for c in range(54):                                                                                                               
            my_total += hands[e, p, c]                                                                                                    
            opp_total += hands[e, opp, c]                                                                                                 
        if my_total == 0:                                                                                                                 
            dones[e] = True                                                                                                               
            winners[e] = np.int8(p)                                                                                                       
            rewards[e] = np.float32(1.0)                                                                                                  
        elif turn_counts[e] >= max_turns:                                                                                                 
            dones[e] = True                                                                                                               
            if my_total < opp_total:                                                                                                      
                winners[e] = np.int8(p)                                                                                                   
                rewards[e] = np.float32(1.0)                                                                                              
            elif my_total > opp_total:                                                                                                    
                winners[e] = np.int8(opp)                                                                                                 
                rewards[e] = np.float32(-1.0)                                                                                             
        else:                                                                                                                             
            rewards[e] = np.float32((opp_total - my_total) / 54.0 * 0.05)                                                                 
            if switch:                                                                                                                    
                current_player[e] = opp                                                                                                   
        out_turns[e] = turn_counts[e]