import random as rand

²
def generate_card(player_score):
    card_number = rand.randint(1, 10)
    card_color = - ((rand.randint(0, 3) % 2) - 1 / 2) * 2
    # The card color 1 is black with probability 0.66 and -1 is red
    player_score += card_color * card_number
    return player_score


def first_step():
    return rand.randint(1, 10), rand.randint(1, 10)


def step(do_hit, player_score, dealer_score):
    reward = 0

    if do_hit == 1:
        player_score = generate_card(player_score)
        if player_score > 21 or player_score <= 0:
            reward = -1

    else:
        while dealer_score < 17:
            dealer_score = generate_card(dealer_score)
        if dealer_score > 21:
            reward = 1
        elif dealer_score == player_score:
            reward = 0
        else:
            reward = -1

    return player_score, reward

