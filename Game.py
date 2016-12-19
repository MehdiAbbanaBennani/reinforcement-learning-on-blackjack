import random as rand


class Game():

    @staticmethod
    def generate_card(player_score):
        card_number = rand.randint(1, 10)
        card_color = - ((rand.randint(0, 2) % 2) - 1 / 2) * 2
        # The card color 1 is black with probability 0.66 and -1 is red
        player_score += card_color * card_number
        return player_score

    @staticmethod
    def first_step():
        return [rand.randint(1, 10), rand.randint(1, 10)]

    def step(self, do_hit, scores):
        reward = 0
        is_terminal = 0
        player_score = scores[0]
        dealer_score = scores[1]

        if do_hit == 1:
            player_score = self.generate_card(player_score)
            if player_score > 21 or player_score <= 0:
                reward = -1
                is_terminal = 1

        else:
            while 0 < dealer_score < 17:
                dealer_score = self.generate_card(dealer_score)

            if dealer_score > 21 or player_score > dealer_score or dealer_score < 1:
                reward = 1
            elif dealer_score == player_score:
                reward = 0
            else:
                reward = -1

            is_terminal = 1

        scores = [player_score, dealer_score]

        return scores, reward, is_terminal
