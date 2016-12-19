from game_with_memory.Game_with_memory import GameMemory

limit = 50
i = 0

GameMemory = GameMemory(black_count=20,
                        red_count=10)
scores = GameMemory.first_step()
while i < limit :
    scores, reward, is_terminal, full_memory = GameMemory.step(do_hit=1, scores=scores)

    i += 1
    print(scores)
print(1)