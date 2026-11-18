import sys, os

def resource_path(rel):
    """Путь к ресурсу и при запуске из исходника, и внутри exe."""
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)

import pygame
import random

from pygame.constants import MOUSEBUTTONDOWN

pygame.init()

# --- КОНСТАНТЫ И НАСТРОЙКИ ---
clock = pygame.time.Clock()
FPS = 60
# ---размеры экрана---
WIDTH, HEIGHT = 1400, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Крестики-нолики")
avatar = pygame.image.load(resource_path('img.png')).convert_alpha()
pygame.display.set_icon(avatar)

# ---переменные---
ecran = "игра"
board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

CELL = 250          # размер клетки
BX, BY = 50, 10     # левый верхний угол поля
GAP = 20            # толщина линии сетки

turn = 1

# --- ДОБАВЛЕНО: кто начинает партию (2 — нейросеть первой партии ходит первой) ---
first_player = 2
turn = first_player

BTN = pygame.Rect(927, 47, 406, 106)

# --- ДОБАВЛЕНО: кнопка обнуления счёта ---
SCORE_BTN = pygame.Rect(927, 575, 406, 106)

# --- ДОБАВЛЕНО: счёт (победы игрока / нейросети / ничьи) ---
score = {"player": 0, "ai": 0, "draw": 0}
counted = False   # чтобы партия засчиталась один раз, а не каждый кадр

# --- ДОБАВЛЕНО: результат партии (None — играем, иначе 0/1/2) ---
result = None

# --- ИСПРАВЛЕНО: старт отсчёта времени — один раз, до цикла ---
start_time = pygame.time.get_ticks()

# --- ИЗМЕНЕНО: draw_text умеет ужимать шрифт под max_width ---
def draw_text(x, y, t, int, color="white", max_width=None):
    font = pygame.font.Font(None, int)
    text = font.render(f"{t}", True, f"{color}")
    # --- ДОБАВЛЕНО: уменьшаем шрифт, пока текст не влезет ---
    if max_width is not None:
        while text.get_width() > max_width and int > 10:
            int -= 2
            font = pygame.font.Font(None, int)
            text = font.render(f"{t}", True, f"{color}")
    screen.blit(text, (x, y))
# --- ДОБАВЛЕНО: координаты клетки по её номеру (0..8) ---
def cell_pos(i):
    col, row = i % 3, i // 3
    return BX + col * (CELL + GAP), BY + row * (CELL + GAP)

# --- ДОБАВЛЕНО: рисование крестика ---
def draw_x(x, y):
    m = 60            # отступ от углов клетки
    lw = 15           # толщина линии
    c = "white"
    pygame.draw.line(screen, c, (x + m, y + m), (x + CELL - m, y + CELL - m), lw)
    pygame.draw.line(screen, c, (x + CELL - m, y + m), (x + m, y + CELL - m), lw)

# --- ДОБАВЛЕНО: рисование нолика ---
def draw_o(x, y):
    cx, cy = x + CELL // 2, y + CELL // 2
    pygame.draw.circle(screen, "white", (cx, cy), CELL // 2 - 55, 15)

# --- ДОБАВЛЕНО: рисуем всё поле из board (0 — пусто, 1 — крестик, 2 — нолик) ---
def draw_board():
    for i, v in enumerate(board):
        x, y = cell_pos(i)
        if v == 1:
            draw_x(x, y)
        elif v == 2:
            draw_o(x, y)

draw_text(250, 300, "загрузка...", 200)
pygame.display.flip()
import numpy as np
from tensorflow import keras

model = keras.models.load_model(resource_path('tic_tac_toe_Stas.gpt_2.keras'))
def check_game_over(b):
    """1 - победа крестика, 2 - нолика, 0 - ничья, None - играем."""
    WIN = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for a, c, d in WIN:
        if b[a] != 0 and b[a] == b[c] == b[d]:
            return b[a]
    if all(v != 0 for v in b):
        return 0
    return None

def free_cells(b):
    return [i for i, v in enumerate(b) if v == 0]

def encode(b, player):
    me, opp = (1, 2) if player == 1 else (2, 1)
    return np.array([1.0 if c == me else -1.0 if c == opp else 0.0 for c in b])

def q_values(b, player):
    """БЫСТРОЕ предсказание: без накладок model.predict."""
    return model(encode(b, player)[None], training=False).numpy()[0]

def best_action(q, b):
    """Лучший ход по оценкам сети (только свободные клетки)."""
    q_masked = [q[i] if v == 0 else -999 for i, v in enumerate(b)]
    return q_masked.index(max(q_masked))

# --- ДОБАВЛЕНО: текст результата по коду q (игрок — крестик, нейросеть — нолик) ---
def result_text(q):
    if q == 1:
        return "Победа игрока!", "dark green"
    if q == 2:
        return "Победа нейросети!", "red"
    return "Ничья!", "yellow"

running = True
while running:
    keys = pygame.key.get_pressed()
    # реагирование
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            # Реакция на нажатие кнопки мыши
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if BTN.collidepoint(event.pos):
                board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                # --- ИЗМЕНЕНО: чередуем не текущий ход, а того, кто начинает партию ---
                first_player = 3 - first_player
                turn = first_player
                # --- ДОБАВЛЕНО: сброс результата и таймера ---
                result = None
                counted = False
                start_time = pygame.time.get_ticks()
            # --- ДОБАВЛЕНО: кнопка обнуления счёта ---
            elif SCORE_BTN.collidepoint(event.pos):
                score = {"player": 0, "ai": 0, "draw": 0}
                counted = False
            # --- ИЗМЕНЕНО: игрок кликает только в свой ход (1) и пока партия идёт ---
            elif result is None and ecran == "игра" and turn == 1:
                col = (event.pos[0] - BX) // (CELL + GAP)
                row = (event.pos[1] - BY) // (CELL + GAP)
                if 0 <= col <= 2 and 0 <= row <= 2:
                    i = row * 3 + col
                    if board[i] == 0:
                        board[i] = turn
                        turn = 3 - turn  # ход переходит: 1 <-> 2
    if ecran == "игра":
        screen.fill("gray")
        screen.fill("black", rect=(300, 10, 20, 770))
        screen.fill("black", rect=(550, 10, 20, 770))
        screen.fill("black", rect=(50, 260, 770, 20))
        screen.fill("black", rect=(50, 510, 770, 20))

        screen.fill("red", rect=(925, 45, 410, 110))
        screen.fill("orange", rect=(927, 47, 406, 106))
        draw_text(933, 70, "сбросить поле", 80, "white")
        draw_text(960, 700, "нейросеть: Stas_gpt2", 50, "brown")
        draw_text(960, 750, "автор: Соколовский Стас", 50, "brown")
        draw_board()

        q = check_game_over(board)
        # --- ДОБАВЛЕНО: счёт обновляем один раз в момент конца партии ---
        if q != None:
            result = q
            if not counted:
                if result == 1:
                    score["player"] += 1
                elif result == 2:
                    score["ai"] += 1
                else:
                    score["draw"] += 1
                counted = True
        if result != None:
            # --- ИЗМЕНЕНО: передаём max_width, чтобы надпись уместилась ---
            t, c = result_text(result)
            draw_text(933, 300, t, 90, c, max_width=WIDTH - 933 - 10)

        # --- ДОБАВЛЕНО: счёт (под результатом, над кнопкой обнуления) ---
        draw_text(933, 390, f"побед: {score['player']}", 60, "dark green")
        draw_text(933, 440, f"проигрышей: {score['ai']}", 60, "red")
        draw_text(933, 490, f"ничьих: {score['draw']}", 60, "yellow")

        # --- ДОБАВЛЕНО: кнопка обнуления счёта ---
        screen.fill("dark blue", rect=(SCORE_BTN.x - 2, SCORE_BTN.y - 2, SCORE_BTN.width + 4, SCORE_BTN.height + 4))
        screen.fill("light blue", rect=SCORE_BTN)
        draw_text(933, 595, "обнулить счёт", 70, "black", max_width=SCORE_BTN.width - 20)

    if ecran == "игра" and result is None and turn == 2 and 0 in board:
        qv = q_values(board, 2)
        board[best_action(qv, board)] = 2
        turn = 3 - turn
    clock.tick(FPS)
    pygame.display.flip()
pygame.quit()
