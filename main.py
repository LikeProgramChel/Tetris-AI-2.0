import pygame
import random
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os
import json

# Глобальные настройки
settings = {
    "language": "ru",
    "sound_enabled": True,
    "resolution": [1000, 600],
    "custom_resolution": False,
    "theme": "light"
}

# Тексты для разных языков
texts = {
    "ru": {
        "new_game": "Новая игра",
        "load_game": "Загрузить игру",
        "save_game": "Сохранить игру",
        "highscores": "Рекорды",
        "exit": "Выход",
        "settings": "Настройки",
        "language": "Язык: Русский",
        "sound": "Звук: Вкл",
        "resolution": "Разрешение: {0}x{1}",
        "custom_res": "Ввести разрешение вручную",
        "width": "Ширина:",
        "height": "Высота:",
        "theme": "Тема: Светлая",
        "back": "Назад",
        "score": "Счет: ",
        "game_over": "Вы проиграли",
        "press_esc": "Нажмите ESC",
        "saved": "Игра сохранена!"
    },
    "en": {
        "new_game": "New Game",
        "load_game": "Load Game",
        "save_game": "Save Game",
        "highscores": "Highscores",
        "exit": "Exit",
        "settings": "Settings",
        "language": "Language: English",
        "sound": "Sound: On",
        "resolution": "Resolution: {0}x{1}",
        "custom_res": "Enter resolution manually",
        "width": "Width:",
        "height": "Height:",
        "theme": "Theme: Light",
        "back": "Back",
        "score": "Score: ",
        "game_over": "Game Over",
        "press_esc": "Press ESC",
        "saved": "Game saved!"
    }
}

# Темы оформления
themes = {
    "light": {
        "name_ru": "Светлая",
        "name_en": "Light",
        "background": "background_light.jpg",
        "colors": [
            (255, 255, 255),  # Пустая клетка
            (120, 37, 179),   # Фиолетовый
            (100, 179, 179),  # Бирюзовый
            (80, 34, 22),     # Коричневый
            (80, 134, 22),    # Зелёный
            (180, 34, 22),    # Красный
            (180, 34, 122),   # Розовый
        ],
        "white": (0, 0, 0),
        "black": (255, 255, 255),
        "gray": (128, 128, 128),
        "button_color": (100, 100, 100),
        "button_hover_color": (150, 150, 150)
    },
    "dark": {
        "name_ru": "Тёмная",
        "name_en": "Dark",
        "background": "background_dark.jpg",
        "colors": [
            (50, 50, 50),     # Пустая клетка
            (200, 50, 200),   # Яркий фиолетовый
            (50, 200, 200),   # Яркий бирюзовый
            (150, 100, 50),   # Тёмный оранжевый
            (50, 200, 50),    # Яркий зелёный
            (200, 50, 50),    # Яркий красный
            (200, 50, 150),   # Яркий розовый
        ],
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (80, 80, 80),
        "button_color": (50, 50, 50),
        "button_hover_color": (100, 100, 100)
    }
}

# Доступные разрешения
resolutions = [(800, 600), (1000, 600), (1280, 720)]

# Функции для работы с настройками
def load_settings():
    global settings
    if os.path.exists("settings.json") and os.path.getsize("settings.json") > 0:
        try:
            with open("settings.json", "r") as f:
                settings = json.load(f)
            settings["resolution"] = tuple(settings["resolution"])
            if settings["theme"] not in themes:
                settings["theme"] = "light"
        except (json.JSONDecodeError, ValueError):
            save_settings()
    else:
        save_settings()

def save_settings():
    temp_settings = settings.copy()
    temp_settings["resolution"] = list(temp_settings["resolution"])
    with open("settings.json", "w") as f:
        json.dump(temp_settings, f)

# Класс для текстового ввода
class TextInput:
    def __init__(self, x, y, width, height, font, initial_text=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = initial_text
        self.font = font
        self.active = False
        self.color = BUTTON_COLOR
        self.hover_color = BUTTON_HOVER_COLOR

    def draw(self, screen):
        color = self.hover_color if self.active else self.color
        pygame.draw.rect(screen, color, self.rect)
        text_surface = self.font.render(self.text, True, WHITE)
        screen.blit(text_surface, (self.rect.x + 5, self.rect.y + 5))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if event.unicode.isdigit():
                    self.text += event.unicode

    def get_value(self):
        return int(self.text) if self.text.isdigit() else 0

# Класс для распознавания рук
class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        if not self.lmList:
            return [0] * 5
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

# Класс для фигур
class Figure:
    figures = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],              # I
        [[4, 5, 9, 10], [2, 6, 5, 9]],             # Z
        [[6, 7, 9, 10], [1, 5, 6, 10]],            # S
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],  # J
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]], # L
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],    # T
        [[1, 2, 5, 6]]                              # O
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        self.color = random.randint(1, len(colors) - 1)
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])

# Класс игры Тетрис
class Tetris:
    def __init__(self, height, width):
        self.level = 1
        self.score = 0
        self.state = "start"
        self.field = []
        self.height = height
        self.width = width
        self.x = 100
        self.y = 60
        self.zoom = 20
        self.figure = None
        self.next_figure = None
        self.paused = False
        self.field = [[0 for _ in range(width)] for _ in range(height)]
        self.new_next_figure()

    def new_figure(self):
        if self.next_figure:
            self.figure = self.next_figure
            self.figure.x = self.width // 2 - 2
            self.figure.y = 0
        self.new_next_figure()
        if self.intersects():
            self.state = "gameover"
            if settings["sound_enabled"]:
                game_over_sound.play()
            save_highscore(self.score)

    def new_next_figure(self):
        self.next_figure = Figure(0, 0)

    def intersects(self):
        intersection = False
        if self.figure:
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in self.figure.image():
                        if (i + self.figure.y > self.height - 1 or
                            j + self.figure.x > self.width - 1 or
                            j + self.figure.x < 0 or
                            self.field[i + self.figure.y][j + self.figure.x] > 0):
                            intersection = True
        return intersection

    def break_lines(self):
        lines = 0
        for i in range(self.height - 1, -1, -1):
            if all(self.field[i]):
                lines += 1
                del self.field[i]
                self.field.insert(0, [0 for _ in range(self.width)])
        self.score += lines ** 2 * 10

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines()
        self.new_figure()

    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation
        elif settings["sound_enabled"]:
            rotate_sound.play()

    def save_game(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def restart(self):
        self.__init__(20, 10)

    @staticmethod
    def load_game(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

# Обновлённый класс кнопок с изображением
class Button:
    def __init__(self, x, y, width, height, text, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.is_hovered = False
        # Загружаем изображение кнопки и масштабируем его под заданный размер
        self.image = pygame.transform.scale(pygame.image.load("button.png"), (width, height))
        # Создаём затемнённую версию для эффекта наведения
        self.hover_image = self.image.copy()
        self.hover_image.fill((50, 50, 50, 100), special_flags=pygame.BLEND_RGBA_SUB)

    def draw(self, screen):
        # Отрисовываем изображение кнопки
        if self.is_hovered:
            screen.blit(self.hover_image, self.rect)
        else:
            screen.blit(self.image, self.rect)
        # Отрисовываем текст поверх изображения
        text_surface = self.font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

# Функции для рекордов
def save_highscore(score, player_name="Player"):
    highscores = load_highscores()
    highscores.append((score, player_name))
    highscores = sorted(highscores, reverse=True)[:5]
    with open("highscores.txt", "w") as f:
        for s, name in highscores:
            f.write(f"{s},{name}\n")

def load_highscores():
    highscores = []
    if os.path.exists("highscores.txt"):
        with open("highscores.txt", "r") as f:
            for line in f:
                score, name = line.strip().split(",")
                highscores.append((int(score), name))
    return sorted(highscores, reverse=True)[:5]

# Создание кнопок меню
def create_menu_buttons():
    button_width = 200
    button_height = 50
    button_x = size[0] // 2 - button_width // 2
    lang = settings["language"]
    return [
        Button(button_x, 200, button_width, button_height, texts[lang]["new_game"], font),
        Button(button_x, 270, button_width, button_height, texts[lang]["load_game"], font),
        Button(button_x, 340, button_width, button_height, texts[lang]["save_game"], font),
        Button(button_x, 410, button_width, button_height, texts[lang]["highscores"], font),
        Button(button_x, 480, button_width, button_height, texts[lang]["settings"], font),
        Button(button_x, 550, button_width, button_height, texts[lang]["exit"], font),
    ]

# Создание кнопок настроек
def create_settings_buttons():
    button_width = 300
    button_height = 50
    button_x = size[0] // 2 - button_width // 2
    lang = settings["language"]
    sound_text = texts[lang]["sound"].replace("Вкл" if lang == "ru" else "On", "Выкл" if lang == "ru" else "Off") if not settings["sound_enabled"] else texts[lang]["sound"]
    res_text = texts[lang]["resolution"].format(*settings["resolution"])
    custom_res_text = f"[{'X' if settings['custom_resolution'] else ' '}] {texts[lang]['custom_res']}"
    theme_text = texts[lang]["theme"].replace("Светлая" if lang == "ru" else "Light", themes[settings["theme"]]["name_" + lang])
    return [
        Button(button_x, 200, button_width, button_height, texts[lang]["language"], font),
        Button(button_x, 270, button_width, button_height, sound_text, font),
        Button(button_x, 340, button_width, button_height, res_text, font),
        Button(button_x, 410, button_width, button_height, custom_res_text, font),
        Button(button_x, 480, button_width, button_height, theme_text, font),
        Button(button_x, 550, button_width, button_height, texts[lang]["back"], font),
    ]

# Меню настроек
def settings_menu(screen):
    global size, background, colors, WHITE, BLACK, GRAY, BUTTON_COLOR, BUTTON_HOVER_COLOR
    buttons = create_settings_buttons()
    width_input = TextInput(size[0] // 2 + 50, 480, 100, 30, font, str(settings["resolution"][0]))
    height_input = TextInput(size[0] // 2 + 50, 520, 100, 30, font, str(settings["resolution"][1]))
    while True:
        screen.blit(background, (0, 0))
        mouse_pos = pygame.mouse.get_pos()
        lang = settings["language"]

        for button in buttons:
            button.check_hover(mouse_pos)
            button.draw(screen)

        if settings["custom_resolution"]:
            width_text = font.render(texts[lang]["width"], True, WHITE)
            height_text = font.render(texts[lang]["height"], True, WHITE)
            screen.blit(width_text, (size[0] // 2 - 150, 485))
            screen.blit(height_text, (size[0] // 2 - 150, 525))
            width_input.draw(screen)
            height_input.draw(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for button in buttons:
                    if button.is_hovered:
                        if button.text.startswith(texts[lang]["language"].split(":")[0]):
                            settings["language"] = "en" if settings["language"] == "ru" else "ru"
                            save_settings()
                            buttons = create_settings_buttons()
                        elif button.text.startswith(texts[lang]["sound"].split(":")[0]):
                            settings["sound_enabled"] = not settings["sound_enabled"]
                            if settings["sound_enabled"]:
                                pygame.mixer.music.unpause()
                            else:
                                pygame.mixer.music.pause()
                            save_settings()
                            buttons = create_settings_buttons()
                        elif button.text.startswith(texts[lang]["resolution"].split(":")[0]) and not settings["custom_resolution"]:
                            current_idx = resolutions.index(settings["resolution"]) if settings["resolution"] in resolutions else 0
                            settings["resolution"] = resolutions[(current_idx + 1) % len(resolutions)]
                            size = settings["resolution"]
                            screen = pygame.display.set_mode(size)
                            background = pygame.transform.scale(pygame.image.load(themes[settings["theme"]]["background"]), size)
                            save_settings()
                            buttons = create_settings_buttons()
                        elif button.text.startswith("[") and texts[lang]["custom_res"] in button.text:
                            settings["custom_resolution"] = not settings["custom_resolution"]
                            save_settings()
                            buttons = create_settings_buttons()
                        elif button.text.startswith(texts[lang]["theme"].split(":")[0]):
                            settings["theme"] = "dark" if settings["theme"] == "light" else "light"
                            colors = themes[settings["theme"]]["colors"]
                            WHITE = themes[settings["theme"]]["white"]
                            BLACK = themes[settings["theme"]]["black"]
                            GRAY = themes[settings["theme"]]["gray"]
                            BUTTON_COLOR = themes[settings["theme"]]["button_color"]
                            BUTTON_HOVER_COLOR = themes[settings["theme"]]["button_hover_color"]
                            background = pygame.transform.scale(pygame.image.load(themes[settings["theme"]]["background"]), size)
                            save_settings()
                            buttons = create_settings_buttons()
                        elif button.text == texts[lang]["back"]:
                            if settings["custom_resolution"]:
                                w, h = width_input.get_value(), height_input.get_value()
                                if w > 0 and h > 0:
                                    settings["resolution"] = (w, h)
                                    size = settings["resolution"]
                                    screen = pygame.display.set_mode(size)
                                    background = pygame.transform.scale(pygame.image.load(themes[settings["theme"]]["background"]), size)
                                    save_settings()
                            return
            if settings["custom_resolution"]:
                width_input.handle_event(event)
                height_input.handle_event(event)

        pygame.display.flip()
        clock.tick(fps)

# Главное меню
def main_menu(screen, buttons, game=None):
    while True:
        screen.blit(background, (0, 0))
        mouse_pos = pygame.mouse.get_pos()
        lang = settings["language"]

        for button in buttons:
            button.check_hover(mouse_pos)
            button.draw(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN:
                for button in buttons:
                    if button.is_hovered:
                        if button.text == texts[lang]["new_game"]:
                            return "new_game"
                        elif button.text == texts[lang]["load_game"]:
                            return "load_game"
                        elif button.text == texts[lang]["save_game"]:
                            if game:
                                game.save_game("save.pkl")
                                print(texts[lang]["saved"])
                            return "continue"
                        elif button.text == texts[lang]["highscores"]:
                            show_highscores(screen)
                        elif button.text == texts[lang]["settings"]:
                            settings_menu(screen)
                            buttons = create_menu_buttons()
                        elif button.text == texts[lang]["exit"]:
                            return "quit"

        pygame.display.flip()
        clock.tick(fps)

# Таблица рекордов
def show_highscores(screen):
    highscores = load_highscores()
    lang = settings["language"]
    while True:
        screen.blit(background, (0, 0))
        title = font1.render(texts[lang]["highscores"], True, WHITE)
        screen.blit(title, (size[0] // 2 - title.get_width() // 2, 50))

        for i, (score, name) in enumerate(highscores):
            text = font.render(f"{i+1}. {name}: {score}", True, WHITE)
            screen.blit(text, (size[0] // 2 - text.get_width() // 2, 150 + i * 40))

        back_button = Button(size[0] // 2 - 100, 400, 200, 50, texts[lang]["back"], font)
        mouse_pos = pygame.mouse.get_pos()
        back_button.check_hover(mouse_pos)
        back_button.draw(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and back_button.is_hovered:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

        pygame.display.flip()
        clock.tick(fps)

# Инициализация Pygame
pygame.init()
pygame.mixer.init()
load_settings()
pygame.mixer.music.load('background_music.mp3')
pygame.mixer.music.play(-1)
if not settings["sound_enabled"]:
    pygame.mixer.music.pause()

# Звуковые эффекты
rotate_sound = pygame.mixer.Sound("rotate.wav")
game_over_sound = pygame.mixer.Sound("game_over.wav")

# Установка темы
colors = themes[settings["theme"]]["colors"]
WHITE = themes[settings["theme"]]["white"]
BLACK = themes[settings["theme"]]["black"]
GRAY = themes[settings["theme"]]["gray"]
BUTTON_COLOR = themes[settings["theme"]]["button_color"]
BUTTON_HOVER_COLOR = themes[settings["theme"]]["button_hover_color"]

size = settings["resolution"]
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Tetris")

background = pygame.transform.scale(pygame.image.load(themes[settings["theme"]]["background"]), size)

# Шрифты
font_path = "font.ttf"
font = pygame.font.Font(font_path, 25)
font1 = pygame.font.Font(font_path, 65)

# Инициализация
detector = HandDetector()
done = False
clock = pygame.time.Clock()
fps = 15
game = Tetris(20, 10)
counter = 0
cap = cv2.VideoCapture(0)

# Главное меню
buttons = create_menu_buttons()
menu_result = main_menu(screen, buttons)

if menu_result == "new_game":
    game = Tetris(20, 10)
elif menu_result == "load_game":
    game = Tetris.load_game("save.pkl")
elif menu_result == "quit":
    pygame.quit()
    sys.exit()

# Основной цикл
while not done:
    if not game.paused:
        if game.figure is None:
            game.new_figure()
        counter += 1
        if counter > 100000:
            counter = 0
        if counter % (fps // 2) == 0 and game.state == "start":
            game.go_down()

    # Обработка веб-камеры
    success, img = cap.read()
    if success:
        img = detector.findHands(img, draw=True)
        lmList, bbox = detector.findPosition(img, draw=True)
        if len(lmList) != 0:
            fingers = detector.fingersUp()
            if fingers[1] == 1 and fingers[4] == 0:
                game.go_side(-1)
                time.sleep(0.1)
            elif fingers[1] == 0 and fingers[4] == 1:
                game.go_side(1)
                time.sleep(0.1)
            elif fingers[1] == 1 and fingers[4] == 1:
                game.rotate()
                time.sleep(0.1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.rot90(img)
        img = pygame.surfarray.make_surface(img)
        img = pygame.transform.scale(img, (320, 240))

    # Отрисовка
    screen.blit(background, (0, 0))
    if success:
        screen.blit(img, (size[0] - 320, 0))

    for i in range(game.height):
        for j in range(game.width):
            pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
            if game.field[i][j] > 0:
                pygame.draw.rect(screen, colors[game.field[i][j]],
                                 [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])

    if game.figure:
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in game.figure.image():
                    pygame.draw.rect(screen, colors[game.figure.color],
                                     [game.x + game.zoom * (j + game.figure.x) + 1,
                                      game.y + game.zoom * (i + game.figure.y) + 1,
                                      game.zoom - 2, game.zoom - 2])

    if game.next_figure:
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in game.next_figure.image():
                    pygame.draw.rect(screen, colors[game.next_figure.color],
                                     [size[0] // 2 - 40 + game.zoom * j, 20 + game.zoom * i,
                                      game.zoom - 2, game.zoom - 2])

    lang = settings["language"]
    text = font.render(texts[lang]["score"] + str(game.score), True, WHITE)
    text_game_over = font1.render(texts[lang]["game_over"], True, (255, 125, 0))
    text_game_over1 = font1.render(texts[lang]["press_esc"], True, (255, 215, 0))

    screen.blit(text, [0, 0])
    if game.state == "gameover":
        screen.blit(text_game_over, [20, 200])
        screen.blit(text_game_over1, [25, 265])

    pygame.display.flip()
    clock.tick(fps)

    # Обработка событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                game.restart()
            elif event.key == pygame.K_p:
                game.paused = not game.paused
            elif event.key == pygame.K_s:
                game.save_game("save.pkl")
                print(texts[lang]["saved"])
            elif event.key == pygame.K_m:
                menu_result = main_menu(screen, create_menu_buttons(), game)
                if menu_result == "new_game":
                    game = Tetris(20, 10)
                elif menu_result == "load_game":
                    game = Tetris.load_game("save.pkl")
                elif menu_result == "quit":
                    done = True
            elif event.key == pygame.K_LEFT:
                game.go_side(-1)
            elif event.key == pygame.K_RIGHT:
                game.go_side(1)
            elif event.key == pygame.K_DOWN:
                game.go_down()
            elif event.key == pygame.K_UP:
                game.rotate()
            elif event.key == pygame.K_SPACE:
                game.go_space()

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
pygame.quit()