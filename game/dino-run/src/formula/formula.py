#!/usr/bin/python3
import pygame
import os
import random
from gym import Env
from gym.spaces import Box, Discrete
from mss import mss
import pyautogui
import sys
#from Dqn import DQNAgent
import torch.optim as optim
import torch 


import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

class DQNAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.optimizer = None
        self.network()
          
    def network(self):
        # Layers
        self.f1 = nn.Linear(3, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 3)
        # weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x
    
    def get_state(self, has_cactus, has_pterodactyl, is_big):
        """
        Return the state.
        The state is a numpy array of 11 values, representing:
            - Danger 1 OR 2 steps ahead
            - Danger 1 OR 2 steps on the right
            - Danger 1 OR 2 steps on the left
            - Snake is moving left
            - Snake is moving right
            - Snake is moving up
            - Snake is moving down
            - The food is on the left
            - The food is on the right
            - The food is on the upper side
            - The food is on the lower side      
        """
        states = [has_cactus, 
                 has_pterodactyl,
                 is_big]

        int_states = [1 if state else 0 for state in states]
 
        # for i in range(len(state)):
        #     if state[i]:
        #         state[i]=1
        #     else:
        #         state[i]=0

        return np.asarray(int_states)

    def set_reward(self, player, is_dead):
        """
        Return the reward.
        The reward is:
            -10 when Snake crashes. 
            +10 when Snake eats food
            0 otherwise
        """
        self.reward = 0
        if is_dead:
            self.reward = -10
            return self.reward
        if player.obstacle:
            self.reward = 10
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        """
        Replay memory.
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to(DEVICE)
            state_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True).to(DEVICE)
            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][np.argmax(action)] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()            

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(next_state.reshape((1, 2)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 2)), dtype=torch.float32, requires_grad=True).to(DEVICE)
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()

sys.setrecursionlimit(100000000)
pygame.init()

# Global Constants
pygame.display.set_caption('Dino Run')
SCREEN_HEIGHT = 800
SCREEN_WIDTH = 2200
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
RUNNING = [pygame.image.load("images/Dino/DinoRun1.png"),
           pygame.image.load("images/Dino/DinoRun2.png")]
JUMPING = pygame.image.load("images/Dino/DinoJump.png")
DUCKING = [pygame.image.load("images/Dino/DinoDuck1.png"),
           pygame.image.load("images/Dino/DinoDuck2.png")]
MUSHROOM = pygame.image.load("images/Powerup/mush.png")
BIGJUMPING = pygame.image.load("images/BigDino/DinoJump.png")
BIGRUNNING = [pygame.image.load("images/BigDino/DinoRun1.png"),
           pygame.image.load("images/BigDino/DinoRun2.png")]
BIGDUCKING = [pygame.image.load("images/BigDino/DinoDuck1.png"),
           pygame.image.load("images/BigDino/DinoDuck2.png")]
SMALL_CACTUS = [pygame.image.load("images/Cactus/SmallCactus1.png"),
                pygame.image.load("images/Cactus/SmallCactus2.png"),
                pygame.image.load("images/Cactus/SmallCactus3.png")]
LARGE_CACTUS = [pygame.image.load("images/Cactus/LargeCactus1.png"),
                pygame.image.load("images/Cactus/LargeCactus2.png"),
                pygame.image.load("images/Cactus/LargeCactus3.png")]
BIRD = [pygame.image.load("images/Bird/Bird1.png"),
        pygame.image.load("images/Bird/Bird2.png")]
CLOUD = pygame.image.load("images/Other/Cloud.png")
BG = pygame.image.load("images/Other/Track.png")

# Commands
print("")
print("\033[36m📚 HOW TO PLAY?\033[0m")
print("\033[32m🟢 Start moving T-Rex with ANY KEY \033[0m")
print("\033[38;5;214m🟠 Play using UP KEY 🔼 and DOWN KEY 🔽 \033[0m")
print("\033[31m🔴 Press the \"ESCAPE\" KEY on the Dino Run \"GAME OVER\" screen to end the game! \033[0m")
print("")

ZERO = 0
ONE = 1 
TWO = 2

class BigDinosaur:
    X_POS = 80
    Y_POS = 280
    Y_POS_DUCK = 320
    JUMP_VEL = 8.5

    def __init__(self):
        self.duck_img = BIGDUCKING
        self.run_img = BIGRUNNING
        self.jump_img = BIGJUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        if userInput[pygame.K_UP] and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput[pygame.K_DOWN] and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput[pygame.K_DOWN]):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < - self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.obstacle = []

    def update(self, userInput):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        if userInput[pygame.K_UP] and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput[pygame.K_DOWN] and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput[pygame.K_DOWN]):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def update_dqn(self, action: int):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0
        print(action)
        if action == 0 and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif action == 1 and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        else:
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < - self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()
    
    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def get_obstacles(self):
        return self.rect.y, self.rect.x


    def showInfo(self):
        font = pygame.font.Font('freesansbold.ttf', 14)
        textY = font.render("Obstacle height: " + str(SCREEN_HEIGHT - self.rect.y), True, (0, 0, 0))
        textRectY = textY.get_rect()
        textRectY.center = (100, 40)
        SCREEN.blit(textY, textRectY)
        textX = font.render("Obstacle distance: " + str(self.rect.x), True, (0, 0, 0))
        textRectX = textX.get_rect()
        textRectX.center = (100, 60)
        SCREEN.blit(textX, textRectX)



class PowerUp:
    def __init__(self):
        self.image = MUSHROOM
        # self.type = type
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH + 300
        self.rect.y = 320


    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            powerups.pop()

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            powerups.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image, self.rect)

class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = 250
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1


class DinoRun(Env):

    def __init__(self):
        self.powerups = []
        self.powerupOn = False
        self.time = 0
        self.dinoCopy = BigDinosaur()
        self.speed = 0
        self.run = True
        self.player = Dinosaur()
        self.clock = pygame.time.Clock()
        self.cloud = Cloud()
        self.collide = False
        self.game_speed = 20
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.points = 0
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.obstacles = []
        self.death_count = 0
    

    def step(self, action, record: False):
        reward = 0.2
        if action == ZERO:
            reward += 0.1
            self.player.update(action)
        if action == ONE:
            self.player.update(action)
        if action == TWO:
            self.player.update(action)

        pass
def score(game: DinoRun):
    game.points += 1
    if game.points % 100 == 0:
        game.game_speed += 1

    text = game.font.render("Points: " + str(game.points), True, (0, 0, 0))
    textRect = text.get_rect()
    textRect.center = (1000, 40)
    SCREEN.blit(text, textRect)

def main():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles, powerups, powerupON, time, dinoCopy, speed
    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    dinoCopy = player
    playerBig = BigDinosaur()
    cloud = Cloud()
    game_speed = 20
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    powerups = []
    death_count = 0
    powerupOn = False
    time = 0
    speed = game_speed

    def showGameSpeed(game_speed):
        font = pygame.font.Font('freesansbold.ttf', 14)
        text = font.render("Game Speed: " + str(game_speed), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (100, 80)
        SCREEN.blit(text, textRect)
        textClock = font.render("Speed: " + str(speed), True, (0, 0, 0))
        textClockRect = text.get_rect()
        textClockRect.center = (100, 250)
        SCREEN.blit(textClock, textClockRect)
        
    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1

        text = font.render("Points: " + str(points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while run:
        # print('Ta rolando')
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
        
                run = False

        SCREEN.fill((255, 255, 255))
        userInput = pygame.key.get_pressed()

        player.draw(SCREEN)
        player.update(userInput)
        if(powerupOn):
            playerBig.dino_rect.x = player.dino_rect.x
            playerBig.dino_rect.y = player.dino_rect.y
            player = playerBig

        if(powerupOn) and (pygame.time.get_ticks() - time > 3000):
            dinoCopy.dino_rect.x = player.dino_rect.x
            dinoCopy.dino_rect.y = player.dino_rect.y
            player = dinoCopy
            powerupOn = False

        if len(obstacles) == 0:
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 2) == 2:
                obstacles.append(Bird(BIRD))

        if (len(powerups)==0) and (game_speed - speed > 3):
            powerups.append(PowerUp())

        for powerup in powerups:
            if powerupOn == False:
                powerup.rect.y = 300
                if(game_speed - speed > 3):
                    powerup.draw(SCREEN)
                    powerup.update()

            if player.dino_rect.colliderect(powerup.rect):
                powerupOn = True
                powerup.rect.y = 0
                speed = game_speed
                time = pygame.time.get_ticks()
                powerup.rect.x = SCREEN_WIDTH + 300


        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            obstacle.showInfo()
            #information_thread(obstacle.get_obstacles())

            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(0)
                death_count += 1
                menu(death_count)

        

        background() 
        #showGameSpeed(game_speed)

        cloud.draw(SCREEN)
        cloud.update()

        score()

        clock.tick(30)

        pygame.display.update()

def replay_or_quit(death_count: int):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                print("➡️  Thank you for using Ritchie CLI! 🆒")
                pygame.quit()
                quit()
        if death_count > 0 and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                main()
        elif event.type == pygame.KEYUP and death_count == 0:
            main()

import keyboard

def information_thread(obstacles: tuple):
    height = obstacles[0]
    distance = obstacles[1]

    if height == 550 and distance < 200:
        pyautogui.keyDown('down')
        time.sleep(1)
        pyautogui.keyUp('down')
    elif height < 550 and distance < 200:
        pyautogui.keyDown('up')
        pyautogui.keyUp('up')
        # Coloque aqui o código para receber/enviar informações
        # Pode ser um loop que verifica eventos ou comunicação com algum servidor, por exemplo
def menu(death_count):
    global points
    run = True
    while run:
        SCREEN.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 30)

        if death_count == 0:
            text = font.render("Press any Key to Start", True, (0, 0, 0))
        if death_count > 0:
            text = font.render("Press ESC to Exit or   Key to Restart", True, (0, 0, 0))
            score = font.render("Your Score: " + str(points), True, (0, 0, 0))
            scoreRect = score.get_rect()
            scoreRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
            SCREEN.blit(score, scoreRect)
            # while True:
            #     for event in pygame.event.get():
            #         if event.type == pygame.QUIT:
            #             pygame.quit()
            #             quit()
        replay_or_quit(death_count)

        
        textRect = text.get_rect()
        textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        SCREEN.blit(text, textRect)
        SCREEN.blit(RUNNING[0], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 140))
        pygame.display.update()
      


import numpy as np
import random 

DEVICE = 'cpu'

def initialize_game(player: Dinosaur, game: DinoRun, food, agent: DQNAgent, batch_size):
    has_cactus = False
    has_pterodactyl = False
    is_big = False
    state_init1 = agent.get_state(has_cactus, has_pterodactyl, is_big)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.update_dqn(action)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.collide)
    agent.remember(state_init1, action, reward1, state_init2, game.collide)
    agent.replay_new(agent.memory, batch_size)

def get_record(score, record):
    if score >= record:
        return score
    else:
        return record
    

def background(game: DinoRun):
    image_width = BG.get_width()
    SCREEN.blit(BG, (game.x_pos_bg, game.y_pos_bg))
    SCREEN.blit(BG, (image_width + game.x_pos_bg, game.y_pos_bg))
    if game.x_pos_bg <= -image_width:
        SCREEN.blit(BG, (image_width + game.x_pos_bg, game.y_pos_bg))
        game.x_pos_bg = 0
    game.x_pos_bg -= game.game_speed

def display_ui(cloud, game: DinoRun, record):
    background(game)
    score(game)
    cloud.draw(SCREEN)
    cloud.update(game.game_speed)



def display(player: Dinosaur, game: DinoRun, record):
    #game.gameDisplay.fill((255, 255, 255))
    SCREEN.fill((255, 255, 255))
    cloud = Cloud()
    display_ui(cloud, game, record)
    #player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    player.draw(SCREEN)
    if len(game.obstacles) == 0:
        if random.randint(0, 2) == 0:
            game.obstacles.append(SmallCactus(SMALL_CACTUS))
        elif random.randint(0, 2) == 1:
            game.obstacles.append(LargeCactus(LARGE_CACTUS))
        elif random.randint(0, 2) == 2:
            game.obstacles.append(Bird(BIRD))

        if (len(game.powerups)==0) and (game.game_speed - game.speed > 3):
            game.powerups.append(PowerUp())
        
        for powerup in game.powerups:
            if game.powerupOn == False:
                powerup.rect.y = 300
                if(game.game_speed - game.speed > 3):
                    powerup.draw(SCREEN)
                    powerup.update(game.game_speed)

            if player.dino_rect.colliderect(powerup.rect):
                game.powerupOn = True
                powerup.rect.y = 0
                game.speed = game.game_speed
                game.time = pygame.time.get_ticks()
                powerup.rect.x = SCREEN_WIDTH + 300


        for obstacle in game.obstacles:
            obstacle.draw(SCREEN)
            obstacle.update(game.game_speed)
            obstacle.showInfo()

            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(0)
                death_count += 1

def update_screen():
    pygame.display.update()


    
def run(params):
    """
    Run the DQN algorithm, based on the parameters previously set.   
    """
    pygame.init()
    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    while counter_games < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Initialize classes
        game = DinoRun()
        player1 = game.player
        #food1 = game.food
        obstacle = game.obstacles
        # Perform first move
        initialize_game(player1, game, obstacle, agent, params['batch_size'])
        if params['display']:
            display(player1, game, record)
        while True:
            a = 2
        
    return 0

import argparse
import datetime

def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/100
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200    # neurons in the first layer
    params['second_layer_size'] = 20   # neurons in the second layer
    params['third_layer_size'] = 50    # neurons in the third layer
    params['episodes'] = 250          
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['train'] = True
    params["test"] = False
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params


if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()

    args = parser.parse_args()
    print("Args", args)
    params['display'] = True
    params['train'] = True
    #params['speed'] = s.sargpeed
    if params['train']:
        print("Training...")
        
        params['load_weights'] = False   # when training, the network is not pre-trained
        run(params)
    if params['test']:
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True
        run(params)


def run():
    menu(death_count=0)