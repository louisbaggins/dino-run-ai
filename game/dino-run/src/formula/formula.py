#!/usr/bin/python3
import pygame
import os
import random
pygame.init()

# Global Constants
pygame.display.set_caption('Dino Run')
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
RUNNING = [pygame.image.load("images/Dino/DinoRun1.png"),
           pygame.image.load("images/Dino/DinoRun2.png")]
JUMPING = pygame.image.load("images/Dino/DinoJump.png")
DUCKING = [pygame.image.load("images/Dino/DinoDuck1.png"),
           pygame.image.load("images/Dino/DinoDuck2.png")]
SMALL_CACTUS = [pygame.image.load("images/Cactus/SmallCactus1.png"),
                pygame.image.load("images/Cactus/SmallCactus2.png"),
                pygame.image.load("images/Cactus/SmallCactus3.png")]
LARGE_CACTUS = [pygame.image.load("images/Cactus/LargeCactus1.png"),
                pygame.image.load("images/Cactus/LargeCactus2.png"),
                pygame.image.load("images/Cactus/LargeCactus3.png")]
BIRD = [pygame.image.load("images/Bird/Bird1.png"),
        pygame.image.load("images/Bird/Bird2.png")]
CLOUD = pygame.image.load("images/Other/Cloud.png")
MUSHROOM = pygame.image.load("images/Powerup/mush.png")
BIGJUMPING = pygame.image.load("images/BigDino/DinoJump.png")
BIGRUNNING = [pygame.image.load("images/BigDino/DinoRun1.png"),
           pygame.image.load("images/BigDino/DinoRun2.png")]
BIGDUCKING = [pygame.image.load("images/BigDino/DinoDuck1.png"),
           pygame.image.load("images/BigDino/DinoDuck2.png")]
BG = pygame.image.load("images/Other/Track.png")

# Commands
print("")
print("\033[36m📚 HOW TO PLAY?\033[0m")
print("\033[32m🟢 Start moving T-Rex with ANY KEY \033[0m")
print("\033[38;5;214m🟠 Play using UP KEY 🔼 and DOWN KEY 🔽 \033[0m")
print("\033[31m🔴 Press the \"ESCAPE\" KEY on the Dino Run \"GAME OVER\" screen to end the game! \033[0m")
print("")

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

class BigDinosaur:
    X_POS = 80
    Y_POS = 280
    Y_POS_DUCK = 325
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

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)
    
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
        # textClock = font.render("Speed: " + str(speed), True, (0, 0, 0))
        # textClockRect = text.get_rect()
        # textClockRect.center = (100, 250)
        # SCREEN.blit(textClock, textClockRect)

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        SCREEN.fill((255, 255, 255))
        userInput = pygame.key.get_pressed()

        player.draw(SCREEN)
        player.update(userInput)

        if(powerupOn): #    Quando muda o dinossauro pequeno para o grande
            playerBig.dino_rect.x = player.dino_rect.x
            playerBig.dino_rect.y = player.dino_rect.y
            player = playerBig
            
        if(powerupOn) and (pygame.time.get_ticks() - time > 3000): #    Quando o dinossauro volta ao normal
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
            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(2000)
                death_count += 1
                menu(death_count)
        background()
        showGameSpeed(game_speed)

        cloud.draw(SCREEN)
        cloud.update()

        score()

        clock.tick(30)

        pygame.display.update()

def replay_or_quit():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                print("➡️  Thank you for using Ritchie CLI! 🆒")
                pygame.quit()
                quit()
        elif event.type == pygame.KEYUP:
            main()

def menu(death_count):
    global points
    run = True
    while run:
        SCREEN.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 30)

        if death_count == 0:
            text = font.render("Press any Key to Start", True, (0, 0, 0))
        elif death_count > 0:
            text = font.render("Press ESC to Exit or other Key to Restart", True, (0, 0, 0))
            score = font.render("Your Score: " + str(points), True, (0, 0, 0))
            scoreRect = score.get_rect()
            scoreRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
            SCREEN.blit(score, scoreRect)
        textRect = text.get_rect()
        textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        SCREEN.blit(text, textRect)
        SCREEN.blit(RUNNING[0], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 140))
        pygame.display.update()
        replay_or_quit()

def run():
    menu(death_count=0)