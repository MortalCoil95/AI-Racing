import math
import random
import sys
import os


import neat 
import pygame

from utils import scale_image

TRACK = scale_image(pygame.image.load(r"C:\Users\snehi\Desktop\RacingCar\track1.png"),1.1)

#FINISH = scale_image(pygame.image.load(r"C:\Users\snehi\Desktop\RacingCar\finish.png"), 0.5)

CAR = scale_image(pygame.image.load(r"C:\Users\snehi\Desktop\RacingCar\red-car.png"),0.05)

CAR_SIZE_X = CAR.get_width()
CAR_SIZE_Y = CAR.get_height()

BORDER = (0, 0, 0)
    
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()

current_generation = 0 

class AbstractCar:
    def __init__(self):
        
        self.CAR = scale_image(pygame.image.load(r"C:\Users\snehi\Desktop\RacingCar\red-car.png"),0.05)
        self.rotated_CAR = self.CAR
        
        self.START_POS = [370, 390]
        self.angle = 300
        self.vel = 0
        
        
        self.speed_set = False

        self.center = [self.START_POS[0]+ CAR_SIZE_X/2, self.START_POS[1] + CAR_SIZE_Y/2]

        self.radars = []
        self.drawing_radars = []

        self.alive = True

        self.distance = 0
        self.time = 0
    
    def draw(self, WIN):
        WIN.blit(self.rotated_CAR, self.START_POS)
        self.draw_radar(WIN)
    
    def draw_radar(self, WIN):
        for radar in self.radars:
            POS = radar[0]
            pygame.draw.line(WIN, (0,225,0), self.center, POS,1)
            pygame.draw.circle(WIN, (255,255,255), POS, 5)

    
    def collide(self, TRACK):
        self.alive = True
        for point in self.corners:
            
            if TRACK.get_at((int(point[0]), int(point[1]))) == BORDER:
                self.alive = False
                break

    def check_radar(self,degree,TRACK):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360-(self.angle+degree)))* length)
        y = int(self.center[1] + math.cos(math.radians(360-(self.angle+degree)))* length)

        while not TRACK.get_at((x, y)) == BORDER and length < 300 :
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360-(self.angle+degree)))* length)
            y = int(self.center[1] + math.cos(math.radians(360-(self.angle+degree)))* length)
        
        dist = int(math.sqrt(math.pow(x - self.center[0], 2))) + math.pow(y - self.center[1],2)
        self.radars.append([(x,y),dist])
    
    
    

    def update(self,TRACK):

        if not self.speed_set:
            self.vel = 20
            self.speed_set = True
        
        self.rotated_Car = self.rotate_center(self.CAR,self.angle)
        self.START_POS[0] += (math.cos(math.radians(360 - self.angle))*self.vel)
        self.START_POS[0] = max(self.START_POS[0], 20) 
        self.START_POS[0] = min(self.START_POS[0], WIDTH - 120)

        self.distance += self.vel
        self.time += 1

        self.START_POS[1] += math.sin(math.radians(360 - self.angle))*self.vel
        self.START_POS[1] = max(self.START_POS[0], 20) 
        self.START_POS[1] = min(self.START_POS[0], WIDTH - 120)

        self.center = [int(self.START_POS[0]) + CAR_SIZE_X/2, int(self.START_POS[1])+ CAR_SIZE_Y/2]

        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.collide(TRACK)
        self.radars.clear()

        for d in range(-90, 120, 45):
            self.check_radar(d,TRACK)

    def get_data(self):
        radars = self.radars
        return_values = [0,0,0,0,0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1]/30)
        
        return return_values    

    def is_alive(self):
        return self.alive
        
    def get_reward(self):
        return self.distance/(CAR_SIZE_X/2)
    
    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

def run_simulation(genomes,config):

    nets=[]
    cars=[]

    pygame.init()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        g.fitness = 0

        cars.append(AbstractCar())
    
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    TRACK = scale_image(pygame.image.load(r"C:\Users\snehi\Desktop\RacingCar\track1.png"),1.1)
    
    global current_generation
    current_generation += 1

    counter = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10
            elif choice == 1:
                car.angle -= 10
            elif choice == 2:
                if(car.vel - 2>=12):
                    car.vel -= 2 
            else:
                car.vel += 2
        
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(TRACK)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break        

        counter += 1
        if counter == 30*40:
            break

        WIN.blit(TRACK, (0,0))
        for car in cars:
            if car.is_alive():
               car.draw(WIN)
            

        text = generation_font.render("Generation: " + str(current_generation), True, (255,255,255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        WIN.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (255,255,255))
        text_rect = text.get_rect()
        text_rect.center = (150, 100)
        WIN.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    
    # Load Config
    config_path =r"C:\Users\snehi\Desktop\RacingCar\config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Run Simulation For A Maximum of 1000 Generations
    population.run(run_simulation, 1000)

