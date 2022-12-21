# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:47:10 2022

@author: Amine
"""

import gym
from gym import spaces
import numpy as np
import cv2
import time

FPS = 45
SCALE = 5.0
RADIUS = 4*SCALE
WIDTH = 160*SCALE
HEIGHT = 80*SCALE
MAX_VELOCITY = WIDTH*0.5
CAGE = 15*SCALE

REWARD_STEP = -1
REWARD_AWAY = -5
REWARD_COLLIDE_IN = -25
REWARD_COLLIDE_OUT = 25
REWARD_BLOCKED = 100
REWARD_GOAL = -100

class Mallet():
    def __init__(self,x,y,radius):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.mass = 500
        self.radius = radius
    
    def setVelocity(self, velocity):
        self.vx += velocity[0]
        self.vy += velocity[1]
        
    def setPosition(self, position):
        self.x = position[0]
        self.y = position[1]
    
    def step(self, dt):
        self.vx = self.vx
        self.vy = self.vy
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
    
    def getPosition(self):
        return np.array([self.x,self.y])
    
    def getVelocity(self):
        return np.array([self.vx,self.vy])

class Puck():
    def __init__(self,x,y,radius):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.mass = 350
        self.radius = radius
    
    def setVelocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]
    
    def setPosition(self, position):
        self.x = position[0]
        self.y = position[1]
    
    def step(self, dt):
        self.vx = self.vx - 0.25*self.vx * dt
        self.vy = self.vy - 0.25*self.vy * dt
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
        
        if self.x >= WIDTH/2 - self.radius:
            self.vx = -0.99*self.vx
            self.vy = 0.99*self.vy
            
            dx = self.x - (WIDTH/2 - self.radius)
            self.x = self.x - 2*dx
        
        elif self.x <= -WIDTH/2 + self.radius:
            self.vx = -0.99*self.vx
            self.vy = 0.99*self.vy
            
            dx = self.x + (WIDTH/2 - self.radius)
            self.x = self.x - 2*dx
        
        if self.y >= HEIGHT/2 - self.radius:
            self.vx = 0.99*self.vx
            self.vy = -0.99*self.vy
            
            dy = self.y - (HEIGHT/2 - self.radius)
            self.y = self.y - 2*dy
        
        elif self.y <= -HEIGHT/2 + self.radius:
            self.vx = 0.99*self.vx
            self.vy = -0.99*self.vy
            
            dy = self.y - (-HEIGHT/2 + self.radius)
            self.y = self.y - 2*dy
        
        if (self.x <= -WIDTH/2 + self.radius+1) and (abs(self.y) <= CAGE):
            goal = True
            return goal
    
    def getPosition(self):
        return np.array([self.x,self.y])
    
    def getVelocity(self):
        return np.array([self.vx,self.vy])

class AirHockey(gym.Env):
    
    metadata = {'render.modes': ['human',None]}

    def __init__(self,max_steps=200,render_mode='human'):
        super(AirHockey, self).__init__()
        self.render_mode = render_mode
        
        self.steps = 0
        self.max_steps = max_steps
        self.puck = Puck(0,0,RADIUS)
        self.mallet = Mallet(-60*SCALE,0,RADIUS)
        cv2.namedWindow(winname='AirHockey')
        low = np.array(
            [   
                #for mallet :
                #positional bounds
                -HEIGHT/2.0,
                -WIDTH/2.0,
                # velocity bounds
                -MAX_VELOCITY*2.0,
                -MAX_VELOCITY*2.0,
                #for puck :
                -HEIGHT/2.0,
                -WIDTH/2.0,
                -MAX_VELOCITY*2.0,
                -MAX_VELOCITY*2.0,
            ]
        ).astype(np.float32)
        
        high = np.array(
           [
                #for mallet :
                #positional bounds
                HEIGHT/2.0,
                WIDTH/2.0,
                # velocity bounds
                MAX_VELOCITY*2.0,
                MAX_VELOCITY*2.0,
                #for puck :
                HEIGHT/2.0,
                WIDTH/2.0,
                MAX_VELOCITY*2.0,
                MAX_VELOCITY*2.0,
           ]
        ).astype(np.float32)
            
        self.observation_space = spaces.Box(low, high)
        
        self.action_space = spaces.Box(-MAX_VELOCITY, +MAX_VELOCITY, (2,), dtype=np.float32)
    
    def checkCollision(self,puck,mallet):
        Ppi = np.array([puck.x, puck.y])
        Pmi = np.array([mallet.x, mallet.y])
        
        direction = Ppi - Pmi
        d = np.linalg.norm(direction)
        if d==0 or d>puck.radius+mallet.radius:
            return 0
        direction = direction/d
        
        corr = (puck.radius+mallet.radius - d)/2.0
        puck.setPosition(puck.getPosition() + direction*corr)
        mallet.setPosition(mallet.getPosition() - direction*corr)
        
        v1 = np.dot(mallet.getVelocity(),direction)
        v2 = np.dot(puck.getVelocity(),direction)
        
        m1 = mallet.mass
        m2 = puck.mass
        
        newV1 = (m1 * v1 + m2 * v2 - m2 * (v1 - v2)*0)/(m1 + m2)
        newV2 = (m1 * v1 + m2 * v2 - m1 * (v2 - v1)*0)/(m1 + m2)
        
        puck.setVelocity(direction*(newV2-v2))
        mallet.setVelocity(direction*(newV1-v1))
        
        return 1
    
    def updateState(self):
        px, py = self.puck.getPosition()
        vpx, vpy = self.puck.getVelocity()
        
        mx, my = self.mallet.getPosition()
        vmx, vmy = self.mallet.getVelocity()
        
        self.state = [mx,my,vmx,vmy,px,py,vpx,vpy]
    
    def step(self, action):
        '''
        action : liste de type [Vx,Vy]
        
        '''
        info = {}
        
        
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    done = True
        
        self.render(self.render_mode)
        
        
        
        
        done = False
        action = np.clip(action, -MAX_VELOCITY, +MAX_VELOCITY).astype(np.float32)
        reward = REWARD_STEP
        self.steps -=1
        dt = 1/FPS
        
        precedent_delta = np.linalg.norm(self.puck.getPosition() - self.mallet.getPosition())
        
        
        self.mallet.setVelocity(action)
        self.mallet.step(dt)
        goal = self.puck.step(dt)
        
        if goal:
            reward -= REWARD_GOAL
            done = True
            print("GOAL")
        
        
        
        collide = self.checkCollision(self.puck,self.mallet)
        if(collide==1):
            reward += REWARD_COLLIDE
            
        
        reward += 0.01 * (np.linalg.norm(self.puck.getPosition() - self.mallet.getPosition()) - precedent_delta)
        
        
        self.updateState()
        
        return self.state, reward, done, info
            
    def reset(self):
        self.img = np.zeros((int(WIDTH),int(HEIGHT),3),dtype='uint8')
        P=(np.random.rand(2)-0.5)*HEIGHT*0.8
        self.puck = Puck(P[0],P[1],RADIUS)
        self.mallet = Mallet(-70*SCALE,0,RADIUS)
        
        self.puck.setVelocity((np.random.randint(-500,-200), -P[1]*0.5))
        
        px, py = self.puck.getPosition()
        vpx, vpy = self.puck.getVelocity()
        
        mx, my = self.mallet.getPosition()
        vmx, vmy = self.mallet.getVelocity()
        
        self.state = [mx,my,vmx,vmy,px,py,vpx,vpy]
        
        self.steps = 200
        self._max_episode_steps = 200
        
        return self.state
        
    def render(self, mode='human', close=False):

        if mode == 'human':
            # Visualisation de l'environnement
            
            self.img = np.ones((int(HEIGHT),int(WIDTH),3),dtype='uint8')*255
            cv2.rectangle(self.img, (0,0), (int(WIDTH)-1,int(HEIGHT)-1), (0,0,255), 6)
            cv2.line(self.img, (0,int(HEIGHT/2-CAGE)), (0,int(HEIGHT/2+CAGE)), (255,0,0), 6)
            cv2.line(self.img, (int(WIDTH/2),0), (int(WIDTH/2),int(HEIGHT)), (0,0,255), 3)
            cv2.circle(self.img, (int(WIDTH/2),int(HEIGHT/2)),int(SCALE*10), (255,255,255), -1)
            cv2.circle(self.img, (int(WIDTH/2),int(HEIGHT/2)),int(SCALE*10), (0,0,255), 4)
            cv2.circle(self.img, (int(WIDTH/2),int(HEIGHT/2)),int(SCALE*1), (0,0,255), -1)
            
            #self.img = np.zeros((WIDTH,HEIGHT,3),dtype='uint8')
            
            posXp= (int(self.puck.getPosition()[0]+WIDTH/2))
            posYp=(-int(self.puck.getPosition()[1]-HEIGHT/2))
            posXm=(int(self.mallet.getPosition()[0]+WIDTH/2))
            posYm=(-int(self.mallet.getPosition()[1]-HEIGHT/2))
            
            
            cv2.circle(self.img,(posXp,posYp),int(self.puck.radius),(0,0,0),-1)
            cv2.circle(self.img,(posXm,posYm),int(self.mallet.radius),(255,0,0),-1)
            cv2.imshow('AirHockey',self.img)
            time.sleep(1/FPS)
        
        elif mode == None:
            pass
        pass
    
    
    

steps = 200
episodes = 5

env = AirHockey(100,'human')

for episode in range(episodes):
  state = env.reset()
  done = False
  score = 0
  print("Etat initial : ", state)

  for steps in range(steps):
    action = env.action_space.sample()
    if steps < 200:
        action = np.array([np.random.randint(-4,5),np.random.randint(-5,5)])
    else : 
        action = np.array([0,0])
    n_state, reward, done, info = env.step(action)
    score += reward
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
    if done == True:
        print(done)
        break

    #print("Action jouée : ", action, "|| Nouvel état : ", n_state, " || reward associé : ", reward, " || Score total : ", score) 

cv2.destroyAllWindows()
