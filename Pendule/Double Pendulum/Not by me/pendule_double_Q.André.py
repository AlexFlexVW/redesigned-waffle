import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt

# PARAMETRES
m = 1
g = 9.81
l = 0.25
#
time_stop = 10
dt = 1.e-3

# CONDITIONS INITIALES
th1 = np.pi 
th2 = 0.0
thp1 = 0.1
thp2 = 0.0

# FONCTIONS POUR THETA DOT DOT
def theta_pp_1(g, l, th1, th2, thp1, thp2):
  delta = th1 - th2

  res = -1.0 / ( (4./3) * l - (3./4) * l * np.cos(delta)**2 )
  res *= thp1**2 * (3./4) * l * np.cos(delta) * np.sin(delta) + thp2**2 * (1./2) * l * np.sin(delta) + (3./2) * g * (np.sin(th1) - (1./2) * np.sin(th2) * np.cos(delta))

  return res
#
def theta_pp_2(g, l, th1, th2, thp1, thp2):
  delta = th1 - th2

  res = -1.0 / ( (1./3) * l - (3./16) * l * np.cos(delta)**2 ) 
  res *= - thp1**2 * (1./2) * l * np.sin(delta) - thp2**2 * (3./16) * l * np.cos(delta) * np.sin(delta) + (1./2) * g * (np.sin(th2) - (9./8) * np.sin(th1) * np.cos(delta))

  return res

###################
# BOUCLE TEMPORELLE
time = 0.0

times = [time]
th1s = [th1]
th2s = [th2]
thp1s = [thp1]
thp2s = [thp2]

while time < time_stop:
  thp1 += dt * theta_pp_1(g, l, th1, th2, thp1, thp2)
  thp2 += dt * theta_pp_2(g, l, th1, th2, thp1, thp2)
  #
  th1 += dt * thp1
  th2 += dt * thp2
  #
  time += dt
  #
  times.append(time)
  th1s.append(th1)
  th2s.append(th2)
  thp1s.append(thp1)
  thp2s.append(thp2)




###################
# INITIALISATION PYGAME
pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pendule double – vitesse physique")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (200, 0, 0)
BLUE  = (0, 0, 200)

clock = pygame.time.Clock()

# Origine du pendule (point d'attache)
origin = np.array([WIDTH // 2, HEIGHT // 4])

# Paramètres graphiques
scale = 300      # pixels par mètre
r1, r2 = 8, 8    # rayons des masses

# Police pour affichage des vitesses
font = pygame.font.SysFont("Arial", 18)

###################
# BOUCLE D'ANIMATION
running = True
t_visu = 0.0
i = 0
N = len(th1s)

while running and i < N - 1:

    # Temps réel écoulé (s)
    dt_visu = clock.tick(60) / 1000 / 2  # divisé par 2 pour ralentir l'animation
    t_visu += dt_visu

    # Correspondance temps réel ↔ simulation
    i = int(t_visu / dt)
    if i >= N:
        break

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Angles et vitesses à l'instant i
    th1 = th1s[i]
    th2 = th2s[i]
    thp1 = thp1s[i]
    thp2 = thp2s[i]

    # Positions physiques
    x1 = l * np.sin(th1)
    y1 = l * np.cos(th1)

    x2 = x1 + l * np.sin(th2)
    y2 = y1 + l * np.cos(th2)

    # Conversion en pixels
    p1 = origin + scale * np.array([x1, y1])
    p2 = origin + scale * np.array([x2, y2])

    p1 = p1.astype(int)
    p2 = p2.astype(int)

    # AFFICHAGE
    screen.fill(WHITE)

    # Tiges
    pygame.draw.line(screen, BLACK, origin, p1, 2)
    pygame.draw.line(screen, BLACK, p1, p2, 2)

    # Masses
    pygame.draw.circle(screen, RED, p1, r1)
    pygame.draw.circle(screen, BLUE, p2, r2)

    # Affichage des vitesses angulaires
    txt1 = font.render(f"thp1 = {thp1:.2f} rad/s", True, BLACK)
    txt2 = font.render(f"thp2 = {thp2:.2f} rad/s", True, BLACK)
    screen.blit(txt1, (20, 20))
    screen.blit(txt2, (20, 45))

    pygame.display.flip()

pygame.quit()
sys.exit()