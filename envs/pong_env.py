import sys
import random
import pygame
import numpy as np
from pygame.locals import *
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class Pong2PEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 screen_size=(400, 300),
                 bat_height=50,
                 ball_speed=2,
                 bat_speed=2,
                 max_round=20):
        self._screen_width, self._screen_height = screen_size
        self.observation_space = spaces.Tuple([
            spaces.Box(
                low=0,
                high=255,
                shape=(self._screen_height, self._screen_width, 3)),
            spaces.Box(
                low=0,
                high=255,
                shape=(self._screen_height, self._screen_width, 3))
        ])
        self.action_space = spaces.Tuple(
            [spaces.Discrete(3), spaces.Discrete(3)])

        pygame.init()
        self._surface = pygame.Surface((self._screen_width,
                                        self._screen_height))

        self._game = PongGame(
            mode='double',
            window_size=screen_size,
            bat_height=bat_height,
            ball_speed=ball_speed,
            bat_speed=bat_speed,
            max_round=max_round)
        self._viewer = None

    def _step(self, action):
        assert self.action_space.contains(action)
        left_player_action, right_player_action = action
        bat_directions = [-1, 0, 1]
        rewards, done = self._game.step(bat_directions[left_player_action],
                                        bat_directions[right_player_action])
        obs = self._get_screen_img_double_player()
        return (obs, rewards, done, {})

    def _reset(self):
        self._game.restart()
        obs = self._get_screen_img_double_player()
        return obs

    def _render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            pygame.quit()
            return
        img = self._get_screen_img()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img)

    def _get_screen_img_double_player(self):
        self._game.draw(self._surface)
        surface_flipped = pygame.transform.flip(self._surface, True, True)
        self._game.draw_scoreboard(self._surface)
        self._game.draw_scoreboard(surface_flipped)
        obs = self._surface_to_img(self._surface)
        obs_flip = self._surface_to_img(surface_flipped)
        return obs, obs_flip

    def _get_screen_img(self):
        self._game.draw(self._surface)
        self._game.draw_scoreboard(self._surface)
        obs = self._surface_to_img(self._surface)
        return obs

    def _surface_to_img(self, surface):
        img = pygame.surfarray.array3d(surface).astype(np.uint8)
        return np.transpose(img, (1, 0, 2))


class Pong1PEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 screen_size=(400, 300),
                 bat_height=50,
                 ball_speed=2,
                 bat_speed=2,
                 max_round=20):
        self._screen_width, self._screen_height = screen_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._screen_height, self._screen_width, 3))
        self.action_space = spaces.Discrete(3)

        pygame.init()
        self._surface = pygame.Surface((self._screen_width,
                                        self._screen_height))

        self._game = PongGame(
            mode='single',
            window_size=screen_size,
            bat_height=bat_height,
            ball_speed=ball_speed,
            bat_speed=bat_speed,
            max_round=max_round)
        self._viewer = None

    def _step(self, action):
        assert self.action_space.contains(action)
        bat_directions = [-1, 0, 1]
        rewards, done = self._game.step(bat_directions[action], None)
        obs = self._get_screen_img()
        return (obs, rewards[0], done, {})

    def _reset(self):
        self._game.restart()
        obs = self._get_screen_img()
        return obs

    def _render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            pygame.quit()
            return
        img = self._get_screen_img()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img)

    def _get_screen_img(self):
        self._game.draw(self._surface)
        self._game.draw_scoreboard(self._surface)
        obs = self._surface_to_img(self._surface)
        return obs

    def _surface_to_img(self, surface):
        img = pygame.surfarray.array3d(surface).astype(np.uint8)
        return np.transpose(img, (1, 0, 2))


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class PongGame():
    def __init__(self,
                 mode,
                 window_size=(400, 300),
                 line_thickness=10,
                 bat_height=50,
                 ball_speed=5,
                 bat_speed=5,
                 max_round=20):
        self._mode = mode
        self._max_round = max_round
        self._score_left, self._score_right = 0, 0

        self._arena = Arena(window_size, line_thickness)
        self._ball = Ball(self._arena.centerx - line_thickness // 2,
                          self._arena.centery - line_thickness // 2,
                          line_thickness, ball_speed)
        self._left_bat = Bat(30, self._arena.centery - bat_height // 2,
                             line_thickness, bat_height, bat_speed)
        if mode == 'double':
            self._right_bat = Bat(self._arena.right - line_thickness - 30,
                                  self._arena.centery - bat_height // 2,
                                  line_thickness, bat_height, bat_speed)
        elif mode == 'single':
            self._right_bat = AutoBat(self._arena.right - line_thickness - 30,
                                      self._arena.centery - bat_height // 2,
                                      line_thickness, bat_height, bat_speed)
        else:
            raise ArgmentException("Game mode [%s] is not supported.")
        self._scoreboard = Scoreboard(self._arena.centerx + 25,
                                      self._arena.top + 25)

    def step(self, left_bat_dir, right_bat_dir):
        self._ball.move(self._arena, self._left_bat, self._right_bat)
        self._left_bat.move(left_bat_dir, self._arena)
        if self._mode == 'double':
            self._right_bat.move(right_bat_dir, self._arena)
        elif self._mode == 'single':
            self._right_bat.move(self._arena, self._ball)

        if self._ball.out_of_arena(self._arena) == 1:
            self._score_right += 1
            rewards = (-1, 1)
            self._reset()
        elif self._ball.out_of_arena(self._arena) == 2:
            self._score_left += 1
            rewards = (1, -1)
            self._reset()
        else:
            rewards = (0, 0)

        if self._score_right + self._score_left >= self._max_round:
            done = True
        else:
            done = False
        return rewards, done

    def _reset(self):
        self._ball.reset()
        self._left_bat.reset()
        self._right_bat.reset()

    def restart(self):
        self._score_left, self._score_right = 0, 0
        self._reset()

    def draw(self, surface):
        self._arena.draw(surface)
        self._ball.draw(surface)
        self._left_bat.draw(surface)
        self._right_bat.draw(surface)

    def draw_scoreboard(self, surface):
        self._scoreboard.draw(surface, self._score_left, self._score_right)


class Arena(pygame.sprite.Sprite):
    def __init__(self, window_size, border_thickness):
        self._window_width, self._window_height = window_size
        self._border_thickness = border_thickness
        self._rect = pygame.Rect(
            self._border_thickness, self._border_thickness,
            self._window_width - 2 * self._border_thickness,
            self._window_height - 2 * self._border_thickness)

    def draw(self, surface):
        surface.fill(WHITE)
        pygame.draw.rect(surface, BLACK, self._rect)
        pygame.draw.line(surface, WHITE, (self._window_width // 2, 0),
                         (self._window_width // 2, self._window_height),
                         (self._border_thickness // 4))

    @property
    def left(self):
        return self._rect.left

    @property
    def right(self):
        return self._rect.right

    @property
    def top(self):
        return self._rect.top

    @property
    def bottom(self):
        return self._rect.bottom

    @property
    def centerx(self):
        return self._rect.centerx

    @property
    def centery(self):
        return self._rect.centery


class Ball(pygame.sprite.Sprite):
    def __init__(self, x, y, size, speed):
        self._x_init, self._y_init = x, y
        self._speed = speed
        self._dir_x = random.uniform(-1, 1)
        self._dir_y = random.uniform(-1, 1)
        self._rect = pygame.Rect(x, y, size, size)

    def reset(self):
        self._rect.x = self._x_init
        self._rect.y = self._y_init
        self._dir_x = random.uniform(-1, 1)
        self._dir_y = random.uniform(-1, 1)

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self._rect)

    def move(self, arena, left_bat, right_bat):
        self._rect.x += (self._dir_x * self._speed)
        self._rect.y += (self._dir_y * self._speed)

        if self._dir_y < 0 and self._rect.top <= arena.top:
            self._bounce('y')
            self._rect.top = arena.top
        if self._dir_y > 0 and self._rect.bottom >= arena.bottom:
            self._bounce('y')
            self._rect.top = arena.bottom

        if (self._dir_x < 0 and self._rect.left <= left_bat.right and
                self._rect.bottom >= left_bat.top and
                self._rect.top <= left_bat.bottom):
            self._bounce('x')
            self._rect.left = left_bat.right
        if (self._dir_x > 0 and self._rect.right >= right_bat.left and
                self._rect.bottom >= right_bat.top and
                self._rect.top <= right_bat.bottom):
            self._bounce('x')
            self._rect.right = right_bat.left

    def out_of_arena(self, arena):
        if self._rect.left < arena.left:
            return 1  # left out
        elif self._rect.right > arena.right:
            return 2  # right out
        else:
            return 0  # not out

    def _bounce(self, axis):
        if axis == 'x':
            self._dir_x *= -1
        elif axis == 'y':
            self._dir_y *= -1

    @property
    def dir_x(self):
        return self._dir_x

    def dir_y(self):
        return self._dir_y

    @property
    def centerx(self):
        return self._rect.centery

    @property
    def centery(self):
        return self._rect.centery


class Bat(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, speed):
        self._x_init, self._y_init = x, y
        self._speed = speed
        self._rect = pygame.Rect(x, y, width, height)

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self._rect)

    def move(self, direction, arena):
        self._rect.y += direction * self._speed
        if self._rect.bottom > arena.bottom:
            self._rect.y += arena.bottom - self._rect.bottom
        elif self._rect.top < arena.top:
            self._rect.y += arena.top - self._rect.top

    def reset(self):
        self._rect.x = self._x_init
        self._rect.y = self._y_init

    @property
    def left(self):
        return self._rect.left

    @property
    def right(self):
        return self._rect.right

    @property
    def top(self):
        return self._rect.top

    @property
    def bottom(self):
        return self._rect.bottom


class AutoBat(Bat):
    def move(self, arena, ball):
        #If ball is moving away from paddle, center bat
        if ball.dir_x < 0:
            if self._rect.centery < arena.centery:
                self._rect.y += self._speed
            elif self._rect.centery > arena.centery:
                self._rect.y -= self._speed
        #if ball moving towards bat, track its movement. 
        elif ball.dir_x > 0:
            if self._rect.centery < ball.centery:
                self._rect.y += self._speed
            else:
                self._rect.y -= self._speed
        if self._rect.bottom > arena.bottom:
            self._rect.y += arena.bottom - self._rect.bottom
        elif self._rect.top < arena.top:
            self._rect.y += arena.top - self._rect.top


class Scoreboard():
    def __init__(self, x, y, font_size=20):
        self._x = x
        self._y = y
        self._font = pygame.font.Font('freesansbold.ttf', font_size)

    def draw(self, surface, score_left, score_right):
        result_surf = self._font.render('Score = %d : %d' %
                                        (score_left, score_right), True, WHITE)
        rect = result_surf.get_rect()
        rect.topleft = (self._x, self._y)
        surface.blit(result_surf, rect)


#Main function
def main():
    pygame.init()
    pygame.display.set_caption('Pong')
    pygame.mouse.set_visible(0)  # make cursor invisible
    surface = pygame.display.set_mode((400, 300))
    fps_clock = pygame.time.Clock()

    game = PongGame(window_size=(400, 300), speed=4)

    while True:  #main game loop
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            # mouse movement commands
            #elif event.type == MOUSEMOTION:
            #game.bats['user'].move(event.pos)

        _, done = game.step(1, -1)
        if done:
            game.reset()
        game.draw(surface)
        pygame.display.update()
        fps_clock.tick(120)


if __name__ == '__main__':
    main()
