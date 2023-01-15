---
layout: post 
title: "Learning to use OpenAI Gym" 
date: 2023-01-13
--- 

<style>
div.warn {    
    background-color: #fcf2f2;
    border-color: #dFb5b4;
    border-left: 5px solid #dfb5b4;
    padding: 0.5em;
    }
 </style>

 <style>
div.info {    
    background-color:#D6EAF8;
    border-color: #3498DB;
    border-left: 5px solid #3498DB;
    padding: 0.5em;
    }
 </style>


<center><em>Reinforcement learning,<br>
    A path to machine control,<br>
    Through trial and error,<br>
    Optimizing the goal.<br><br>
    Rewards guide the way<br>
    A signal to the brain,<br>
    A system to improve,<br>
    A future to gain.<br><br>
    But, the journey is long, <br>
    And the road is steep,<br> 
    Only the strong survive,<br> 
    To reach the final leap.<br><br>
    So, let us strive, <br> 
    To make our algorithms thrive,<br> 
    Through reinforcement learning,<br> 
    We'll make machines come alive.<br>
</em></center><br>

Thank you, ChatGPT for this cute poem! Although, I'm not really sure how I feel about the last line haha. 

I decided to tackle some reinforcement learning exercises that I found in my textbook, and think I might make this a several part series. Although I have a summer’s worth of reinforcement learning experience, I kind of jumped right into it and skipped over all of the basics. It’s worthwhile to formally introduce myself to some of these concepts. So let’s get into it: 

<div class=info>
First, I’m not sure if anyone else has this problem, but I could spend hours trying to figure out why certain packages aren’t installing on my local computer. I think from now on if I have to use anything besides numpy, I’ll go to Google Colab.
</div><br>

In this first part, we make sure that all of the required libraries are installed and up to date, make sure our plots will be nicely formatted, animations can be made and figures saved—  I’m in a rhyming mood :) 


# Setup and Intro to OpenAI gym 



```python
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Is this notebook running on Colab?
IS_COLAB = "google.colab" in sys.modules


if IS_COLAB or IS_KAGGLE:
    !apt update && apt install -y libpq-dev libsdl2-dev swig xorg-dev xvfb
    %pip install -U tf-agents pyvirtualdisplay
    %pip install -U gym~=0.21.0
    %pip install -U gym[box2d,atari,accept-rom-license]

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```


```python
import gym
```
First we are going to create an environment with `make()`:  
For this example we are going to use the CartPole environment. This is a 2D simulation with a pole balanced on a cart. 


```python
env = gym.make('CartPole-v1',render_mode="rgb_array")
```
Initialize the environment using `reset()`. Returned is the first set of observations. For the CartPole environment, the observations of are the following four: 

<ol>
 <li>Cart’s horizontal position (0.0 = center) </li>
 <li>Cart’s velocity (>0 = right)</li>
 <li>Pole’s angle (0.0 = vertical) </li>
 <li>Pole’s angular velocity (>0 = clockwise)</li>
</ol>

![image]({{site.url}}/assets/images/OpenAIGym_files/diagram.png){: width="500" } 



```python
obs = env.reset()
```


```python
env.np_random.random(32)
obs
```




<blockquote> array([-0.2303003 , -0.9796561 ,  0.21622495,  1.1510738 ], dtype=float32) </blockquote>




```python
try:
  import pyvirtualdisplay
  display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
  print('worked')
except ImportError:
  pass
```

Use the `render()` function to show the environment. 

```python
img = env.render()
```


```python
img.shape
```




  <blockquote> (400, 600, 3) </blockquote>




```python
def plt_env(env):
  plt.figure(figsize=(5,4))
  img = env.render()
  plt.imshow(img)
  plt.axis("off")
  return img 
```


```python
plt_env(env)
plt.show()
```


    
![image]({{site.url}}/assets/images/OpenAIGym_files/OpenAIGym_11_0.png)
    

`env.action_space()` shows which actions are possible in the environment.
<em>Discrete(2)</em> means that there are two possible actions that the agent can take: 0 (accelerating left) or 1 (accelerating right). 


```python
env.action_space
```




  <blockquote> Discrete(2) </blockquote>


Just as an example, let’s accelerate the cart to the left by setting action  = 0. The `step()` function executes the action and returns four values: 

<dl>
  <dt><b>obs</b></dt>
    <dd>The newest observation. If we compare the two angular velocities <em>obs[2]</em> we will see that it increases, which means the pole is moving to the right, as we expect it to! 
    </dd>
  <dt><b>reward</b></dt>
    <dd>We want the episode to run for as long as possible, so we set the reward to 1 at each step. 
    </dd>
  <dt><b>done</b></dt>
    <dd>When the episode is over, done will be equal to TRUE. This will happen either when the angle of the pole falls below 0 (which means it falls off the screen) or we reach the end of the 200 steps, which means we have won. 
    </dd>
  <dt><b>info</b></dt>
    <dd>This provides extra information for training or debugging.</dd>
</dl>

<div class='info'> I am imagining that episodes in this context are akin to epochs. After going through all of the steps per epoch, the environment is reset and the agent’s reward is set to 0. </div><br>

```python
action = 0
stats=env.step(action)
obs = stats[0]
reward = stats[1]
done = stats[2]
info = stats[3]
```


```python
plt_env(env)
plt.show()
```


    
![image]({{site.url}}/assets/images/OpenAIGym_files/OpenAIGym_14_0.png)
    



```python
save_fig("cart_pole_plot")
```

   <blockquote> Saving figure cart_pole_plot<br>
    <p> <Figure size 432x288 with 0 Axes> </p>
    </blockquote>



```python
print(obs)
print(reward)
print(done)
print(info)
```

   <blockquote> 
   [-0.2303003  -0.9796561   0.21622495  1.1510738 ] <br>
    1.0 <br>
    True <br>
    False <br>
    </blockquote>



```python
if done:
  obs = env.reset()
```
To demonstrate how this works all together, let’s create a simple hard-coded policy that will accelerate the cart to the left when the pole is leaning towards and accelerate it to the right when the pole is leaning right. 

# Creating a Simple-Hardcoded Policy


```python
def basic_policy(obs):
  angle = obs[2]
  return 0 if angle < 0 else 1 
```


```python
import array
from numpy.core.memmap import dtype

env.np_random.random(32)
totals=[] 

for episode in range(500):
  episode_rewards = 0 
  obs = env.reset()
  obs = np.array(obs[0])

  for step in range(200):
    #action will either be 0 or 1 
    action = basic_policy(obs)
    #get all information from each action 
    stats = env.step(action)
    obs = np.array(stats[0])
    reward = stats[1]
    done = stats[2]
    info = stats[3]

    episode_rewards += reward 

    if done: 
       break
       
  totals.append(episode_rewards) 
```


```python
print('mean',np.mean(totals) ,np.std(totals),np.min(totals),np.max(totals))

```
The max value indicates that out of 500 tries, the pole managed to stay upright for 72 consecutive steps. 


   <blockquote>  mean: 42.992 <br>
                 std: 8.717105941767601 <br>
                 min: 24.0 <br>
                 max: 72.0</blockquote>


# Visualization

```python
env.np_random.random(32)

frames = [] 

obs = env.reset()
obs = np.array(obs[0])
for step in range(200):
  img = env.render()
  frames.append(img)
  action = basic_policy(obs)

  stats=env.step(action)
  obs = np.array(stats[0])
  reward = stats[1]
  done = stats[2]
  info = stats[3]

  if done: 
    break
```




```python
def update_scene(num,frames,patch): 
  patch.set_data(frames[num])
  return patch

def plot_animation(frames,repeat=False,interval=40):
  fig = plt.figure()
  patch = plt.imshow(frames[0])
  plt.axis('off')
  anim = animation.FuncAnimation(fig,update_scene, fargs=(frames,patch),
                                 frames=len(frames),repeat=repeat,interval=interval)
  
  plt.close()
  return anim 
```


```python
plot_animation(frames)
```





<link rel="stylesheet"
href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/
css/font-awesome.min.css">
<script language="javascript">
  function isInternetExplorer() {
    ua = navigator.userAgent;
    /* MSIE used to detect old browsers and Trident used to newer ones*/
    return ua.indexOf("MSIE ") > -1 || ua.indexOf("Trident/") > -1;
  }

  /* Define the Animation class */
  function Animation(frames, img_id, slider_id, interval, loop_select_id){
    this.img_id = img_id;
    this.slider_id = slider_id;
    this.loop_select_id = loop_select_id;
    this.interval = interval;
    this.current_frame = 0;
    this.direction = 0;
    this.timer = null;
    this.frames = new Array(frames.length);

    for (var i=0; i<frames.length; i++)
    {
     this.frames[i] = new Image();
     this.frames[i].src = frames[i];
    }
    var slider = document.getElementById(this.slider_id);
    slider.max = this.frames.length - 1;
    if (isInternetExplorer()) {
        // switch from oninput to onchange because IE <= 11 does not conform
        // with W3C specification. It ignores oninput and onchange behaves
        // like oninput. In contrast, Mircosoft Edge behaves correctly.
        slider.setAttribute('onchange', slider.getAttribute('oninput'));
        slider.setAttribute('oninput', null);
    }
    this.set_frame(this.current_frame);
  }

  Animation.prototype.get_loop_state = function(){
    var button_group = document[this.loop_select_id].state;
    for (var i = 0; i < button_group.length; i++) {
        var button = button_group[i];
        if (button.checked) {
            return button.value;
        }
    }
    return undefined;
  }

  Animation.prototype.set_frame = function(frame){
    this.current_frame = frame;
    document.getElementById(this.img_id).src =
            this.frames[this.current_frame].src;
    document.getElementById(this.slider_id).value = this.current_frame;
  }

  Animation.prototype.next_frame = function()
  {
    this.set_frame(Math.min(this.frames.length - 1, this.current_frame + 1));
  }

  Animation.prototype.previous_frame = function()
  {
    this.set_frame(Math.max(0, this.current_frame - 1));
  }

  Animation.prototype.first_frame = function()
  {
    this.set_frame(0);
  }

  Animation.prototype.last_frame = function()
  {
    this.set_frame(this.frames.length - 1);
  }

  Animation.prototype.slower = function()
  {
    this.interval /= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.faster = function()
  {
    this.interval *= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.anim_step_forward = function()
  {
    this.current_frame += 1;
    if(this.current_frame < this.frames.length){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.first_frame();
      }else if(loop_state == "reflect"){
        this.last_frame();
        this.reverse_animation();
      }else{
        this.pause_animation();
        this.last_frame();
      }
    }
  }

  Animation.prototype.anim_step_reverse = function()
  {
    this.current_frame -= 1;
    if(this.current_frame >= 0){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.last_frame();
      }else if(loop_state == "reflect"){
        this.first_frame();
        this.play_animation();
      }else{
        this.pause_animation();
        this.first_frame();
      }
    }
  }

  Animation.prototype.pause_animation = function()
  {
    this.direction = 0;
    if (this.timer){
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  Animation.prototype.play_animation = function()
  {
    this.pause_animation();
    this.direction = 1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function() {
        t.anim_step_forward();
    }, this.interval);
  }

  Animation.prototype.reverse_animation = function()
  {
    this.pause_animation();
    this.direction = -1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function() {
        t.anim_step_reverse();
    }, this.interval);
  }
</script>

<style>
.animation {
    display: inline-block;
    text-align: center;
}
input[type=range].anim-slider {
    width: 374px;
    margin-left: auto;
    margin-right: auto;
}
.anim-buttons {
    margin: 8px 0px;
}
.anim-buttons button {
    padding: 0;
    width: 36px;
}
.anim-state label {
    margin-right: 8px;
}
.anim-state input {
    margin: 0;
    vertical-align: middle;
}
</style>

<div class="animation">
  <img id="_anim_imge1d6ab7d88ab4343aa55e9874d6f8be1">
  <div class="anim-controls">
    <input id="_anim_slidere1d6ab7d88ab4343aa55e9874d6f8be1" type="range" class="anim-slider"
           name="points" min="0" max="1" step="1" value="0"
           oninput="anime1d6ab7d88ab4343aa55e9874d6f8be1.set_frame(parseInt(this.value));"> 
      </input>
    <div class="anim-buttons">
      <button onclick="anime1d6ab7d88ab4343aa55e9874d6f8be1.slower()"><i class="fa fa-minus"></i></button>
      <button onclick="anime1d6ab7d88ab4343aa55e9874d6f8be1.first_frame()"><i class="fa fa-fast-backward">
          </i></button>
      <button onclick="anime1d6ab7d88ab4343aa55e9874d6f8be1.previous_frame()">
          <i class="fa fa-step-backward"></i></button>
      <button onclick="anime1d6ab7d88ab4343aa55e9874d6f8be1.reverse_animation()">
          <i class="fa fa-play fa-flip-horizontal"></i></button>
      <button onclick="anime1d6ab7d88ab4343aa55e9874d6f8be1.pause_animation()"><i class="fa fa-pause">
          </i></button>
      <button onclick="anime1d6ab7d88ab4343aa55e9874d6f8be1.play_animation()"><i class="fa fa-play"></i>
          </button>
      <button onclick="anime1d6ab7d88ab4343aa55e9874d6f8be1.next_frame()"><i class="fa fa-step-forward">
          </i></button>
      <button onclick="anime1d6ab7d88ab4343aa55e9874d6f8be1.last_frame()"><i class="fa fa-fast-forward">
          </i></button>
      <button onclick="anime1d6ab7d88ab4343aa55e9874d6f8be1.faster()"><i class="fa fa-plus"></i></button>
    </div>
    <form action="#n" name="_anim_loop_selecte1d6ab7d88ab4343aa55e9874d6f8be1" class="anim-state">
      <input type="radio" name="state" value="once" id="_anim_radio1_e1d6ab7d88ab4343aa55e9874d6f8be1"
             checked>
      <label for="_anim_radio1_e1d6ab7d88ab4343aa55e9874d6f8be1">Once</label>
      <input type="radio" name="state" value="loop" id="_anim_radio2_e1d6ab7d88ab4343aa55e9874d6f8be1"
             >
      <label for="_anim_radio2_e1d6ab7d88ab4343aa55e9874d6f8be1">Loop</label>
      <input type="radio" name="state" value="reflect" id="_anim_radio3_e1d6ab7d88ab4343aa55e9874d6f8be1"
             >
      <label for="_anim_radio3_e1d6ab7d88ab4343aa55e9874d6f8be1">Reflect</label>
    </form>
  </div>
</div>


<script language="javascript">
  /* Instantiate the Animation class. */
  /* The IDs given should match those used in the template above. */
  (function() {
    var img_id = "_anim_imge1d6ab7d88ab4343aa55e9874d6f8be1";
    var slider_id = "_anim_slidere1d6ab7d88ab4343aa55e9874d6f8be1";
    var loop_select_id = "_anim_loop_selecte1d6ab7d88ab4343aa55e9874d6f8be1";
    var frames = new Array(45);

  frames[0] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAHWklEQVR4nO3dQW9c1RnH4feOY5KCaNUAwoAEVapK\
SJWadsMiEmuW5Guk60Z8im6zRSibLrqouuyaNlRdhgUKyiIFJAeEZeLgFCfx6cIRQjgSztw5Pvo7\
z2N5FtejmXcx0s9z55w7U2utFQCEWYweAACWIWAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkY\
AJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQS\
MAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACI\
JGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoAB\
EEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQKRT\
oweAp8G3X92qh3v3Dh1/9sU36tTpnw2YCPIJGHTWWqv//vMvdff2zUN/e/Pd9+r5jV8PmAryOYUI\
3bVqo0eAE0jAoLf2/Q2wQgIG3TX9gg4EDDprrZWCweoJGACRBAx6a+3gF1gpAYPurEKEHgQMOms/\
uAVWR8Cgt2YVIvQgYNCdVYjQg4BBbzYyQxcCBp01G5mhCwGD3lqzDhE6EDAAIgkY9GYjM3QhYNCd\
eEEPAgadHSxCFDFYNQGD3polHNCDgEF3NjJDDwIGvekXdCFg0FlTMOhCwKA3CzigCwGD7lo1EYOV\
EzDoTbugCwGDzg4u5qtisGoCBr2JF3QhYHAsRAxWTcCgNxfzhS4EDDpzISnoQ8CgN1/IDF0IGHTn\
FCL0IGDQnXhBDwIGnbXWqtr+Y/82HfMscJIIGHT23Z2vam/3m0PHT//8pVp/7hcDJoKTQcCgs7a/\
/9h3YNPiVE3T2oCJ4GQQMBhlKucQYQYBg0EmBYNZBAxGmcQL5hAwGEnEYGkCBqNMkxOIMIOAwTA+\
A4M5BAwGmWrSL5hBwGCU6fsbYAkCBsOIF8whYDDINE01WYUISxMwGErAYFkCBqN49wWzCBiMMk0i\
BjMIGAwjXjCHgMEgB6voRQyWJWAwiktJwSwCBsO4lBTMIWAwin7BLAIGg/hCS5hHwGAUCzhgFgGD\
YewDgzkEDEbxbSowi4DBID4Dg3kEDEaZvAWDOU6NHgAS7ezs1PXr149038Wdz2rtMce3t7fro4/+\
XTX99P+RGxsbde7cuSecEk62qbXWRg8Baa5du1YXLlw40n3f/t3r9ec/vnPo+L8+/qz+dOUf9WB/\
/ycf49KlS3XlypUnnhNOMu/A4Bjst0V9ufd6bd1/pc4svq3Xztyo9ugHWI6AQWetFnXz3u/r5u4f\
qtWiqlrd3vtV/e/h5/IFM1jEAZ19vffqo3it1cGqjUVtP3i5Prn7VikYLE/AoLP9WnsUrx+a6n5b\
dwoRZhAw6Gx9+q7Wpr0fHW11ZtopS6hgeQIGnf1yfbN++9yH9cx0r6pardX9evX0p/WbZ/8zejSI\
9tQs4tjc3Bw9AifI1tbWke/76edb9cFf36+dh3+vOw9erGcW9+qF9S/q9tfbR36M3d1dr2G62NjY\
GD3C0p6agF29enX0CJwgt27dOvJ9N7fu1t8+/GTW8924ccNrmC4uX748eoSl2cgMS3iSjcyrYCMz\
HOYzMAAiCRgAkQQMgEgCBkAkAQMgkoABEOmp2QcGq3T27Nm6ePHisT3f+fPnj+25IIV9YABEcgoR\
gEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJ\
GACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABE\
EjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBI/webOv/JParJywAAAABJRU5ErkJggg==\
"
  frames[1] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAHWklEQVR4nO3dQW9c1RnH4feOY5KCaNUAwoAEVapK\
SJWadsMiEmuW5Guk60Z8im6zRSibLrqouuyaNlRdhgUKyiIFJAeEZeLgFCfx6cIRQjgSztw5Pvo7\
z2N5FtejmXcx0s9z55w7U2utFQCEWYweAACWIWAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkY\
AJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQS\
MAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACI\
JGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoAB\
EEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQKRT\
oweAp8G3X92qh3v3Dh1/9sU36tTpnw2YCPIJGHTWWqv//vMvdff2zUN/e/Pd9+r5jV8PmAryOYUI\
3bVqo0eAE0jAoLf2/Q2wQgIG3TX9gg4EDDprrZWCweoJGACRBAx6a+3gF1gpAYPurEKEHgQMOms/\
uAVWR8Cgt2YVIvQgYNCdVYjQg4BBbzYyQxcCBp01G5mhCwGD3lqzDhE6EDAAIgkY9GYjM3QhYNCd\
eEEPAgadHSxCFDFYNQGD3polHNCDgEF3NjJDDwIGvekXdCFg0FlTMOhCwKA3CzigCwGD7lo1EYOV\
EzDoTbugCwGDzg4u5qtisGoCBr2JF3QhYHAsRAxWTcCgNxfzhS4EDDpzISnoQ8CgN1/IDF0IGHTn\
FCL0IGDQnXhBDwIGnbXWqtr+Y/82HfMscJIIGHT23Z2vam/3m0PHT//8pVp/7hcDJoKTQcCgs7a/\
/9h3YNPiVE3T2oCJ4GQQMBhlKucQYQYBg0EmBYNZBAxGmcQL5hAwGEnEYGkCBqNMkxOIMIOAwTA+\
A4M5BAwGmWrSL5hBwGCU6fsbYAkCBsOIF8whYDDINE01WYUISxMwGErAYFkCBqN49wWzCBiMMk0i\
BjMIGAwjXjCHgMEgB6voRQyWJWAwiktJwSwCBsO4lBTMIWAwin7BLAIGg/hCS5hHwGAUCzhgFgGD\
YewDgzkEDEbxbSowi4DBID4Dg3kEDEaZvAWDOU6NHgAS7ezs1PXr149038Wdz2rtMce3t7fro4/+\
XTX99P+RGxsbde7cuSecEk62qbXWRg8Baa5du1YXLlw40n3f/t3r9ec/vnPo+L8+/qz+dOUf9WB/\
/ycf49KlS3XlypUnnhNOMu/A4Bjst0V9ufd6bd1/pc4svq3Xztyo9ugHWI6AQWetFnXz3u/r5u4f\
qtWiqlrd3vtV/e/h5/IFM1jEAZ19vffqo3it1cGqjUVtP3i5Prn7VikYLE/AoLP9WnsUrx+a6n5b\
dwoRZhAw6Gx9+q7Wpr0fHW11ZtopS6hgeQIGnf1yfbN++9yH9cx0r6pardX9evX0p/WbZ/8zejSI\
9tQs4tjc3Bw9AifI1tbWke/76edb9cFf36+dh3+vOw9erGcW9+qF9S/q9tfbR36M3d1dr2G62NjY\
GD3C0p6agF29enX0CJwgt27dOvJ9N7fu1t8+/GTW8924ccNrmC4uX748eoSl2cgMS3iSjcyrYCMz\
HOYzMAAiCRgAkQQMgEgCBkAkAQMgkoABEOmp2QcGq3T27Nm6ePHisT3f+fPnj+25IIV9YABEcgoR\
gEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJ\
GACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABE\
EjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBI/webOv/JParJywAAAABJRU5ErkJggg==\
"
  frames[2] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAHWklEQVR4nO3dQW9c1RnH4feOY5KCaNUAwoAEVapK\
SJWadsMiEmuW5Guk60Z8im6zRSibLrqouuyaNlRdhgUKyiIFJAeEZeLgFCfx6cIRQjgSztw5Pvo7\
z2N5FtejmXcx0s9z55w7U2utFQCEWYweAACWIWAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkY\
AJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQS\
MAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACI\
JGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoAB\
EEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQKRT\
oweAp8G3X92qh3v3Dh1/9sU36tTpnw2YCPIJGHTWWqv//vMvdff2zUN/e/Pd9+r5jV8PmAryOYUI\
3bVqo0eAE0jAoLf2/Q2wQgIG3TX9gg4EDDprrZWCweoJGACRBAx6a+3gF1gpAYPurEKEHgQMOms/\
uAVWR8Cgt2YVIvQgYNCdVYjQg4BBbzYyQxcCBp01G5mhCwGD3lqzDhE6EDAAIgkY9GYjM3QhYNCd\
eEEPAgadHSxCFDFYNQGD3polHNCDgEF3NjJDDwIGvekXdCFg0FlTMOhCwKA3CzigCwGD7lo1EYOV\
EzDoTbugCwGDzg4u5qtisGoCBr2JF3QhYHAsRAxWTcCgNxfzhS4EDDpzISnoQ8CgN1/IDF0IGHTn\
FCL0IGDQnXhBDwIGnbXWqtr+Y/82HfMscJIIGHT23Z2vam/3m0PHT//8pVp/7hcDJoKTQcCgs7a/\
/9h3YNPiVE3T2oCJ4GQQMBhlKucQYQYBg0EmBYNZBAxGmcQL5hAwGEnEYGkCBqNMkxOIMIOAwTA+\
A4M5BAwGmWrSL5hBwGCU6fsbYAkCBsOIF8whYDDINE01WYUISxMwGErAYFkCBqN49wWzCBiMMk0i\
BjMIGAwjXjCHgMEgB6voRQyWJWAwiktJwSwCBsO4lBTMIWAwin7BLAIGg/hCS5hHwGAUCzhgFgGD\
YewDgzkEDEbxbSowi4DBID4Dg3kEDEaZvAWDOU6NHgAS7ezs1PXr149038Wdz2rtMce3t7fro4/+\
XTX99P+RGxsbde7cuSecEk62qbXWRg8Baa5du1YXLlw40n3f/t3r9ec/vnPo+L8+/qz+dOUf9WB/\
/ycf49KlS3XlypUnnhNOMu/A4Bjst0V9ufd6bd1/pc4svq3Xztyo9ugHWI6AQWetFnXz3u/r5u4f\
qtWiqlrd3vtV/e/h5/IFM1jEAZ19vffqo3it1cGqjUVtP3i5Prn7VikYLE/AoLP9WnsUrx+a6n5b\
dwoRZhAw6Gx9+q7Wpr0fHW11ZtopS6hgeQIGnf1yfbN++9yH9cx0r6pardX9evX0p/WbZ/8zejSI\
9tQs4tjc3Bw9AifI1tbWke/76edb9cFf36+dh3+vOw9erGcW9+qF9S/q9tfbR36M3d1dr2G62NjY\
GD3C0p6agF29enX0CJwgt27dOvJ9N7fu1t8+/GTW8924ccNrmC4uX748eoSl2cgMS3iSjcyrYCMz\
HOYzMAAiCRgAkQQMgEgCBkAkAQMgkoABEOmp2QcGq3T27Nm6ePHisT3f+fPnj+25IIV9YABEcgoR\
gEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJ\
GACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABE\
EjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBI/webOv/JParJywAAAABJRU5ErkJggg==\
"
  frames[3] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAHe0lEQVR4nO3dwYtV5xnH8efcuaMxjtaiCS4CpVAM\
SaBx56J/QLLJxn+hu26L/0T33XTrqiuZZemmFAotIZvESIUiKVYwiYYkKk7NeN8uEkphYjX3zOvL\
7/r5wAzDmcudZzN859z7nDlTa60VAIRZjB4AANYhYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJ\
wACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMg\
koABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIG\
QCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEE\
DIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAi\
CRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAA\
RBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnA\
AIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCS\
gAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZA\
pOXoAWDTffPwXj384taB48uXdurl068NmAg2g4BBZ/c/vVH/+MNvDxw/9ZO362fv/KqmaRowFeTz\
EiIM00YPANEEDAZp//MZ+OEEDEZpTb9gBgGDUVorBYP1CRgAkQQMBmnVqjVnYLAuAYNRxAtmETAY\
SsRgXQIGo9hChFkEDAZpZQsR5hAwGKU1+YIZBAxGssgBaxMwGEW8YBYBg0G8BwbzCBiMol8wi4DB\
MAoGcwgYjGILEWYRMBikVVnkgBkEDIYRL5hDwGAU9wODWQQMRtIvWJuAwSCttbLGAesTMBilNUsc\
MIOAwTDiBXMIGAwlYrAuAYNBmhtawiwCBsOoF8whYDBKK1uIMIOAwTC2EGEOAYNRxAtmETAYxA0t\
YR4Bg1H0C2YRMOhsmp7wjbayxAEzCBh0dvTEK7X98qkDx/e+/ry+efDVgIlgMwgYdDZtbdW0+J5f\
tbaq1lbPfyDYEAIG3U3ffQCHScCgu0m/oAMBg96mqknB4NAJGHQmXtCHgEFv0/R/dumBdQkYAJEE\
DHqbbCFCDwIGnU3lJUToQcCgN+2CLgQMuptqcgYGh07AoDfxgi4EDDqb/Csp6ELAoDf9gi4EDLpT\
MOhBwKA7/0wKehAw6Ozb65glDA6bgEF34gU9CBj0Nv33E3CIBAy6c0NL6EHAoLdpssYBHQgYdDY5\
A4MuBAx68x4YdCFg0J14QQ8CBr1N7gcGPSxHDwCprl+/Xnfv3n36A9vj2rp//3v/Wrx69aNqNz59\
6lNM01Tnz5+vY8eO/fBBYUNNrbU2eghIdPHixbpy5cpTH7e9XNTvfv1evfXTVw9875e/2a2Pbnz2\
1OfY2tqqa9eu1blz59aaFTaRMzDorLWqVlUPHx+vW/8+V49WR+uVI/+q09u3Ro8G0QQMOmut1YP9\
E/XB1+/Wvcc/rqqpbu69Wa8f/1tV7Y4eD2JZ4oDOWlV9fP8Xde/x6fr2V26qVS3r+oML9dX+mcHT\
QS4Bg85aa7Xftg8cX9WyWtsaMBFsBgGDzlqremlx78Dx7WmvltOjARPBZhAweA7eOP6XevXIJ7Wo\
/apa1dHFg/r5iT/VzvKL0aNBrBdmieP27dujR2DD7O3tPfNjf//H9+tHJ/9edx69VvvtSJ1aflZ/\
3fqybn1+8MzsSe7cuVMnT55cZ1R4orNnz44eYW0vTMAuX748egQ2zM2bN5/5sX/+8J/ffXV1rZ+1\
Wq1qd3e3zpyx9MHhunTp0ugR1uZCZljTs17IfBhcyAwHeQ8MgEgCBkAkAQMgkoABEEnAAIgkYABE\
emGuA4PDduHChXpeV6EsFova2dl5Lj8LUrgODIBIXkIEIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkY\
AJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQS\
MAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACI\
JGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoAB\
EEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQB\
AyDSfwBCdfjZH03jggAAAABJRU5ErkJggg==\
"
  frames[4] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAILklEQVR4nO3dz6sdZx3H8e/MufnRJm2sKSHSIqhI\
1OxcdONeEBeB4L/hSrr1f3DjWujGTffWVUFUCi6CWqSKtoQUYpJqSYz35uaeeVwEhHJvm/TcO/Pw\
OX29FncxA+d+F2d4M8/MnBlaa60AIMzYewAA2ISAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQB\
AyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBI\
AgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgA\
kQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIw\
ACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgk\
YABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQ\
ScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAED\
IJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgC\
BkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACR\
dnoPANustVYP77xf08H+oX3nLn2tVqfOdJgKtoOAwZxaq/ff/kXtfXz7k9uHoa7+6Kf1/Jdf6TMX\
bAFLiNBJWx/0HgGiCRj00Kqmg8e9p4BoAgadTGsBg+MQMOikCRgci4BBJ87A4HgEDDpxDQyOR8Cg\
i+YuRDgmAYM5DVVnv3T5yF27//5w4WFguwgYzGqosxcuHbnn0YOPFp4FtouAwcyG1aneI8BWEjCY\
2ShgMAsBg5mNOwIGcxAwmJklRJiHgMHMLCHCPAQMZjauvLUI5iBgMLPPugbWWltwEtguAgYzGoah\
algdua+1aeFpYLsIGPQyTSIGxyBg0Emb1lWTgMGmBAw6adO6Wlv3HgNiCRh00qZ1NWdgsDEBg07a\
tK5yDQw2JmDQyZMlRAGDTQkYdNLaZAkRjkHAYGanz12onbPnD23ff/BRHew+6DARbAcBg5mtzpyr\
8dTZQ9vXj/dqfbDfYSLYDgIGMxuGVQ2DQw1OmqMKZjaMYw2jQw1OmqMKZjaMqypnYHDiHFUws2Fc\
OQODGTiqYGbDOLoGBjNwVMHMhtFNHDAHRxXMbBhWVZYQ4cQ5qmBuw1BDDUfv81NSsDEBg5k9eSvz\
0fum9eNlh4EtImDQ0XQgYLApAYOOmjMw2JiAQUeWEGFzAgYdWUKEzQkYdNTWB71HgFgCBh1ZQoTN\
CRgsYHX6+SO3P969v/AksD0EDBbwwle+eeT2h3f+sfAksD0EDBYwrE71HgG2joDBAsbVTu8RYOsI\
GCxgdAYGJ07AYAGWEOHkCRgsYNwRMDhpAgYLsIQIJ0/AYAGftYTYWltwEtgeAgYLGD7tjczt/3+A\
z0nAYBFHv9Gytana5K3MsAkBg47aNFWb1r3HgEgCBh21thYw2JCAQU+TJUTYlIBBR61N1ZozMNiE\
gEFHT27iEDDYhIBBR27igM0JGCxg3DlV46kzh7ZPjx/Ven+3w0SQzzseYEO7u7t148aNZ/sljcf/\
rdX4XI316BObD/Ye1J/+8PuaXrz11I+4ePFiXblyZdNxYesMze/YwEbee++9unr1aq3XT18CfOmF\
s/WzH/+gvvXVlw/t+8nP36rf/PHmUz/j+vXr9eabb240K2wjZ2CwgPXU6mBddXf/1bq7/2qdHh/V\
K2f+Ws+tHvYeDWIJGCxgalUf7H67Prz//ZpqVVWtbj/6en33xV/3Hg1iuYkDFnDhpW9UXfxhTbVT\
T34XcawH64v17n++13s0iCVgsIBx3KlxPPxKlXU73WEa2A4CBgvY339Ye3v3D20/u3rQYRrYDgIG\
C7hz5+/V/vnLOjM+rKqpxjqoS6c/qO+c+13v0SDWF+Ymjtu3b/cegS1z7969Z36bcmtVb//2V/Xi\
u3+ujw8u1c6wXy+fvlVv1UH97da/nukz9vb2fI85cZcvX+49wsa+MAF74403eo/Alrl79+4zB6yq\
6p2/3Kqqpz+w/Glu3rzpe8yJe/3113uPsDEPMsOGPs+DzCfBg8zwSa6BARBJwACIJGAARBIwACIJ\
GACRBAyASF+Y58DgpJ0/f76uXbtW0zQt8v9ee+21Rf4PpPAcGACRLCECEEnAAIgkYABEEjAAIgkY\
AJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQS\
MAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACI\
JGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoAB\
EEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQB\
AyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBI\
AgZAJAEDIJKAARDpf5pbV6BIRYqgAAAAAElFTkSuQmCC\
"
  frames[5] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAIUUlEQVR4nO3dT4sceR3H8W919ySZhAk4SMy6XmRh\
BZcQFk95Ah4EIQd9BIIQ77l68+4xRyHPYGHJTTxoxMOut6CRzWLEbJTNGPN3/nRXechB12mXSc9U\
lZ/O63WsqqG/hyneU9X1q2m6rusKAMJMxh4AAFYhYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJ\
wACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMg\
koABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIG\
QCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEE\
DIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAi\
CRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAA\
RBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnA\
AIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCS\
gAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZA\
pNnYA8A667qunv/902rn+4f2nbvwzZpunB5hKlgPAgZ96rr69Fe/qN3HD7+4vWnqvR/8tM5uvz3O\
XLAG3EKEkSy7KgOOTsBgDF1Ve7A39hQQTcBgJO38YOwRIJqAwUjcQoTjETAYiYDB8QgY9Kmpmm6c\
WbKjq4OXTwYfB9aJgEGvmtp6692le5797ZOBZ4H1ImDQs8ns1NgjwFoSMOiZgEE/BAx6JmDQDwGD\
nk02BAz6IGDQM1dg0A8Bg55NphvLd3RddV037DCwRgQMetQ0TVWz/DTruraqaweeCNaHgMFIusW8\
ulbAYFUCBiNp23l13WLsMSCWgMFIusXCFRgcg4DBSLp2UV3rCgxWJWAwkq71HRgch4BBzzY2t2p6\
+uyh7fvPH9d899kIE8F6EDDo2cbm+ZqdOhywxd7zWhzsjjARrAcBg54102k1k+nYY8DaETDoWTOZ\
VQkYnDgBg541k2k1E6canDRnFfSsmc7cQoQeCBj0bDKZVtMIGJw0AYO+NZNXL/Vdxst8YWUCBj17\
9Ub65fva+f6ww8AaETAYkYDB6gQMRiRgsDoBgxEJGKxOwGBEAgarEzAYxPJTbXGwN/AcsD4EDAaw\
9fV3l25/+uAPA08C60PAYACz0+eWbm8XBwNPAutDwGAAk9mpsUeAtSNgMAABg5MnYDAAAYOTJ2Aw\
gMmGgMFJEzAYwPRLrsC6rhtwElgfAgZD+B//TqXrOm+khxUJGIyoaxfVtouxx4BIAgYj6tpFdYv5\
2GNAJAGDMbVtda7AYCUCBiPq2oWAwYoEDEbUdW11rVuIsAoBgwFMZhs1mZ0+tH1xsFvzvRcjTAT5\
BAwGsLF5vk6f/+qh7fOXT2v/2T9GmAjyCRgMoJlMq5ksXwsGrEbAYADNZCJgcMIEDAbQNNNqJrOx\
x4C1ImAwgGYyqWYqYHCSBAwG0EymNXELEU6UgMEQmi/7DqzzRnpYgYDBAJqmqWqapfva+cHA08B6\
EDAYWTvfH3sEiCRgMDIBg9UIGIxMwGA1AgYjEzBYjYUpcAz37t2rhw8fHunY6c7O0r8Y//qXP9f9\
/d/WUZ5DvHTpUm1tbb3WjLCums7zu7Cya9eu1Y0bN4507I++9379+PvfefVE4n/46I8P6ic//7CO\
cibevn27rly5ssqosHZcgcFAPtt5VnvtmXqw963abc/V9sZndeHU/do+vzn2aBBJwGAgzw/O1MdP\
vlv/XHytqpq6v/vteufs72tavxx7NIjkIQ4YyIV3fliP5xfr1WnXVFfT+uTF+/Vo/+2xR4NIAgYD\
mc42D33/1dW02vKORFiFgMFAnjx5eOidh9NmvzaavZEmgmwCBgO589HNeuvUn2paB1XV1anmZb13\
7tf1lY2jPYYPfNEb8xDHUdfqwOt48eLFkY/9/PHj+vCDn9Wj+Tdqv92s87PP63fTR/X05d6RHqGv\
qtrZ2fG7zIm6ePHi2COs7I0J2M2bN8cegTV09+7dIx/77OV+ffCbO1V1Z+XPu3XrVt25s/rPw3+7\
fv362COszEJmOIbXWch8Eixkhn/zHRgAkQQMgEgCBkAkAQMgkoABEEnAAIj0xqwDgz5cvny5rl69\
OtjnbW9vD/ZZ8P/OOjAAIrmFCEAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkY\
AJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQS\
MAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZApH8BgNlK3YumXEUA\
AAAASUVORK5CYII=\
"
  frames[6] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAIUklEQVR4nO3dT4sceR3H8W9Vz2SSrBkxLBLXg0hk\
DzGI4jk+ACE3n4HgwXuuPgbPnoTc1IMXD4IHDYInyWFRVkEWlcT8gSRsZjIzPd1VHiK4Mu3upGeq\
ik/n9bpNVUN/D128p6rrV930fd8XAIRppx4AANYhYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJ\
wACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMg\
koABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIG\
QCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEE\
DIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAi\
CRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAA\
RBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnA\
AIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCS\
gAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZA\
pK2pB4BN1vd97T/5qLrF/MS+d7741Zpt70wwFWwGAYMh9X199Nuf1uGLR/+7vWnq69/7UV2++uVp\
5oIN4BIiTGTVWRlwegIGU+iruuOjqaeAaAIGE+kWx1OPANEEDCbiEiKcjYDBRAQMzkbAYEhN1Wz7\
4oodfR0ffDz6OLBJBAwG1dSVL72/cs/e47+NPAtsFgGDgbVbF6YeATaSgMHABAyGIWAwMAGDYQgY\
DKzdFjAYgoDBwJyBwTAEDAbWzrZX7+j76vt+3GFggwgYDKhpmqpm9WHW911V3408EWwOAYOJ9MtF\
9Z2AwboEDCbSdYvq++XUY0AsAYOJ9MulMzA4AwGDifTdsvrOGRisS8BgIn3nOzA4CwGDgW1fulKz\
ncsnts/3X9TicG+CiWAzCBgMbPvSbm1dOBmw5dF+LY8PJ5gINoOAwcCa2ayadjb1GLBxBAwG1rRb\
VQIG507AYGBNO6umdajBeXNUwcCa2ZZLiDAAAYOBte2smkbA4LwJGAytaV8/1HcVD/OFtQkYDOz1\
E+lX7+sW83GHgQ0iYDAhAYP1CRhMSMBgfQIGExIwWJ+AwYQEDNYnYDCK1Yfa8vho5DlgcwgYjODK\
e++v3P7y4YcjTwKbQ8BgBFs776zc3i2PR54ENoeAwQjarQtTjwAbR8BgBAIG50/AYAQCBudPwGAE\
7baAwXkTMBjB7FPOwPq+H3ES2BwCBmP4Pz+n0ve9J9LDmgQMJtR3y+q65dRjQCQBgwn13bL65WLq\
MSCSgMGUuq56Z2CwFgGDCfXdUsBgTQIGE+r7rvrOJURYh4DBCNqt7Wq3dk5sXx4f1uLo1QQTQT4B\
gxFsX9qtnd13T2xfHLys+d7zCSaCfAIGI2jaWTXt6rVgwHoEDEbQtK2AwTkTMBhB08yqabemHgM2\
ioDBCJq2rWYmYHCeBAxG0LSzal1ChHMlYDCG5tO+A+s9kR7WIGAwgqZpqppm5b5ucTzyNLAZBAwm\
1i3mU48AkQQMJiZgsB4Bg4kJGKxHwGBiAgbrsTAFzmA+n9f9+/drufzsn0SZPXu28j/GB//8e/1j\
/oc6zX2Iu7u7dfPmzTeeEzZR07t/F9b2+PHjun79eu3v73/ma7//3W/VD25/+/UdiZ/wx788rB/+\
+Fd1miPx1q1bde/evXXHhY3iDAxG8q9ne9X3TT1fXKvHR1+prWZR7138a13dvTT1aBBJwGAkh/NF\
PTy6Xn9+dauW/XZVVT08+lp9YfGziSeDTG7igJE8P7hcf9q7Vcv+QlU1VdXUq+7z9cHed/7zN/Am\
BAxGcjDvatmfvOjxOmjAmxIwGMn8+LAutAcntl+cvZxgGsgnYDCSZvGivnnlN3Wp/biqumpqWVe3\
H9Q3Pve7qlPdRA980ltzE8ejR4+mHoEN9PTp01M/Sf7x8/36yc9/UQfdr+v58bVqm0W9u/2gDg5f\
nuoW+qrX6858ljlP165dm3qEtb01Abt79+7UI7CB9vb2arFYnO61B/P65e8/PNP7PXnyxGeZc3Xn\
zp2pR1ibhcxwBm+ykPk8WMgM/+U7MAAiCRgAkQQMgEgCBkAkAQMgkoABEOmtWQcGQ9jZ2anbt2/X\
4eHhKO9348aNUd4HElgHBkAklxABiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABE\
EjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIj0b3BbY+AmiVCI\
AAAAAElFTkSuQmCC\
"
  frames[7] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAIg0lEQVR4nO3dTYtkZxnH4ftUdfdMJ87kxUFGEQlM\
QBmCBFy4kFE3LvMVXPsF8iX8AoIbF3GnC12J4MYhS2FA8QUihgQymUlkXpzu6eqqOnVc9EKhu5OZ\
qj7n8K9c1/LphroXffj1eXuq6bquKwAIMxl7AABYh4ABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZA\
JAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQM\
gEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJ\
GACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABE\
EjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkY\
AJF2xh4Attn88FHNHt07tb774su1//L1ESaC7SFg0KNHH/65Prj9y1Pr1775vXrtBz+upmlGmAq2\
g0uIMILVcl7dqh17DIgmYDCCVbuo6lZjjwHRBAxGsFouqhMw2IiAwQhW7aK6lYDBJgQMejTd3a9m\
Mj21vjx6cnIZEVibgEGP9l/5Wk0vvXhqffb4XrXHRyNMBNtDwKBHk53daiYOM+iDIwt6NJnuVtM4\
zKAPjizo0WRHwKAvjizoUTPdqzrjIQ5gcwIGPZpMd87fLsp7YLARAYORtMvjsUeAaAIGY+iqVgsB\
g00IGIykFTDYiIDBSJyBwWYEDHp31kMcXa3cA4ONCBj0qWnqyldfP/NHTz5+b+BhYLsIGPRs94WX\
zlxvF7OBJ4HtImDQs8nu5bFHgK0kYNCz6c6lsUeArSRg0LPJroBBHwQMejY9L2BdV13XDTsMbBEB\
gx41TVN1zm703aq1HyJsQMBgJKvV8iRiwFoEDEbStctaCRisTcBgJKvWGRhsQsBgJJ2AwUYEDHq2\
s7dfkzPeBVvOnlQ7PxphItgOAgY927vy5dp94eqp9cXTx7WcHY4wEWwHAYOeNdOdaibTsceArSNg\
0LPJRMCgDwIGPTs5A9sZewzYOgIGPZt85iVEW0nBugQM+tZMTraUOoPH6GF9AgY9O9kP8eyfrRbH\
ww4DW0TAYEStgMHaBAxG5AwM1idgMKJ2MRt7BIglYDCAS1eunbl+9PDuwJPA9hAwGMD+q18/c332\
6OOBJ4HtIWAwgOnu6c18gc0IGAxgImBw4QQMBjDZvTz2CLB1BAwG4BIiXDwBgwFMdvbO/kFX1XX2\
Q4R1CBgM4vy9ELvVcuBZYDsIGIyoW7W1agUM1iFgMKLVqq1OwGAtAgYj6gQM1iZgMICTr1Q5fbh1\
7aJWy/kIE0E+AYMB7L34Sl1+6Sun1ucHD2r2+P4IE0E+AYMBNNPpuY/Se4we1iNgMIBmMq1msjP2\
GLBVBAwG0EymNZkKGFwkAYMBNJOdagQMLpSAwQA++wyscx8M1iBgMIDzHqOvKu+BwZoEDEbWLo7H\
HgEiCRiMrF3Mxh4BIrmrDBuYz+d1586datv2c393+uDBmf8xfvD+P+v9J/v1LHfBrl69Wm+88cZz\
zwnbqOncPYa13b9/v27cuFGHh4ef+7s//cmP6odvvnZq/Re/u1M/++2fnunzbt26Vbdv337eMWEr\
uYQIA/nX3Ye16qoeLK7X3w++W+8dfqcO2yv1rW9cq+bsrwsDPoNLiDCQD+8/rruzG/W3p7eq7Xar\
quru8ev1ytWDkSeDTM7AYCD/frpffz24VW23Vyff0NzU09VL9ZeD79d539gMnE/AYCBPj9tqz7jo\
cRI04HkJGAzkeD6rvebo1Prl6ZMRpoF8AgYD6RYP680rf6j9yX+qalVNtfXq7kf17S/9seqZHqIH\
/t8X5iGOe/fujT0CW+jTTz995n0M7z04qJ//6td1tPp9PVxcr0mzrGu7H9XR7Ek968ss8/nc3zIX\
6vr162OPsLYvTMDeeeedsUdgCx0cHNRy+Wx7GR4czes37/5jo8/75JNP/C1zod5+++2xR1ibF5lh\
A8/zIvNF8CIz/I97YABEEjAAIgkYAJEEDIBIAgZAJAEDINIX5j0w6MOlS5fqrbfeqtlsmG9Vvnnz\
5iCfAwm8BwZAJJcQAYgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACR\
BAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAA\
IgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRg\
AEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJ\
wACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMg\
koABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACI9F8xe2wyIA7W4AAAAABJRU5E\
rkJggg==\
"
  frames[8] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAIP0lEQVR4nO3dQYsceR3H4V91zySzidl1g2TjLggq\
iriE4MFD3oDgLQffRA6eJFffgxfJUcjda9jbghJZWFRcNkLUrCwbDIuJISHJZKa7/h4CymbG7KRn\
qspv53kOOVT1UL9Dig/176qurrXWCgDCzKYeAABWIWAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQ\
ScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAED\
IJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgC\
BkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACR\
BAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAA\
IgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRg\
AEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJ\
wACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMg\
koABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIG\
QKSNqQeAddZaq0eff1L9YmfPvpNnvlnzzeMTTAXrQcBgSK3VJ+//qrbv3/ni9q6rd3/y8zpx+p1p\
5oI1YAkRJtKWi6lHgGgCBlNoVf1id+opIJqAwUT6pYDBYQgYTKQJGByKgMFEXIHB4QgYTMR3YHA4\
AgaTaO5ChEMSMBhSV7X11bP77nryr9sjDwPrRcBgUF1tvXFm3z1PH94deRZYLwIGA+vmm1OPAGtJ\
wGBgMwGDQQgYDGy2IWAwBAGDgVlChGEIGAzMEiIMQ8BgYLO5txbBEAQMBvai78BaayNOAutFwGBA\
XddVdfN997XWjzwNrBcBg6n0vYjBIQgYTKT1y6pewGBVAgYTaf2yWltOPQbEEjCYSOuX1VyBwcoE\
DCbS+mWV78BgZQIGE3m2hChgsCoBg4m01ltChEMQMBjYsZNv1MbWV/Zs33l4txZPHk4wEawHAYOB\
zY+frNnm1p7ty93tWi52JpgI1oOAwcC6bl5d51SDo+asgoF1s1l1M6caHDVnFQysm82rXIHBkXNW\
wcC62dwVGAzAWQUD62Yz34HBAJxVMLBu5iYOGIKzCgbWdfMqS4hw5JxVMLSuq666/ff5KSlYmYDB\
wJ69lXn/ff1yd9xhYI0IGEyoXwgYrErAYELNFRisTMBgQpYQYXUCBhOyhAirEzCYUFsuph4BYgkY\
TMgSIqxOwGAE82Mn9t2+++TByJPA+hAwGMGpr39n3+2PPr818iSwPgQMRtDNN6ceAdaOgMEIZvON\
qUeAtSNgMIKZKzA4cgIGI7CECEdPwGAEsw0Bg6MmYDACS4hw9AQMRvCiJcTW2oiTwPoQMBhB97/e\
yNz+8w/wkgQMRrH/Gy1b66v13soMqxAwmFDr+2r9cuoxIJKAwYRaWwoYrEjAYEq9JURYlYDBhFrr\
qzVXYLAKAYMJPbuJQ8BgFQIGE3ITB6xOwGAEs43Nmm0e37O9331ay50nE0wE+bzjAQ7h1q1bdefO\
nS//4O7jms9eq1k9/cLmxfbD+ujD31X/+mcHOt65c+fq1KlTq4wKa6drfscGVnbp0qW6cuXKl37u\
zVNb9Yuf/ri+942v7dn3s1++V7/506cHOt7169frwoULLz0nrCNXYDCCZd9quezrab9Vt7e/W9v9\
yTq9+Y86c+xg4QL2EjAYQd+3erTYqt8/+FHdX5ypqq4+3f5+ffvEH6rVe1OPB5HcxAEjWPZ9ffzg\
h3V/8VY9O+26ajWvvz3+Qd3deXvq8SCSgMEIln2r7cVGPf+jvq3m1dd8mqEgnIDBCPplX8fqQT3/\
6pR5t1Ob3dP9/wh4IQGDESxbq29tfVBvH/9rzWu3qlod657Uuyd/W29uHuA2fGCPV+YmjgM9qwMv\
6fHjxwf6XGtVv37/j/XWR3+vu7vv1E7/Wr2+8c/6YH63/vLZvQMf7969e/4vc6TOnj079Qgre2UC\
dvXq1alHYA3dvHnzwJ/94M+3q+p2VX288vGuXbtWN27cWPnv4XmXL1+eeoSVeZAZDuGgDzIfFQ8y\
w3/5DgyASAIGQCQBAyCSgAEQScAAiCRgAER6ZZ4DgyGcP3++Ll68ONrxTp8+Pdqx4P+d58AAiGQJ\
EYBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAi\
CRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAA\
RBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnA\
AIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCS\
gAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZA\
JAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASP8GYJpXJ35U9kYAAAAASUVORK5CYII=\
"
  frames[9] = "data:image/png;base64,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\
AAAASUVORK5CYII=\
"
  frames[10] = "data:image/png;base64,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\
"
  frames[11] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAHX0lEQVR4nO3dv49cVxnH4ffuD+8mWhMSS8YICpAA\
KQ2KXGO3SGnzH1DSu6XmL6BHLihc0SBRpEAr0SEhbJQmK0RQYLMKkbJ2spHXO5ciNLAGr/bm3aPv\
+Hkqa2Y88zZXH58z546neZ7nAoAwG6MHAIDLEDAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAA\
RBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnA\
AIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCS\
gAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZA\
JAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQM\
gEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJ\
GACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABE\
EjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDINLW\
6AFg3Z2ePK6TTz489/jW7l69euPbAyaC9SBg0Gie53ry0UG9/9tfnHvu6995q77/458OmArWgy1E\
aDbP8+gRYC0JGHQTMGghYNBNwKCFgEGzeV6NHgHWkoBBOysw6CBg0M0WIrQQMGjmFCL0EDDoJmDQ\
QsCgmUMc0EPAoJsVGLQQMGg2O4UILQQMulmBQQsBg26+A4MWAgbNbCFCDwGDbrYQoYWAQTM3MkMP\
AYNuAgYtBAyaWYFBDwGDdk4hQgcBg25WYNBCwKCZLUToIWDQTcCghYBBM79GDz0EDLpZgUELAYNu\
AgYtBAya+S1E6CFg0M0KDFoIGDRziAN6CBh0swKDFgIG7QQMOggYNPNLHNBDwKCbgEELAYNm8+rs\
fzwzXekcsG4EDFrN9eTw/ec+c/3W9654FlgvAgad5qrV2elzn9rYunbFw8B6ETAYZbKFCEsIGAwy\
CRgsImAwioDBIgIGwwgYLCFgMMRU0+TygyVcQTCKLURYRMBgFAGDRQQMBnEKEZYRMBhGwGAJAYNB\
pg2XHyzhCoJhrMBgCQGDQXwHBssIGIwiYLCIgMEwAgZLCBgM4pc4YBlXEIxiCxEWETAYxCEOWEbA\
YBQBg0UEDAaZHOKARQQMRnGIAxZxBcEothBhEQGDQRzigGUEDEYRMFhEwGAQKzBYZmv0AJDo0aNH\
dXx8/MLXTTXX5snJufOGc8318OGjmg8+euF7bG5u1u3bt2t7e/uS08J6muZ5nkcPAWnu3r1b+/v7\
L3zdxjTVr372Tn33m6//x+Nnq1X95Oe/rvc++PiF77G3t1cHBwd18+bNS88L68gKDK7AZ2fX6+9f\
/KCe1VZ949pf67XNf9TKvx1hEQGDRnNVPX72Rn3w6dv1+eprVVX1ty/erDdf3a+VfsEiDnFAs4dP\
7tbnq9fqy/8+Zaqz+Vr9+cmP6rNn10ePBtEEDFp9Gaz/djZv1dnsFCIsIWDQaq7dzcfnHt3ZOKlp\
Ph0wD6wPAYNGU1X9cO939cb2hzXVWVWt6pWN43rr+ru1M50PG3BxL80hjsPDw9EjsEaePn16odet\
5rl++Zv9emX3j/Xx6bdqNW/V69uH9fuNx/XP45MLvcc8z3V0dFSr1WrJyPBct27dGj3Cpb00Abt/\
//7oEVgjR0dHF37tu3/4y7//9KdLfdbp6Wk9ePCg9vb2LvX34f+5d+/e6BEuzY3McAkXvZH5q+BG\
Zng+34EBEEnAAIgkYABEEjAAIgkYAJEEDIBIL819YPBVunPnTt24ceNKPmt3d7d2dnau5LMgifvA\
AIhkCxGASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABE\
EjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEj/Aluu+AcpgWTbAAAAAElFTkSuQmCC\
"
  frames[12] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAIb0lEQVR4nO3dz44c1RnG4e9094TBhkBEiJ0IyZYM\
SsQdZBFlnVVYRLkClpEiRUK5j2wRKxa5AK4BsQCJRSJCFAcJBxHH2BAGbDyerqosYENmMGZ6Tpfe\
9vMsq1rT32JKP9Xp+tOmaZoKAMIs5h4AAE5DwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQ\
ScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAED\
IJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgC\
BkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACR\
BAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAA\
IgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRg\
AEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJ\
wACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMg\
koABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASKu5\
B4CHwe2P3q/h3hfHtp/74aVaPfLoDBNBPgGDzqZxrPdf/3Pd/s97x/b97Nd/rMcvXplhKshnCRE6\
m8Z1TeM49xiwcwQMOhvHoWoc5h4Ddo6AQWfTMNQkYHDmBAw6m8Z1TZMlRDhrAgadTaMzMOhBwKCz\
4ehujet7x7a35aoWi+UME8FuEDDo7PDgZh3d+fTY9v3v/6j2zj85w0SwGwQMZtKWq2oLhyCclqMH\
ZtIWy2rNEiKcloDBTNpi6QwMNuDogZm05arKRRxwagIGM1ksVtUEDE5NwKCjaZqqajpx35e/gTkE\
4bQcPdDZNBydvKO1aq1tdxjYIQIGnQ0n3MQMbE7AoLPxSMCgBwGDzk56jBSwOQGDzsZBwKAHAYPO\
LCFCHwIGXU115+MPTtyz/+TFLc8Cu0XAoKep6t5nt07ctf/EhS0PA7tFwGAWrRar7809BEQTMJiJ\
gMFmBAxmImCwGQGDmSz2BAw2IWAwh1a1dAYGGxEw6OrkJ9FXVbXFaotzwO4RMOhoHNY1TeM37PUk\
etiEgEFH43BUNX5TwIBNCBh0NN33DAzYhIBBR9Nw9NVbmYGzJmDQ0Tisq5yBQRcCBh2Nw1FNJ/4G\
1lzDARsSMOjoi1sf1Pru58e2n3vqmVo9cn6GiWB3CBh0NI7rOulesOXefrXFcvsDwQ4RMJhBW66q\
NYcfbMIRBDNYLFdVAgYbcQTBDNpyr9rCVRywCQGDGSwsIcLGHEHQyf1uYG7LPUuIsCFHEHQ0DcOJ\
21tr1ZolRNiEgEFH43Bv7hFgZwkYdDPVeCRg0IuAQS/TV69TAboQMOhmqnHtDAx6ETDoaFo7A4Ne\
BAw6maapDm9/csKeVqv9x7Y+D+waAYNOpnGoL25eO7a9LRZ1/unL2x8IdoyAwda1Wqy+N/cQEE/A\
YNta1WK1N/cUEE/AYOtataUzMNiUgMEMLCHC5gQMtqy1VksBg40JGMygLVdzjwDxBAw6mYajmurk\
V6p4Ej1sTsCgk3E4qrrPO8GAzQgYdDKuj+77UktgMwIGnTgDg74EDDr58kG+Aga9uBQKvoMbN27U\
1atXH+izizs3arFe1/9frjEMY7311ls1rR791r9x5cqVunDhwikmhd0nYPAdvPbaa/Xiiy8+0Gd/\
88vn6w+//Xmtll9f6HjnvQ/rd7//Vd2+++2vWnn55Zcf+PvgYSNg0MmTj+1XW+zVv+4+W5+tn6rH\
V7fqJ49crY/+e7vWwzj3eBBPwKCTYVrVXz//Rf378NmaqlWrqT4++nHdOfqnazvgDLiIAzq5dvf5\
+vDwuZpqUVWtplrUh4fP1T8OflqjgsHGBAw6WU+rqmOXcLS6c6+5PwzOgIBBJ/uLO9Vq+Nq2VkO1\
4cASIpwBAYNOntn/e10593at2mFVTbVqh3Xl3Nv19PIdS4hwBh6aiziuX78+9wjsgIODgwf+7Ot/\
eb9uffqn+mR9oW4PT9T55UH9YHW93r320Xf6Pv+79HTx4sW5Rzi1hyZgr7766twjsAPefPPNB/7s\
u9du1rvXblbV3079fW+88UYNw/DtH4RTeumll+Ye4dTa5NdkeGCvvPLKVm8sdiMzfDO/gQEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgPzX1gcBYuXbpUL7zwwta+7/Lly1v7LkjjPjAAIllCBCCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkY\
AJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQS\
MAAiCRgAkQQMgEgCBkAkAQMg0v8AYENeeEiy0dYAAAAASUVORK5CYII=\
"
  frames[13] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJB0lEQVR4nO3dO49cdx3H4d85M7OzvhBCIIqBKBEI\
ZCMqekSTiiJS5I43QZcy74CKVxApBTXpQpMCmoTGSIBEFEXRkgjnYlvBl73MzDkUoSDxrLPZ2TOH\
7/p5JDdzxt6f5P3ro//MuTR93/cFAGHasQcAgNMQMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgk\
YABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQ\
ScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAED\
IJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgC\
BkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACR\
BAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAA\
IgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRg\
AEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJ\
wACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMg\
0nTsAeBxcP/Tveq7VU1mu9XO5jWZzqudzaudWIJwWlYPDKzvVrX3p9/Vg9sf1mS2W5PZvNqd3ZrM\
duv5n/+qLnzre2OPCJEEDAbWrRaf/1kcVLc4qMX/HFse7o82F6TzHRgMrFsuq18txx4Dzh0Bg4H1\
q0V13WrsMeDcETAY2OLgbq0OH4w9Bpw7AgYDO7p3p5YHdx96ff6N79TOxSdGmAjOBwGDkUx2LlQ7\
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\
6T96vnf1P8Vo4wAAAABJRU5ErkJggg==\
"
  frames[14] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJc0lEQVR4nO3dzY8bdxnA8Wdm7PXuZpO0VWnTF5WX\
tEgVPXBGrfgLeuGA+Ed648SfwI1zxQk4IyQuKFIlJFTEWwuqRCNIm3RJ2ijZl3jtmeEQIQS2SVh7\
PHomn4+0ijSzip/Djr6y/ZvfFG3btgEAyZR9DwAA5yFgAKQkYACkJGAApCRgAKQkYACkJGAApCRg\
AKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAA\
pCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQk\
YACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRg\
AKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAA\
pCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQk\
YACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRg\
AKQkYACkJGAApCRgAKQkYACkNOp7AHgSzKencfr5jSjHu1GNJ1GOJw//HU2iKIq+x4OUBAy24Pgf\
1+Ojn//wYcB2dqMaP/y58NxX45Vvfbfv8SAlAYMtmD+4H21TRz09jnp6/O8T3nzBufkODLZg/uCo\
7xFgcAQMtuDkzo0VZ7wFg/MSMNiCB3dvLj1++eVvbHkSGA4Bgx7tHDzd9wiQloBBx9q2jWiXnxvt\
XdzuMDAgAgYda+t5tE299Fy1s7flaWA4BAw6Vs+n0dSzFWct4oDzEjDoWDObRjM/63sMGBwBg46d\
fnEzpvfvLByvxntRjXZ6mAiGQcCgY21TR7TNwvHJpWdjfOFyDxPBMAgY9KQc70ZZeQcG5yVg0KG2\
bWPVGvpyvBNFNd7uQDAgAgYdq6cnS4+X1SiK0iUI5+XqgY6t3si38CwwWIOAQcdmdqKHTggYdKqN\
+5/+ZemZg+evbnkWGBYBgy61EfXZ6ZITRUwuPrP1cWBIBAx6Mtq1kS+sQ8CgQ+2SG5gjIqKIGE0O\
tjsMDIyAQYea2XTlTvRFVW15GhgWAYMO1Wen0dTzFWctoYd1CBh06Oz4bjTzad9jwCAJGHTo9O6n\
S3fi2Hv6xRhN9nuYCIZDwKAH4/2nohzbyBfWIWDQg2qyF0U56nsMSE3AoCNt265cgViNJ1GUViHC\
OgQMOtPG/MHx0jNFUdrIF9YkYNCVto16RcCA9QkYdKRt25ge3Vl6rhrvbXkaGB4Bg460TR3Hn/11\
4XhRVnHwwms9TATDImCwbUURo90LfU8B6QkYbF0Ro4mAwboEDDrS1vNoo104XhRFVDu+A4N1CRh0\
ZD49ibZZ/jgVS+hhfQIGHanPjiNWPQ8MWJuAQUeOD69HPVvcib4c7UR4BwZrEzDoyOz0/tJ3YAcv\
vBbVeLeHiWBYBAy2bDS5EFG49GBdriLoQNsurj78l9FkP4rSpQfrchVBJ9po5mdLz5SjnYjwHRis\
S8CgA23TxHy6aiPfwjJ62AABgy60TdTTk76ngEETMOhAM5/FyZ0bC8eLahS7l5/rYSIYHgGDDrRN\
HbOTuwvHy2ocu08938NEMDwCBltUFGVUNvKFjRAw6MCyTXwjIqIsY7Szv91hYKAEDDpQnz1YeryI\
IopqtOVpYJgEDDowf3D0P29mBtYnYNCB+em9pfsgFlXVwzQwTAIGHbj3yYfRNvXC8YMXvh5FKWKw\
CQIGHVj1IMvx7kW7cMCGCBhs0Wj3wLPAYEMEDDbs4eKN5Qs4RpP9sJEvbIaAwYa1Tb1yGX2UpY8Q\
YUMEDDbsYcBO+x4DBk/AYMPaeh7zFQErXHKwMa4m2LD59ChObv9t4fho9yD2v/RKDxPBMAkYbFjb\
ttE284XjRTWO0eSgh4lgmGzKBo9hNpvF+++/H3W9eHPyfyum96JqF9caTs9m8Zvf/i6K0eSR/8el\
S5fijTfeOOe08GQoWhu2wSMdHh7G1atX4+jo6JG/+7UXn44ff/87UZX/+QHH9Vt343s/+Gk0zaMv\
uTfffDOuXbt27nnhSeAdGGzYpf1JRBTx+exKfHb25ahiHi/tfhQRd1fdHgacg4DBhl0+2I2b01fj\
g5O3om7HERFxc/pqPDP/iX7BBlnEARv20pWvxJ+O34q63YmH34QVcdJcjj8cfbvv0WBQBAw27LWX\
n41myYcb83YctpGCzREw2LCyqGNSLt7IfHz/s/AlGGyOgMGG7VVH8c2Lv4y98l5ENFFEHc+MP4nr\
v/+RpzTDBj0xizhu3brV9wgkdvv27ceOz89+9UG898e/x2nzi/hidiXKoo5nxzfizx9//Nivd3Z2\
5m+Wrbhy5UrfI5zbExOwd999t+8RSOzo6Chms9lj/e6vP/xk7dc7PDz0N8tWvPPOO32PcG5uZIbH\
8P/cyLwJbmSGR/MdGAApCRgAKQkYACkJGAApCRgAKQkYACk9MfeBwTomk0m8/fbbcXq6uEVUF15/\
/fWtvA5k5j4wAFLyESIAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp/RNQMKAq\
eI0VrwAAAABJRU5ErkJggg==\
"
  frames[15] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJfElEQVR4nO3dv48c5RnA8Wd2Z/fOPp8hJ2KMAgiC\
REGESForRZqkpsrfENFT5h+gTUGVCqVPESnQBCkKokBKUiQiIYFItsFA7AN8v3Zvd2dSHEoQN6cc\
tzM7etafj2RZnlnfPoVHX9/tO+8UdV3XAQDJDPoeAAAuQsAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEip7HsAeBDMp0dxtHs7BqPNGI42YjDaOPm93IiiKPoeD1ISMFiB\
g3//K/7x21+cBGy8GcPRya+ta0/Hkzd+2vd4kJKAwQrMjvairhaxmB7EYnrwvxO++YIL8xkYrMDs\
4PO+R4C1I2CwAl/c+kvj8Y3tR1Y8CawPAYMVqKt54/Gta99d8SSwPgQMejS6tN33CJCWgEHH6moR\
dVU3nisFDC5MwKBji9k0qvlx47liYCEwXJSAQccWs0lUi+aAARcnYNCx2cHnMT/aP3V8uLEVw/Fm\
DxPBehAw6NjsaC8Wx4enjm9c2bGIA5YgYNCTwXgzBsNx32NAWgIGHarrOupq0XhuONqMQTla8USw\
PgQMOjafnP78KyKiGAwjCpcgXJSrBzo2P9o785xHqcDFCRh0qo6DezfPOCdesAwBgy7VEZPP7jSe\
euiJ7614GFgvAga9KGJ8ZafvISA1AYNONe+BGIWNfGFZAgYdWswmZy6jH5QbK54G1ouAQYfm06Oo\
Fs3PAgOWI2DQoekXnzRuIxVRWIQISxIw6NB0/15Us+mp41uPPBmjzSs9TATrQ8CgB+Wlq1EMbSMF\
yxAw6Ehd12cuQiw3LsVg6GGWsAwBgw4tZpPG44NybB9EWJIrCLpS1/ZBhA4JGHSmjtlZARMvWJqA\
QUeqahF7H/391PFiMIztx57tYSJYLwIGXanrqBez08eLIsZb31r9PLBmBAxWrojSPoiwNAGDjlTz\
45Ol9F9TFEUMR5s9TATrRcCgI/PpQdR11XjOCkRYnoBBR+aT/ead6MULWiFg0JH9O/+MquFG5q1r\
T8dg5FEqsCwBg45UTSsQI2J06aEoBsMVTwPrR8BgxcrNrShsIwVLcxVBB+q6jrpqfpBlOb5kH0Ro\
gasIOlDXVcwnB80ni4FViNACAYMuVIuYT/b7ngLWmoBBBxazSdw/Yx/E0eWrPUwE60fAoAt1NO6D\
OCjHsfXtp1Y/D6whAYMVKgbDKDev9D0GrAUBgw407sAREVEMoty4vNphYE0JGHRgPt1v3sg3whJ6\
aIkrCTowO9qPiNMBA9ojYNCByed3oq5O70Q/2nrYPWDQEgGDDhztfhTR8CiVK48+Yx9EaImAwQqV\
m1c8TgVaImDQsrquGxdwRHwZsBAwaIOAQcvqatH4HLCIiGLox4fQFgGDltXVPObHh2eet4gD2iFg\
0LLF8SSm9++eOl4MyxjZhQNaI2DQssVsEtP7n546Xo4vx+bDj/UwEawnAYMVKYZlDG0jBa0RMGjb\
GSsQi2F58jRmoBUCBi1bHB81Hi+KIopBueJpYH0JGLRsdrRnG0RYAQGDlh3duxXNBbN8HtokYNCy\
g7s3G49fffw520hBi/xAHs7hgw8+iI8//vhcrx3u7jb+z/D2p1/Eh2+/fa6v8fzzz8f29vY3mBAe\
PEV91qZtwH+99NJL8eqrr57rta/87Mfxo+8/der4z3/5u3jjnffP9TXeeuutuHHjxjcZER44vgOD\
Fg0HRYzKQRxXm3F78mxMqq3YGd2Ja+ObMTme9z0erBUBgxZtjssYjh6KP97/SXw2vxYRRdycPBfP\
XP5T1PFG3+PBWrGIA1q0MSrj1uKH8dn80Ti5vIqoYxjvH/4g7h4/3vd4sFYEDFr06M5WPHH95Duv\
r6pjGFV4lAq0ScCgReVgEFfHh/H1+8Bmx4cxnez1MxSsKZ+BQcue3XonZvVGfDJ9OhZRxriYRLn3\
m7h96899jwZr5YEJ2Hnv4YEmh4dnP6Dyqz68uxev/OrNqOL3cW/2nTiuLsV2eTfmh7djNq/O/X67\
u7v+zbIS169f73uEC3tgAvbaa6/1PQKJvffee+d63e7eUfz6D3/78k9/vfD7vf766/Huu+9e+O/D\
eb388st9j3BhbmSGc/gmNzK3wY3M8P9ZxAFASgIGQEoCBkBKAgZASgIGQEoCBkBKD8x9YLCMF154\
IV588cWVvd/Ozs7K3guych8YACn5ESIAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAAp/QevfZWXipV+uAAAAABJRU5ErkJggg==\
"
  frames[16] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJd0lEQVR4nO3dz4skZxnA8aeqf8xsZnYTN8maYAwm\
AQkr+OMU0IsXvUly8m/wH8jRf8CrVz0FL17iRUQEURByECQHEV0VSTabrMlk3N2ZTndPd1d5WEFI\
17Cb6aount7PB5aFt5id5zDFd2fmrbeKuq7rAIBkyr4HAICLEDAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFIa9j0APAqW82lMj9+LcrQfg9FelKO9+38P96Ioir7Hg5QE\
DLZg8tG/4u+/+vH9gI33YzC6/+fg2gvx/De/3/d4kJKAwRYspidRV6tYzSexmk/+f8E3X3BhfgcG\
W7CY3Ol7BNg5AgZbcPfmnxvX9y4/teVJYHcIGGxBXS0b1w+uvbjlSWB3CBj0aHTpct8jQFoCBh2r\
q1XUVd14bShgcGECBh1bLeZRLc8arxWljcBwUQIGHVstZlGtmgMGXJyAQccWkzuxnJ6urQ/2DmIw\
3u9hItgNAgYdW0xPYnX2ydr63uFVmzhgAwIGPSnH+1EOxn2PAWkJGHSoruuoq1XjtcFoP8rhaMsT\
we4QMOjYcrb++6+IiKIcRBRuQbgodw90bDk9OfeaV6nAxQkYdKqOycfvnnNNvGATAgZdqiNm//mg\
8dLjX/zKloeB3SJg0IsixodX+x4CUhMw6FTzGYhROMgXNiVg0KHVYnbuNvpyuLflaWC3CBh0aDmf\
RrVqfhcYsBkBgw7N7/678RipiMImRNiQgEGH5qcfR7WYr60fPPV8jPYPe5gIdoeAQQ+Gl65EMXCM\
FGxCwKAjdV2fuwlxuHcpyoGXWcImBAw6tFrMGtfL4dg5iLAhdxB0pa6dgwgdEjDoTB2L8wImXrAx\
AYOOVNUqTt7/29p6UQ7i8rNf7mEi2C0CBl2p66hXi/X1oojxwee2Pw/sGAGDrSti6BxE2JiAQUeq\
5dn9rfSfUhRFDEb7PUwEu0XAoCPL+STqumq8ZgcibE7AoCPL2WnzSfTiBa0QMOjI6Qf/iKrhQeaD\
ay9EOfIqFdiUgEFHqqYdiBExuvR4FOVgy9PA7hEw2LLh/kEUjpGCjbmLoAN1XUddNb/Icji+5BxE\
aIG7CDpQ11UsZ5Pmi0VpFyK0QMCgC9UqlrPTvqeAnSZg0IHVYhb3zjkHcfTYlR4mgt0jYNCFOhrP\
QSyH4zh4+kvbnwd2kIDBFhXlIIb7h32PATtBwKADjSdwREQUZQz3HtvuMLCjBAw6sJyfNh/kG2EL\
PbTEnQQdWExPI2I9YEB7BAw6MLvzQdTV+kn0o4MnPAMGLREw6MD0+P2IhlepHH7+JecgQksEDLZo\
uH/odSrQEgGDltV13biBI+J/AQsBgzYIGLSsrlaN7wGLiCgGfnwIbREwaFldLWN59sm5123igHYI\
GLRsdTaL+b2jtfViMIyRUzigNQIGLVstZjG/9+Ha+nD8WOw/8WwPE8FuEjDYkmIwjIFjpKA1AgZt\
O2cHYjEY3n8bM9AKAYOWrc6mjetFUURRDrc8DewuAYOWLaYnjkGELRAwaNn045vRXDDb56FNAgYt\
mxy927h+5bnrjpGCFvmBPDzAdDqNt99++9zjoT5tcHzc+D/D9z68G7feeuuBH3/16tV4+eWXP+OU\
8Ogp6oe9K+ERdePGjbh+/XqsVue8ZflTfvSD78S3v/6ltfUf/vS38es//vOBH//aa6/Fm2+++VnH\
hEeO78CgRYOyiOFgGB+dPRcfnT0X43IeX9i7EZcGk5idLfseD3aKgEGL9sejOC6+Fn+6992oYhAR\
ddyevxjfuPKbvkeDnWMTB7ToiSdfivLp70UVw7i/67CMk9WT8ZfTb9lZDy0TMGhRUQ6jLEdr67eO\
z+Kd23d6mAh2l4BBi4bFIsblfG29OjuOu5P1deDiBAxadHlwHF+9/LvYKycRUUUZy7g2fideGv8+\
JrOzvseDnfLIbOK4fft23yOQ1NHR+ru9znPr6CR+8vOfxWT1y7izvBbD4iyeGt+Kuyf3YrGsHurf\
mM1mvl7ZmmeeeabvES7skQnYG2+80fcIJHV0dBRV9XDxOT6Zxi/+8NeNPt/Nmzd9vbI1r7/+et8j\
XJgHmeEBPuuDzJvyIDM8HL8DAyAlAQMgJQEDICUBAyAlAQMgJQEDIKVH5jkwuKjDw8N49dVXH/pZ\
sE298sorW/k8kJ3nwABIyY8QAUhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASOm/\
48yg9gzdSb8AAAAASUVORK5CYII=\
"
  frames[17] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJd0lEQVR4nO3dz4skZxnA8aeqf8xsZnYTN8maYAwm\
AQkr+OMU0IsXvUly8m/wH8jRf8CrVz0FL17iRUQEURByECQHEV0VSTabrMlk3N2ZTndPd1d5WEFI\
17Cb6aount7PB5aFt5id5zDFd2fmrbeKuq7rAIBkyr4HAICLEDAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFIa9j0APAqW82lMj9+LcrQfg9FelKO9+38P96Ioir7Hg5QE\
DLZg8tG/4u+/+vH9gI33YzC6/+fg2gvx/De/3/d4kJKAwRYspidRV6tYzSexmk/+f8E3X3BhfgcG\
W7CY3Ol7BNg5AgZbcPfmnxvX9y4/teVJYHcIGGxBXS0b1w+uvbjlSWB3CBj0aHTpct8jQFoCBh2r\
q1XUVd14bShgcGECBh1bLeZRLc8arxWljcBwUQIGHVstZlGtmgMGXJyAQccWkzuxnJ6urQ/2DmIw\
3u9hItgNAgYdW0xPYnX2ydr63uFVmzhgAwIGPSnH+1EOxn2PAWkJGHSoruuoq1XjtcFoP8rhaMsT\
we4QMOjYcrb++6+IiKIcRBRuQbgodw90bDk9OfeaV6nAxQkYdKqOycfvnnNNvGATAgZdqiNm//mg\
8dLjX/zKloeB3SJg0IsixodX+x4CUhMw6FTzGYhROMgXNiVg0KHVYnbuNvpyuLflaWC3CBh0aDmf\
RrVqfhcYsBkBgw7N7/678RipiMImRNiQgEGH5qcfR7WYr60fPPV8jPYPe5gIdoeAQQ+Gl65EMXCM\
FGxCwKAjdV2fuwlxuHcpyoGXWcImBAw6tFrMGtfL4dg5iLAhdxB0pa6dgwgdEjDoTB2L8wImXrAx\
AYOOVNUqTt7/29p6UQ7i8rNf7mEi2C0CBl2p66hXi/X1oojxwee2Pw/sGAGDrSti6BxE2JiAQUeq\
5dn9rfSfUhRFDEb7PUwEu0XAoCPL+STqumq8ZgcibE7AoCPL2WnzSfTiBa0QMOjI6Qf/iKrhQeaD\
ay9EOfIqFdiUgEFHqqYdiBExuvR4FOVgy9PA7hEw2LLh/kEUjpGCjbmLoAN1XUddNb/Icji+5BxE\
aIG7CDpQ11UsZ5Pmi0VpFyK0QMCgC9UqlrPTvqeAnSZg0IHVYhb3zjkHcfTYlR4mgt0jYNCFOhrP\
QSyH4zh4+kvbnwd2kIDBFhXlIIb7h32PATtBwKADjSdwREQUZQz3HtvuMLCjBAw6sJyfNh/kG2EL\
PbTEnQQdWExPI2I9YEB7BAw6MLvzQdTV+kn0o4MnPAMGLREw6MD0+P2IhlepHH7+JecgQksEDLZo\
uH/odSrQEgGDltV13biBI+J/AQsBgzYIGLSsrlaN7wGLiCgGfnwIbREwaFldLWN59sm5123igHYI\
GLRsdTaL+b2jtfViMIyRUzigNQIGLVstZjG/9+Ha+nD8WOw/8WwPE8FuEjDYkmIwjIFjpKA1AgZt\
O2cHYjEY3n8bM9AKAYOWrc6mjetFUURRDrc8DewuAYOWLaYnjkGELRAwaNn045vRXDDb56FNAgYt\
mxy927h+5bnrjpGCFvmBPDzAdDqNt99++9zjoT5tcHzc+D/D9z68G7feeuuBH3/16tV4+eWXP+OU\
8Ogp6oe9K+ERdePGjbh+/XqsVue8ZflTfvSD78S3v/6ltfUf/vS38es//vOBH//aa6/Fm2+++VnH\
hEeO78CgRYOyiOFgGB+dPRcfnT0X43IeX9i7EZcGk5idLfseD3aKgEGL9sejOC6+Fn+6992oYhAR\
ddyevxjfuPKbvkeDnWMTB7ToiSdfivLp70UVw7i/67CMk9WT8ZfTb9lZDy0TMGhRUQ6jLEdr67eO\
z+Kd23d6mAh2l4BBi4bFIsblfG29OjuOu5P1deDiBAxadHlwHF+9/LvYKycRUUUZy7g2fideGv8+\
JrOzvseDnfLIbOK4fft23yOQ1NHR+ru9znPr6CR+8vOfxWT1y7izvBbD4iyeGt+Kuyf3YrGsHurf\
mM1mvl7ZmmeeeabvES7skQnYG2+80fcIJHV0dBRV9XDxOT6Zxi/+8NeNPt/Nmzd9vbI1r7/+et8j\
XJgHmeEBPuuDzJvyIDM8HL8DAyAlAQMgJQEDICUBAyAlAQMgJQEDIKVH5jkwuKjDw8N49dVXH/pZ\
sE298sorW/k8kJ3nwABIyY8QAUhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASOm/\
48yg9gzdSb8AAAAASUVORK5CYII=\
"
  frames[18] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJfElEQVR4nO3dv48c5RnA8Wd2Z/fOPp8hJ2KMAgiC\
REGESForRZqkpsrfENFT5h+gTUGVCqVPESnQBCkKokBKUiQiIYFItsFA7AN8v3Zvd2dSHEoQN6cc\
tzM7etafj2RZnlnfPoVHX9/tO+8UdV3XAQDJDPoeAAAuQsAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEip7HsAeBDMp0dxtHs7BqPNGI42YjDaOPm93IiiKPoeD1ISMFiB\
g3//K/7x21+cBGy8GcPRya+ta0/Hkzd+2vd4kJKAwQrMjvairhaxmB7EYnrwvxO++YIL8xkYrMDs\
4PO+R4C1I2CwAl/c+kvj8Y3tR1Y8CawPAYMVqKt54/Gta99d8SSwPgQMejS6tN33CJCWgEHH6moR\
dVU3nisFDC5MwKBji9k0qvlx47liYCEwXJSAQccWs0lUi+aAARcnYNCx2cHnMT/aP3V8uLEVw/Fm\
DxPBehAw6NjsaC8Wx4enjm9c2bGIA5YgYNCTwXgzBsNx32NAWgIGHarrOupq0XhuONqMQTla8USw\
PgQMOjafnP78KyKiGAwjCpcgXJSrBzo2P9o785xHqcDFCRh0qo6DezfPOCdesAwBgy7VEZPP7jSe\
euiJ7614GFgvAga9KGJ8ZafvISA1AYNONe+BGIWNfGFZAgYdWswmZy6jH5QbK54G1ouAQYfm06Oo\
Fs3PAgOWI2DQoekXnzRuIxVRWIQISxIw6NB0/15Us+mp41uPPBmjzSs9TATrQ8CgB+Wlq1EMbSMF\
yxAw6Ehd12cuQiw3LsVg6GGWsAwBgw4tZpPG44NybB9EWJIrCLpS1/ZBhA4JGHSmjtlZARMvWJqA\
QUeqahF7H/391PFiMIztx57tYSJYLwIGXanrqBez08eLIsZb31r9PLBmBAxWrojSPoiwNAGDjlTz\
45Ol9F9TFEUMR5s9TATrRcCgI/PpQdR11XjOCkRYnoBBR+aT/ead6MULWiFg0JH9O/+MquFG5q1r\
T8dg5FEqsCwBg45UTSsQI2J06aEoBsMVTwPrR8BgxcrNrShsIwVLcxVBB+q6jrpqfpBlOb5kH0Ro\
gasIOlDXVcwnB80ni4FViNACAYMuVIuYT/b7ngLWmoBBBxazSdw/Yx/E0eWrPUwE60fAoAt1NO6D\
OCjHsfXtp1Y/D6whAYMVKgbDKDev9D0GrAUBgw407sAREVEMoty4vNphYE0JGHRgPt1v3sg3whJ6\
aIkrCTowO9qPiNMBA9ojYNCByed3oq5O70Q/2nrYPWDQEgGDDhztfhTR8CiVK48+Yx9EaImAwQqV\
m1c8TgVaImDQsrquGxdwRHwZsBAwaIOAQcvqatH4HLCIiGLox4fQFgGDltXVPObHh2eet4gD2iFg\
0LLF8SSm9++eOl4MyxjZhQNaI2DQssVsEtP7n546Xo4vx+bDj/UwEawnAYMVKYZlDG0jBa0RMGjb\
GSsQi2F58jRmoBUCBi1bHB81Hi+KIopBueJpYH0JGLRsdrRnG0RYAQGDlh3duxXNBbN8HtokYNCy\
g7s3G49fffw520hBi/xAHs7hgw8+iI8//vhcrx3u7jb+z/D2p1/Eh2+/fa6v8fzzz8f29vY3mBAe\
PEV91qZtwH+99NJL8eqrr57rta/87Mfxo+8/der4z3/5u3jjnffP9TXeeuutuHHjxjcZER44vgOD\
Fg0HRYzKQRxXm3F78mxMqq3YGd2Ja+ObMTme9z0erBUBgxZtjssYjh6KP97/SXw2vxYRRdycPBfP\
XP5T1PFG3+PBWrGIA1q0MSrj1uKH8dn80Ti5vIqoYxjvH/4g7h4/3vd4sFYEDFr06M5WPHH95Duv\
r6pjGFV4lAq0ScCgReVgEFfHh/H1+8Bmx4cxnez1MxSsKZ+BQcue3XonZvVGfDJ9OhZRxriYRLn3\
m7h96899jwZr5YEJ2Hnv4YEmh4dnP6Dyqz68uxev/OrNqOL3cW/2nTiuLsV2eTfmh7djNq/O/X67\
u7v+zbIS169f73uEC3tgAvbaa6/1PQKJvffee+d63e7eUfz6D3/78k9/vfD7vf766/Huu+9e+O/D\
eb388st9j3BhbmSGc/gmNzK3wY3M8P9ZxAFASgIGQEoCBkBKAgZASgIGQEoCBkBKD8x9YLCMF154\
IV588cWVvd/Ozs7K3guych8YACn5ESIAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAAp/QevfZWXipV+uAAAAABJRU5ErkJggg==\
"
  frames[19] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJeUlEQVR4nO3dy44cZxmA4a8OPT3HHAzEjkGRIEqQ\
iKKIrbdIrCMWwA2wyT47LiErNrmAiAtgRXZISF5EioiCFFkkinPAOUHGOLHn0DPdVSzMQTDdeJju\
6tJXfh7JslTVnv4WLr3qqb/+Ltq2bQMAkin7HgAALkLAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIqe57AHgYTCdHcXT7VpSjzahG4yhH4/t/1+MoiqLv8SAlAYM1OPjr\
h/Heb391P2Abm1GN7v/ZeeK78dS1n/Y9HqQkYLAG0+O70TazmE0OYjY5+PcJH77gwtwDgzWYHt/r\
ewQYHAGDNTjcv7XgjI9gcFECBmtwfOezuccf/c5za54EhkPAoEcbu4/3PQKkJWDQsbZtI9r55+qt\
vfUOAwMiYNCxdjaNtpnNPVdtbK15GhgOAYOOzaaTaGanC85axAEXJWDQseZ0Es30pO8xYHAEDDp2\
9LfPYnJ3/8zxarQVVb3Rw0QwDAIGHWubWUTbnDk+fuSbMdp5tIeJYBgEDHpSjjajrHwCg4sSMOhQ\
27axaA19OdqIohqtdyAYEAGDjs0mh3OPl1UdRekShIty9UDHFm/kW/guMFiCgEHHTu1ED50QMOhU\
G3c//dPcM7uXn17zLDAsAgZdaiNmJ0dzThQx3ru09nFgSAQMelJv2sgXliFg0KF2zgPMERFRRNTj\
3fUOAwMjYNCh5nSycCf6oqrWPA0Mi4BBh2YnR9HMpgvOWkIPyxAw6NDJwZ1oppO+x4BBEjDo0NGd\
T+fuxLH1+NWox9s9TATDIWDQg9H2Y1GObOQLyxAw6EE13oqirPseA1ITMOhI27YLVyBWo3EUpVWI\
sAwBg860MT0+mHumKEob+cKSBAy60rYxWxAwYHkCBh1p2zYm9/bnnqtGW2ueBoZHwKAjbTOLgy9u\
njlelFXsPvlMDxPBsAgYrFtRRL250/cUkJ6AwdoVUY8FDJYlYNCRdjaNNtozx4uiiGrDPTBYloBB\
R6aTw2ib+V+nYgk9LE/AoCOzk4OIRd8HBixNwKAjB3/5MGanZ3eiL+uNCJ/AYGkCBh05Pbo79xPY\
7pPPRDXa7GEiGBYBgzWrxzsRhUsPluUqgg607dnVh/9Uj7ejKF16sCxXEXSijWZ6MvdMWW9EhHtg\
sCwBgw60TRPTyaKNfAvL6GEFBAy60DYxmxz2PQUMmoBBB5rpaRzu3zpzvKjq2Hz0iR4mguERMOhA\
28zi9PDOmeNlNYrNxy73MBEMj4DBGhVFGZWNfGElBAw6MG8T34iIKMuoN7bXOwwMlIBBB2Ynx3OP\
F1FEUdVrngaGScCgA9Pje//zYWZgeQIGHZgefT13H8SiqnqYBoZJwKADX39yI9pmdub47pPPRlGK\
GKyCgEEHFn2R5Whzzy4csCICBmtUb+76LjBYEQGDFbu/eGP+Ao56vB028oXVEDBYsbaZLVxGH2Xp\
V4iwIgIGK3Y/YEd9jwGDJ2CwYu1sGtMFAStccrAyriZYsenkXhx++fGZ4/Xmbmx/66keJoJhEjBY\
sbZto22mZ44X1Sjq8W4PE8Ew2ZQNzuHmzZvx+eefn+u1xeTrqNqzaw0nJ6fx5ltvR1GPH/gznn/+\
+djb27vApPDwEDA4h1deeSVeffXVc732e1cfj1//8idRlf/5C46P/3wrfv6LH0XTPHiPxOvXr8e1\
a9cuNCs8LAQMVuyR7XGcNlvx0fH347jZiUujz+KJjX/cE7O/L6yMgMGK7ex+I/5w98fx1exyRBTx\
8fEP4untt6KO3/U9GgyKgMGKXX32Z3FneuVfDyy3UcX7hz+M2RdvLf6iS+D/ZhUirFhVb57ZbaON\
Kv74wX74ijBYHQGDFdsq78V/3+yqipM4OPiqn4FgoAQMVuyZnTfj6vi9qOI0ItrYKI7iuZ3rUU4+\
6Hs0GJSH5h7YeZ/hgXkODw/P/drf/P7tuPzOR7F/+u04abZir/4y3qj24+33vzj3z7h9+7b/s6zF\
lStX+h7hwh6agL322mt9j0Bi77777rlf+8aNTyLik4h458Lv9/rrr8eNGzcu/O/hvF5++eW+R7iw\
om3dVoYHeemll879IPMqeJAZHsw9MABSEjAAUhIwAFISMABSEjAAUhIwAFJ6aJ4Dg2W88MIL8eKL\
L67t/S5durS294KsPAcGQEp+hQhASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBK\
AgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoC\
BkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIG\
QEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBK\
fwckqoozXWZqJAAAAABJRU5ErkJggg==\
"
  frames[20] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJB0lEQVR4nO3dO49cdx3H4d85M7OzvhBCIIqBKBEI\
ZCMqekSTiiJS5I43QZcy74CKVxApBTXpQpMCmoTGSIBEFEXRkgjnYlvBl73MzDkUoSDxrLPZ2TOH\
7/p5JDdzxt6f5P3ro//MuTR93/cFAGHasQcAgNMQMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgk\
YABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQ\
ScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAED\
IJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgC\
BkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACR\
BAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAA\
IgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRg\
AEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJ\
wACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMg\
0nTsAeBxcP/Tveq7VU1mu9XO5jWZzqudzaudWIJwWlYPDKzvVrX3p9/Vg9sf1mS2W5PZvNqd3ZrM\
duv5n/+qLnzre2OPCJEEDAbWrRaf/1kcVLc4qMX/HFse7o82F6TzHRgMrFsuq18txx4Dzh0Bg4H1\
q0V13WrsMeDcETAY2OLgbq0OH4w9Bpw7AgYDO7p3p5YHdx96ff6N79TOxSdGmAjOBwGDkUx2LlQ7\
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\
6T96vnf1P8Vo4wAAAABJRU5ErkJggg==\
"
  frames[21] = "data:image/png;base64,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\
eb9uffqn+mR9oW4PT9T55UH9YHW93r320Xf6Pv+79HTx4sW5Rzi1hyZgr7766twjsAPefPPNB/7s\
u9du1rvXblbV3079fW+88UYNw/DtH4RTeumll+Ye4dTa5NdkeGCvvPLKVm8sdiMzfDO/gQEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgPzX1gcBYuXbpUL7zwwta+7/Lly1v7LkjjPjAAIllCBCCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkY\
AJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQS\
MAAiCRgAkQQMgEgCBkAkAQMg0v8AYENeeEiy0dYAAAAASUVORK5CYII=\
"
  frames[22] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAHlElEQVR4nO3dP49cVx3H4d+d2diOSQGJlLUU40Ak\
pBU06WhITWXJL4BXQO8GyRR+E1DRuKZzhYSokwLhApmAsZJIUQaBdh15Lf/bnUORFIRdRbt7fXz0\
HT+PtM3c0c4p9uijc869O1NrrRUAhFmMHgAAnIWAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQB\
AyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBI\
AgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgA\
kQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIw\
ACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgk\
YABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQ\
ScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAED\
IJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgC\
BkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACR\
tkYPADbd88cP6/Hu50de37rwRl186/KAEcFmEDDoqLVW+6t7de8Pvzly7bs/eL9+9PNfDhgVbAZb\
iNBZa+vRQ4CNJGDQWVsLGPQgYNBZa4ejhwAbScCgMysw6EPAoLe1FRj0IGDQmZs4oA8Bg85sIUIf\
AgadNVuI0IWAQWe2EKEPAYPerMCgCwGDzpyBQR8CBp3ZQoQ+BAw6cxMH9CFg0JkVGPQhYNCZMzDo\
Q8CgM1uI0IeAQVetHv3rn8deeWP7vZc8FtgsAgY9tarD50+OvbQ8f/ElDwY2i4DBINO0HD0EiCZg\
MMRU08L0gznMIBhkmkw/mMMMgkGmhS1EmEPAYBRbiDCLGQSDuIkD5hEwGMRNHDCPGQSDWIHBPAIG\
g1iBwTxmEAwiYDCPGQSj2EKEWQQMRpiswGAuMwgGcRMHzCNgMIgVGMxjBsEgAgbzmEHQUav2LVdN\
P5jDDIKeWvvqB3jhBAw6am39rWsw4OwEDHpar63AoBMBg45aW1dZg0EXAgY9OQODbgQMOnIGBv0I\
GHRkCxH6ETDoqbmJA3oRMOiorcULehEw6Kmtq1mBQRcCBh05A4N+BAw6as7AoBsBg57EC7oRMOio\
OQODbgQMelo7A4NeBAw6Wq8Pvo7YN03TsqZpGjAi2BwCBh093v28Dp4+OvL662++U8vzFweMCDaH\
gEFHX91Gf9S0WNY0mX4whxkEA0yLRZUdRJhFwGCEaSoFg3kEDAaYpsXXEQPOSsBggGlaWH/BTAIG\
I1iBwWwCBgNMC2dgMJeAwQhuoYfZtkYPANLs7e3V3bt3T/Texe79Wh73O3Yf1IcffVQ1HXf1m65c\
uVKXL18+5Shh803NfxqFU7l9+3ZdvXr1RO+99rOd+tUvPjjy+h//fL9+/bs/1fOD4x90/l83b96s\
GzdunHqcsOmswKCzw7ao1dMf1pcH23Vx+aDeOf+PWq+bb1qBmQQMOlq3RX386Kf12ZMfV6tFTdXq\
P8+u1NPD+75mBWZykgwdffHsvfrsyU+q1bKqpmq1qH8//379ff/90UODeAIGHa3bVrUj02yqZ+st\
W4gwk4BBR+cWj2tRB//3aqtz9dAWIswkYNDR2+c+rZ3vfFivTU+qqtVyelbvXvhrvXvhL76nGWZ6\
ZW7iWK1Wo4fAhtjb2zvxe+/cW9X0+9/Wlwdv18PD79Xri/1687Uv6tPV7ol/x/7+vr9furl06dLo\
IZzZKxOwW7dujR4CG+KkDzFXVX2yelCfrB5U1d/O/Hl37tzx90s3169fHz2EM/MgM5zSaR5kfhE8\
yAzHcwYGQCQBAyCSgAEQScAAiCRgAEQSMAAivTLPgcGLsr29XdeuXXtpn7ezs/PSPguSeA4MgEi2\
EAGIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMg\
koABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIG\
QCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEE\
DIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAi\
CRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAA\
RBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiPRf/W8sAjj1si0AAAAASUVORK5CYII=\
"
  frames[23] = "data:image/png;base64,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\
"
  frames[24] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAI0UlEQVR4nO3dT4skdx3H8W919/xz3d0kBnYSDxES\
CAnBByAEDwqecvPs0WeQs0/A5+AhHrwL3gRdBFEPCbkIEtQE1uyfxOwmO7Mz/afKwx5Epsfd6Z2u\
4lN5vY7VTc/30MWb6qn6/Zqu67oCgDCToQcAgE0IGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQS\
MAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACI\
JGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoAB\
EEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQB\
AyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBI\
AgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgA\
kQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIw\
ACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgk\
YABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQ\
aTb0ADBm86P7dXL/9pnjO1eeq4PnDgeYCMZDwGCL7n/yYX1885dnjr/4+vfqO9//STWNH0FgU84e\
GEC7nFfXtkOPAdEEDAawWsyra1dDjwHRBAwG0C7nVZ0rMHgWAgZbNNu7Us3k7L+aF0f3a7U4HWAi\
GA8Bgy3av36jpnvfOHP85Ms7tZqfDDARjIeAwRZNZrvVTJxmsA3OLNiiyWy3mmY69BgwSgIGW+QK\
DLbHmQVb9Dhg512Bdb3OAmMjYLBF/2+ljXY573ESGB8BgyF0VauFuxDhWQgYDGQ1fzT0CBBNwGAg\
ngODZyNgsE1N1f7abVO6Ov7sk97HgTERMNiqpg6ef2ntK6dffdbzLDAuAgZbNt05GHoEGCUBgy2b\
7u4PPQKMkoDBlk13XYHBNggYbNlktrv+ha6rrrMaB2xKwGCLmqapOmc1jna1tCszPAMBg4F0q0V1\
7XLoMSCWgMFA2tWiupUrMNiUgMFA2uWiWldgsDEBgy3bu/pCzQ6unTl++uXdWhw9GGAiGAcBgy2b\
7l2p6c7ZZ8Ha5bza1WKAiWAcBAy2bDLdqWY6G3oMGB0Bgy2bTGc1OXdXZmBTAgZb1rgCg60QMNiy\
ZjKt5pyHmT3IDJsTMNiyx6txrH+tXdjUEjYlYDCg1fzR0CNALAGDAa3mrsBgUwIGvVh/qi3nxz3P\
AeMhYNCDa99+fe3xr279tedJYDwEDHow27+69riVOGBzAgY9sCszXD4Bgx4IGFw+AYMerFvMt6qq\
uqqu6/odBkZCwKAHzXT9Wohd11bXtT1PA+MgYNCL9UtxdO2quqUbOWATAgYD6lZLdyLChgQMBtS2\
y2pXy6HHgEgCBj1oJpNq1uwJ1i7nFvSFDQkY9GD3yvO1/9zhmeOLo/t18uDuABNBPgGDHkyms5pM\
d4YeA0ZFwKAHzWRWjYDBpRIw6EEzndZkJmBwmQQMetBMZjWZzs55tbMaB2xAwKAvzfrTrV3Mex4E\
xkHAoAdNs34ljqqq1eJRj5PAeAgYDGw19xwYbELAYGCuwGAzAgY92TlYvyvz6Vef9zwJjIOAQU++\
eeO1tceP731cVe5ChIsSMOjJdHe/zttWBbg4AYOeTHYPhh4BRkXAoCfTnf2hR4BRETDoyXRn7/xf\
EK3EARcmYNCb9fXq2rbapV2Z4aLOW5wNeArz+bzef//9Wq1WT3xvc/qgpl13JmMnJ8f1lz/9sdrp\
3hM/49q1a/XWW29tOC2MS9NZRRQ2dufOnXr11Vfr6Ojoie995cb1+tXPflzTyf/+8HHni4f105//\
uj79/OETP+Ptt9+umzdvbjwvjIkrMOhVU/9eHNad01dq1izr5f2/1XRyXPu7TkW4KGcN9OTRfFUf\
3n257u38qFbd473B/nX6Wr2x/5t66VtX6x+f3h94QsjiJg7oyWryfN2qH9aq263HN3Q0ddxer4/m\
P6gXr18ZejyII2DQk6aZVDPZPXN82dmpGTYhYNCT5fK0Hh2f/ZnwYPLkmzeAswQMevLgwe06/uQX\
dTD5sqraampVL+zcqu9e/V01FvOFC/va3MRx+/btoUdghO7du1dP+yRK11X94c+/rxv//Ht9sTis\
SbOsF3du1W+beX3w0dN9P+fzue8yl+rw8HDoETb2tQnYe++9N/QIjNDDhw9ruVw+9fs/+Oh21VPG\
ap27d+/6LnOp3n333aFH2JgHmeEZXORB5svgQWb4L/8DAyCSgAEQScAAiCRgAEQSMAAiCRgAkb42\
z4HBNuzt7dU777xTJycnvfy9N998s5e/Awk8BwZAJD8hAhBJwACIJGAARBIwACIJGACRBAyASAIG\
QCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEE\
DIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAi\
CRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAA\
RBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnA\
AIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCS\
gAEQ6T8SwWhXdXCiLAAAAABJRU5ErkJggg==\
"
  frames[25] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJgklEQVR4nO3du49cZxnA4ffMZa++JERxnAuKIY6E\
CAVCoqSmdUFPEaWhT8m/QZNFFPkTgqJ0dBjSpAkRSghSMIltETvxetezcznno0BCinY23h17zsl7\
/DySm++MdN5iRz955jvfVKWUEgCQzKDrAQBgFQIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoC\
BkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIG\
QEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBK\
AgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoC\
BkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIG\
QEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBK\
AgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoC\
BkBKAgZASgIGQEoCBkBKo64HgD4rTRPNYhr1/H//mvlRNPU8dp+9EoOhtx88Cu8gWJNmMY9//ukP\
MTu48/941bOjKKWJ137129i68GzXI0JqAgZrUkoTBzc/jvlk/xvr1XAci8lBhIDBI/EdGLStNDGf\
3Ot6CkhPwGBNqmoQG+efObZemjr2//1RBxNBvwgYrEk1GMTus1eWXitN3e4w0EMCButSVTHaOtf1\
FNBbAgZrU8Voe3nASl1HKU3L80C/CBisUTUYLl1fzCZRah8jwqMQMFiTqqpOvFbPHkTTLFqcBvpH\
wGCNxtsXohqOj61P9/8TzWzSwUTQHwIGa7R18bkYbe4cW58dfhX1fNbBRNAfAgZrNNzcicqZh7AW\
AgZrNNrYjmqwPGCeBYNHI2CwRtVgtHwzR4lYTA/aHwh6RMCgI4vJ/a5HgNQEDNatWvY2K3H45b9a\
HwX6RMBgnaoqLn7/taWXjr6+1fIw0C8CBmu2ce57XY8AvSRgsGbj7fPLL5QSpZR2h4EeETBYs8Fo\
c+l6s5hGqectTwP9IWCwRt96HuJ8Gs1CwGBVAgZrdlLDZgd3Y35kKz2sSsBgzbYuXl66kWP+4F4s\
jg47mAj6QcBgzYabOzEcb3U9BvSOgMGaDcdbMRhtnHDVLkRYlYDBmlXDUVTD5b/MXPtNMFiZgMGa\
fdtOxLnzEGFlAgatWP5WWzzYb3kO6A8BgxZcePFHS9f3P/+o5UmgPwQMWnDSeYiNkzhgZQIGLTj5\
PMRwHiKsSMCgBcPNnaXrpamjNHXL00A/CBi0YvlOxKaeRzM/ankW6AcBgxactJG+mU+jngkYrELA\
oAXj3adi8+Jzx9anB3dj8vXNDiaC/AQMWjAcb8Vo2fdgpYnSNO0PBD0gYNCCwWgcAwf6wmMlYNCC\
ajiO4QkH+pZ6bis9rEDAoAVVVZ34y5bOQ4TVCBh0bP7gXtcjQEoCBi3Z2H166fq9Gx+G3wWDsxMw\
aMn551+NZU+ENYtZ+8NADwgYtGS0faHrEaBXBAxaMt7aPeFIjuJZMFiBgEFbquHS5aauo55NWh4G\
8hMw6FipFwIGKxAwaMlgNI7R1rlj6/X8KKb373QwEeQmYNCS4eZObD/9wrH1Zn4Us/tfdjAR5CZg\
0JLBYLT8QF9gJQIGLamGoxhubC+9VkpxHiKckYBBi6oTdiIupoctTwL5CRi0pDrhMN+IiMXRQYT/\
gcGZCBi0aLx7MZY9zTy5+3mU4mFmOAsBgxbtXvpBVIPjHyNO7n4epak7mAjyEjBo0Wjr/Im/Cwac\
jYBBi8Zb55Z/F+b7LzgzAYMWVaPx0vVSStTzo5angdwEDL4DSqmjPrKVHs5CwOA7oFnMY/L1za7H\
gFQEDFo0HG/GuctXj62Xeh6TrwQMzmLU9QDQB7dv345PP/304S8sTcT9aSz7JuzGjRvx2fTPp7rf\
1atX49KlS2cbEnpGwOAxeOedd+KNN9546OuqKuI3134ev/7lT49d23vrrfj9ux+c6n57e3vx+uuv\
n3lO6BMBgxaVEjGb11GXUXwxfSXuL56J86M78cLmP2JzYxRVZUc9nJaAQcvqMooPD34RN6evRIkq\
qihxd/58nN/5ewwHg1jUjpSC07CJA1r22eTH8cX0apQYREQVJQbxxfTVuDf6WQwHTumA0xIwaFkd\
4zh+oG8VP/nhS7G9ufxBZ+A4AYOWPTi8E3W9+MZaFXU8tT2NgXMS4dQEDFr28d/ei0vxlxhV04go\
MaqmcXXng3hx8+OuR4NUnphNHLdu3ep6BHpsf3//1K+9/2ASf3z3dzGpXorD+kLsDu/F06PbUTd1\
HB7NTn0/f9M8DpcvX+56hJU9MQF7++23ux6BHnv//fdP/dq6KfHeXz+JiE9Wvt/169djsVg8/IXw\
EG+++WbXI6ysKsVTJ/Co9vb2TvUg8+O8nweZedL5DgyAlAQMgJQEDICUBAyAlAQMgJQEDICUnpjn\
wGCdrly5EteuXWvtfi+//HJr94LvKs+BAZCSjxABSEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABI6b+4U7XbWgZGaAAAAABJRU5ErkJggg==\
"
  frames[26] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKHElEQVR4nO3dO48dZxnA8WfObXe967VjLNsJCVLE\
RaBAEwooMBJFvgAtX4CWIt8BPgAdXSSKFFAjCiSIkJCAhghDEoOixMSxHSde7+1cZl4KJ9WZNbvH\
OXN4jn8/yc3MWe/TrP6aOe+8U5VSSgBAMr1VDwAAixAwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSGqx6AFhXpZQo9TTq6Tia6Tjq6XE0s3EMty7Exu7lVY8H6QkYLMn9\
d/4Ud278IZrJcdTTT/9NjuPqSz+I57/7w6gqN0DgSQgYLMnxg7ux/8Hbc8cnhw+iNHVUfQGDJ+Ev\
CJak6vVbjx/eezea6aTjaWD9CBgsyflnvxq94ebc8cn+/SjNbAUTwXoRMFiS0fYz0TvhKqx0PAus\
IwGDJRls7kS0BKyUEvX4cAUTwXoRMFiWqoqq7XgpMTn4pOtpYO0IGCxLVUU1GM0dLk0d+7fnVycC\
ZyNgsCS9Xj8uvPBSy5kS08MHnc8D60bAYFmqKkbbz7SeKqVEKZZywJMQMFiaKgab261n6smRpfTw\
hAQMlqSqqoiqdRlHzI73o5lNO54I1ouAwRL1BxsRLXseHt57L2bjgxVMBOtDwGCJtq+8GMOt3bnj\
9fQoSl2vYCJYHwIGSzTY3IneYNh6rhQBgychYLBE/eFm+6a+JWJ6uNf9QLBGBAxWosTk4ONVDwGp\
CRgsUxUx3DrfeurhrRsdDwPrRcBgqaq48MI3W8/U03HHs8B6ETBYsuFJu3E0dZSm6XgaWB8CBkvW\
H82/1DLi0W4c9cxVGCxKwGCJqqqKaH+pSoz37trUF56AgMGSDTbORW+4MXd8evggZsd244BFCRgs\
2ebFqzHaubTqMWDtCBgsWX+09WhPxBaN78BgYQIGS1b1BlH1W3bjiIiph5lhYQIGS1ad8EqViIi9\
/7zV4SSwXgQMOrB54Vrr8fHenY4ngfUhYNCB88997cRzpZQOJ4H1IWDQgeG5i63HS11Hqb2ZGRYh\
YNCB3mDUeryeHsdsctTxNLAeBAxWaLJ/P44/+XDVY0BKAgYdGG7uxHB7/jZiM5tE7QoMFiJg0IHh\
ud3YOH/5hLPFQg5YgIBBB3qDjeiPtlrPzcb2Q4RFCBh0oOr1ouq178Yx2bcbByxCwKAj/WH7e8H2\
3v97RLiFCGclYNCR3S9+vfX47Hi/40lgPQgYdGS480yc9HLLsIgDzkzAoCPDrd3W46WeeZgZFiBg\
0JGq6rdegDX1NGZHbiPCWQkYdKTq96M/nF9KPz16GIf33l3BRJCbgEFHBhvbce4Lz8+fKE00s0n3\
A0FyAgYd6Q2GMdjaaT1XmtpuHHBGAgYdqXqD6I/OtZ6bHu2FZ8HgbAQMOlJVVVQnLKOf7H9sKT2c\
kYBBh4bbF6JtKeLerX9EaeruB4LEBAw6tHP1y617IpZm5jswOCMBgw4Nty9GVPNXYKWUKM1sBRNB\
XgIGHRpsbEfVFrCmjunRwxVMBHkJGHSpJV4Rj24heq0KnI2AQYd6vUGMdi7NHW+m4zi48+8VTAR5\
CRh0qDcYxbnLX2o9V0pjIQecgYBBh6pe/8Rd6R9tJyVgcFoCBl2qquiP2t/MPD3ci9I0HQ8EeQkY\
dOjRCsT2hRz7H7wVzfS424EgMQGDjm1euBpVfzB3fHr00BUYnIGAQce2Lj0Xvf7ohLO+A4PTEjDo\
2PDcblS9tj+9EvXELUQ4LQGDjvUGGxHV/J9eKU1MDj9ZwUSQ0/yNeGBhb775Zuzt7T32M1Uzi/50\
OreUo9SzePvPv4vx5Y9O9bv6/X68/PLLMRwOF5wWcquKJyfhc3P9+vV44403HvuZYb8XP/vxK/G9\
b80/0Pyr39+In/7y8T//mZ2dnbh582ZcuXJloVkhO7cQoWN1U+LmrfsREXFQ78ZbB9+OGwffifvT\
a3HSEntgnluI0LGmlLj34DAezi7FX/deicPm0c4c7x1/I/b7H8ag/8eY1V5uCf+LKzBYgfG0ib/t\
fz8Om8/e0FxFXUZx8cUfxe6FZ1c9HqQgYLAiszL/LFh/sNH6xmZgnoDBCrxz635Mju7NHR/1jqIf\
3swMpyFgsAK37z+Mrwx/G5eGt6KKOqI0UY/vxZXxr2NQHr8MH3jkqVnEcfv27VWPwFNgMpmc6nMP\
Do7j7ke342D8i/jLv7bi1r2j2P/4n/H+B+/G/tHp/o9SSty9ezca+yfyBK5du7bqERb21ATstdde\
W/UIPAXu3Llzqs9NZ0385Oe/idKUmDVNLPI05nQ6jddffz12dnbO/sPwqVdffXXVIyzMg8zwOTrN\
g8yfFw8y87TzHRgAKQkYACkJGAApCRgAKQkYACkJGAApPTXPgUEXrl+/HpcvX+7kd21tbcXGxkYn\
vwv+H3kODICU3EIEICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMg\
JQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAl\
AQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUB\
AyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQED\
ICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMg\
JQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgpf8CWG34RUtA\
1wIAAAAASUVORK5CYII=\
"
  frames[27] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKJElEQVR4nO3dy29cZxnA4febGY/txLmX1i1E9IKK\
oKCCWKGKJfsi8Vd03w1C4j9AYtEdu26APWKDKlZlwUWghkhBatOkpU0c10l8t+fysQgLik+IL5kz\
fSfPszzfWPMubP3GZ853Tqm11gCAZDrTHgAAjkPAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnA\
AEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAA\
SEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABI\
ScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJwABIScAASEnAAEhJ\
wABIScAASEnAAEhJwABIqTftAeBJMNrfjcH2/Rjub8fpp74apeOzI5yUgMEE1PEoNleux/adG7G1\
ejN279+Owda9qHUc3/zRT6J/+vy0R4T0BAwmYLi7Ff/87S9iPNz73PHO3ELsrd8RMHgMnMeACWk6\
TTge7Mb6J9emMA3MHgGDCej0F+LMsy83L9Zx1FrbHQhmkIDBBHQ63eiffapxbbS/EyFgcGICBpNQ\
OtHrn2pc2ln7JMaj/ZYHgtkjYDABpZTozs1HRDmwtnn7/RgP9g7+EHAkAgYTsrT8tej2FxrXnECE\
kxMwmJD+0oUone6B47XWGO1tT2EimC0CBhPS7Z+KUhr+xGqNvfXV9geCGSNg0LI6Hsb6x1enPQak\
J2AwIaXbi7Nf+Ubj2ni0by8YnJCAwYSU0onFi19uXBvt70Ydj1qeCGaLgMEEdeeb94Ltba65lB5O\
SMBgQkopUcrBqxAjIrZWrsdgd6PliWC2CBhM0NLTL8Tc4tmGlep2UnBCAgYTNHf6fHTm5g8u1AeP\
XAGOT8BggrpzC42bmSNqbK3eaH0emCUCBpNUIrr9xcalDc8FgxMRMJioEucuv9K8VKu9YHACAgYT\
Nn/mIc8FG+5FHQ1angZmh4DBhD1sL9hwZzOG+zstTwOzQ8Bggkop0fRMsIiI3bufxP7mWrsDwQwR\
MJiwxQvPRv/MpQPHax27nRScgIDBhM2dPh+9haXGteGOu3HAcQkYTFinOxedbq9xbefupy1PA7ND\
wGDCSilROs0Bu//RlZangdkhYNCC889/p3nBXjA4NgGDFiyce6bx+Hg08FgVOCYBgxY87HZSg+11\
l9LDMQkYtKDT7Ubpzh04Pti+F7vrd6YwEeQnYNCC/tKlWLzwXPOi78HgWAQMWtDtLz50L5gnM8Px\
CBi0oHS60ek1X0q/d3+l5WlgNggYtKCUEnOnzjeurX98NSKcQoSjEjBoyZnnXo6mG/uOR4MI34HB\
kQkYtORhzwUbj4YeqwLHIGDQku7cYuOTVcaD3djf+Kz9gSA5AYOWlFKilO6B48Pdzdhe/WgKE0Fu\
AgYt6Z06G0vLLzWu1RjbCwZHJGDQkm6vH/3TzVciPngumIDBUQgYtKR0utHp9hvXtldvRh2PW54I\
chMwaNHDbuq7efuDqONRy9NAbgIGLTp7+ZXmh1vWGk4hwtEIGLRo/sylKJ2Df3a1jmO4szmFiSAv\
AYMWdebmG4+PBnuxuXK95WkgNwGDFpVON3qLZw8cr6NB7N6/PYWJIC8BgxZ1e/Ox9PQLzYvVXjA4\
CgGDFpVuN/pLFxrXBjubEdWl9HBYAgatKtHpNe8F21tfeXBneuBQBAxaVEppvow+IjZvvx/jwX7L\
E0FeAgYtO3f5leYNzfXB5fTA4QgYtKy/dLHxv7BaxzHYWZ/CRJCTgEHLOr1+RGl4MFgdx/bqzfYH\
gqQEDFpWSone/KkDx+t4FFsrH7Y/ECQlYNCy0unGucvfal6s1V4wOCQBg7aVTvSXLjYuDXY3o7qU\
Hg5FwGAKHrYXbGftoxjubbU8DeQkYNCyUkrMLS5F6c4dWNvb+CxGg70pTAX5CBhMweKly9FbON28\
6MnMcCjNtwQAjmVjYyPee++9R76ujgbR2x8d/ARZa1z5259isPClQ73f8vJyvPjii0cfFGZAqS55\
gsfm3Xffjddee+2Rr+uUEr/62Y/j+eXznzs+rjV+/us/xm/+8I9Dvd8bb7wRb7311rFmheycQoQp\
2R+MYlw7cWvv+bi6+f34YPvbMayL8b2vPzvt0SAFpxBhCsa1xu//cj3KpR/G+9vfjRqdiKixsv9C\
DOJmlIhwagT+P/+BwZT8/dOz/4lXNyJKRHTi7vCZ+Nf4B7Ew77MlPIqAwZTs7Md/4vXfSjx14UJc\
ONNwt3rgcwQMpmQ42IzRcOd/jtZ46ekSlwQMHknAYErWbl+N87u/i37ZiYgape7H0vBKLO28E3uD\
0bTHgy+8J+ZE+61bt6Y9Ak+AtbW1Q7/2/tZuDO/9OQb3bsZfP4y4tXo7NlavxMd37sb61uHuxrG9\
ve13mxNZXl6e9gjH9sQE7O233572CDwBbty4cejXjsY1fvrLd6JGjdGoxvgYWzKvXbvmd5sTefPN\
N6c9wrHZyAyP0WE3Mj8uNjLzJPMdGAApCRgAKQkYACkJGAApCRgAKQkYACk9MfvAoA0XL16M119/\
vbX3e/XVV1t7L/iisQ8MgJScQgQgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMg\
JQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAl\
AQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUB\
AyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQED\
ICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMg\
JQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyCl\
fwOU7/3A/7BZiAAAAABJRU5ErkJggg==\
"
  frames[28] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKRUlEQVR4nO3dTW8cdx3A8d/sg5/thDpNk7ahtAUi\
FJA4teIAB06c6KFHXkdeABduvAMuqOJWceCABDfEhWOpKG3UNrQpQYmdR9tre3e9MxxaCZWdqLbT\
nclv8/lIucxf0f4Otr7y7H/+U1RVVQUAJNNpewAAOA0BAyAlAQMgJQEDICUBAyAlAQMgJQEDICUB\
AyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQED\
ICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMg\
JQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAl\
AQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUB\
AyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQED\
ICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMg\
JQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAl\
AQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUB\
AyAlAQMgJQEDICUBAyAlAQMgJQEDIKVe2wPAPKrKSezf/SxGgwcx3n8Y4/2HMRo8jGoyjpd+/Ivo\
9hfbHhHSEzCYgaPDQXzwh19HeTT80vXu4kpc3P1ZLD/zfEuTwfxwCxFmoYgoOtO/XuV4GHtb11sY\
COaPgMEMdPvLsfHilanrVTmJ0d69qKqqhalgvggYzEDR7cXixrnatcnoMELA4LEJGMxAURTR6fZr\
1/bv/TvKyajhiWD+CBjMyMq5S9HpLUxdH9y+HuV4WPM/gJMQMJiR5WdefORfYVVVNjwNzB8Bgxnp\
L69H0elOXa+qMkZ791uYCOaLgMEsFcXUpaosY+/2xy0MA/NFwGBGOt1enPnmD6YXqjIOH9yylR4e\
k4DBrBSdWP7GxdqlyXgY4XsweCwCBjNSFEV0HnHm4XBnOyajg4YngvkiYDBDSxvno9tfmro+2P4k\
xgd7LUwE80PAYIaWzl6I7uJq/aJbiPBYBAxmqL+8Hp1ezbNgVcRocK/5gWCOCBjMVPHFv/9Xxd7t\
fzU9DMwVAYNZKoo4c2n6VPqIiP27nzU8DMwXAYMZWzl3qfZ6eTSOcnLU8DQwPwQMZqy7sFJ7fbz/\
II4OBw1PA/NDwGCGiqKI/vJGdBenI3Zw72aMdu+0MBXMBwGDGVvcOBf9lbO1a06lh9MTMJix3tJq\
dBemH2aOiBgNHjQ8DcwPAYMZK4pOFDWn0kd8fiIHcDoCBg1Yv3i59vpg63rDk8D8EDBowOr5l2uv\
V5NJlJNxw9PAfBAwaEBvqf48xKPhfoz3dxqeBuaDgEEDugvLtYf6Dne24+DezRYmgvwEDBqwsHI2\
ls6cr1mpoipLb2eGUxAwaEB3YTl6S2u1a+N9W+nhNAQMGlB0OlF0urVrgzsO9YXTEDBoyKN2Iu7d\
+jAi3EKEkxIwaMjac69G3bvBqqpyKj2cgoBBQ/rL67XvtizHwxg7UgpOTMCgIZ1uP3o1W+nHBw9j\
7/bHLUwEuQkYNKS3tBYrmzUvt6yqKI/GttLDCQkYNKTTW4j+av1rVcYHO2EjB5yMgEFTiiI6nV7t\
0oNP341qMml4IMhNwKAhRVHE8jPPRxTTv3bjwX0vt4QTEjBo0NqFb9c+0FxVVVS20sOJCBg0qL9y\
pvblltXkKEaDey1MBHkJGDTpEW9mnowPYufmBw0PA7kJGDSo21+MtQvfmV6oqpiMDm2lhxMQMGhQ\
p7cQK5sv1K4dDQcRNnLAsQkYNKqITm+hdmX35gcxGR82PA/kJWDQoKIoYnH9XO1OxOHuXc+CwQkI\
GDRs9blXotNfrFmpoixtpYfjEjBo2MLKmShqTuSoqipGu3damAhyEjBoWs1JHBER1WQcD2+81/Aw\
kJeAQcOKTjfWL7xau3Y03LOVHo5JwKBhRacbq+dfqV2bjA6jKm3kgOMQMGhBd2G59vpg+9M4Otxr\
eBrIScCgYUVRRH95PYpuf2ptuLvtWTA4JgGDFqyceyl6S2u1a24hwvEIGLSgv7JRfyJHFTHcsZUe\
jkPAoAVF0XnEwfRVPPzsH02PAykJGLShiFh99lu1S+P9h7bSwzEIGLSiiPWL361dKcdDb2eGYxAw\
aMmjNnEc7mzFaP9Bw9NAPgIGLSiKIroLS7Vb6Ue7d+Nof6eFqSCX6RNFgcdycHAQ77zzzld+j1UO\
B9HrLEV3Mp5ae//992J8/faxPm9zczMuX758qlkhs6LybTF8ra5duxZXrlyJyVe826vTKeI3V38e\
33/5/JeuV1UVv/ztX+KPf/vwWJ/35ptvxttvv33qeSErtxChJWVZff6vKmJrdCn+ufej+Gj/hzEs\
V+P1773Q9njwxHMLEVr00X/ux8bzP41rg9ejjG5EVHFr+GqcObvd9mjwxPMXGLToz+8Ov4hXLyKK\
iOjE7mQzPhr+JBb73bbHgyeagEGL7u4Mv4jXl53dOBMXN9dbmAjyEDBo0eToMCbjwdT1lzarePHZ\
jRYmgjwEDFq0++CTWNv9fSwWg4goo6jGsXx0Lc4e/ikOhtPb64H/eWo2cdy6davtEXhKbG9vH/ss\
w92DUQzu/j2Whnfi3U+6cXNrK/buvhc3t+7E/b3jvRfs8PDQzzenduHChbZHOLWnJmBvvfVW2yPw\
lDhJwMqyil/97q8RETGZVFGe4rHMGzdu+Pnm1K5evdr2CKfmQWb4mh33QeaviweZeVr5DgyAlAQM\
gJQEDICUBAyAlAQMgJQEDICUnprnwKApa2tr8cYbb0RZlo183muvvdbI58CTxnNgAKTkFiIAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp/Rc32xNXptSPBAAAAABJRU5ErkJggg==\
"
  frames[29] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKY0lEQVR4nO3dSY8cZxnA8ad6mc0zmYztjBdlxUkQ\
kERRJPCBAxIS4prvkAP5AvkEnDjwAXLOGQkQIhJHJKRICAIoFijeiK14X2Y8PVtPdxeHcIGuxLOk\
q/K0fz/Jl/dtq5+DR393zVvVRVmWZQBAMq2mBwCAwxAwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUuo0PQBMo93eg9hdux39rfXY+++f/tZanHzlfCw//0YURdH0\
iJCegMEE3Pzr7+PuP/84tt6dW4rl516LKNoNTAXTxSVEmIC55dWIik9ZD69+HKPhoIGJYPoIGEzA\
4umXo2iNX+Aoy1FEWTYwEUwfAYMJmFk8Xvl7rnI0jL2t9QYmgukjYDABRasd7dmFsfVhfyvWrv2j\
gYlg+ggYTEB7Zi6Wznx7fKMsY9jfjtJlRDgyAYMJKFqdmF06Ubk32OlFlKOaJ4LpI2AwAUVRRKsz\
U7m3cePTGO7t1DwRTB8BgwlZOvtqtLpzY+u7G/eidJQejkzAYELmllej1e6OrZdlGYP+dgMTwXQR\
MJiQVme2+pFR5Sg271ytfyCYMgIGE1K02nHs1Lmx9XI0jN6tSw1MBNNFwGBCilY7jq2+WLk3GuxF\
OXISEY5CwGCCOrPHKtf7vQdOIsIRCRhMSFEUMXPs6crj9L3bl2Ow/aiBqWB6CBhM0MIzL0Rnfmls\
vSxHnkoPRyRgMEHducXKo/RRRuw8ulv/QDBFBAwmqSiiPTP+UN+IMtY/81BfOAoBg4kqYuVbb1Xu\
DHZ6HuoLRyBgMGGzSycr1we7m04iwhEIGExQURTRarcr93bW78Te5lrNE8H0EDCYsPnjz8bsU8+M\
rQ+2H8Vgd6uBiWA6CBhMWHdhOTpzi5V7e1s+gcFhCRhMWKvdiaJV/aP24NKfa54GpoeAQQ0WT71c\
ub63te4kIhySgEENls6+Wrk+Gg5iNOjXPA1MBwGDGrRn5ivX+737sbN2q+ZpYDoIGNSgPTMf7Yon\
0w92etHfXHMZEQ5BwKAGs0snY+HEs5V7Izczw6EIGNSg1ZmJdneucm/z7mc1TwPTQcCgBkVRROtL\
ArZ+/ZOIcAkRDkrAoCYnXvlBRBTjG2UZ5WhY+zyQnYBBTWaXTlb2azjYjX7PEzngoAQMatJqd6NT\
cRJxb2s9ercvNTAR5CZgUJPO3GIsnHx+fKMsYzToO0oPByRgUJNWdzZmjj1dudfvPYgQMDgQAYOa\
fOVJxGufOMgBByRgUKPl574XRasztv7FQ31HDUwEeQkY1Ghu+VTlV6uU5Sj2tjcamAjyEjCoUas7\
W7k+3NuNLU/kgAMRMKhRuzMbi6fOja2Xw73YXrvZwESQl4BBjVqdbsytnK3cG+5uOcgBByBgUKei\
FZ3Z6u8G2354I0bDQc0DQV4CBjUqiiLmnj4dRXv8JGLv9uUY7e02MBXkJGBQs2OrL0WrU3GYoyxj\
OBAw2C8Bg5p155eiaLXH1svRMDZufNrARJCTgEHNiqIV3fmlsfVyNIzNO1cbmAhyEjCoWdHqxPJz\
r1XulaOhJ3LAPgkY1K0oYmbpeOXWzqO7Mezv1DwQ5CRgULOiKKLVnomqb7fs3boUe9uP6h8KEhIw\
aMDS6XPRXXiqYqeMcuQSIuyHgEEDZhaPf8lR+oj+xr36B4KEBAwaULQ6UbTGLyFGlPHgyl9qnwcy\
EjBoQlHE0tnvVG4N+1tR+nZmeCwBg4Ysrr5UuT7c3Y6RJ3LAYwkYNKQ9M1e5vv3g8+hv3K95GshH\
wKABRVFEd2E52rMLY3uD3U33gsE+CBg0ZH7lbMwsnqjc29vZqHkayEfAoCGt7my0Ot3Kvd6tyzVP\
A/mMfykRcGRXrlyJW7duPfZ1vZv3YqXiv5E3L34c10dn9v1+r7/+eiwtjT8gGKZZUTqvC1+7d999\
N95///3Hvu6n3z8XP3/nx2PrF67eiZ/98nfRHwz39X4fffRRnD9//sBzQmY+gUGDbj/cjIiI3dFc\
fL7zauyMjsXx7s14arEXJ5cX4sZ9vwuDLyNg0KDhaBRbg/n4e+8nsTZYjYgiru18N84trMSLZ/4k\
YPAVHOKABl2/8yh++6+XY21wKr74cSyijHZc3n4rNuOFpseDbzQBgwatb+7E7fVB/P9Xq5TRjmdX\
q78zDPiCgEGDyjJitngUEf97lqpd9ONHb6xGUfW8XyAiBAwa9+j6r+PMzMVoxV5ElBHDXpwe/iFm\
BtebHg2+0Z6YQxz7uScHvi5bW1v7fu3fPr0Wb//wN3Hn3/Nx4fpWbDy8GnfvXIob9zZivze53L9/\
379xDuX06dNNj3BoT0zAPvjgg6ZH4Aly8eLFfb/28o0H8c4vfhXDURnD4SgOc2Pmhx9+GBcuXDjE\
3+RJ99577zU9wqG5kRkmYL83Mn9d3MjMk8jvwABIScAASEnAAEhJwABIScAASEnAAEjpibkPDOr0\
5ptvxttvv13b+62srNT2XvBN4T4wAFJyCRGAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQE\
DICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQM\
gJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyA\
lAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICU\
BAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQE\
DICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQM\
gJQEDICU/gP2jv8xG82d5wAAAABJRU5ErkJggg==\
"
  frames[30] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKeklEQVR4nO3dzW5cZxnA8efMjMcfseM4SdOkTdtA\
o6Kipo3YZMECsWDBLreAWNAb6AWwZMEFdN01EgikSuwrFSFUQM2izRdNaOomcT7s8dgez8xhUQmp\
+JQ4duecPpPfb3neieZZZPTXjN/znqIsyzIAIJlW0wMAwEEIGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACl1mh4AplFZljHe3Y7hTj92t9ajv/bv6N/7LI69ciGW\
X34ziqJoekRIT8BgAu5/8kGsXf1L7Kzfj8Hmw4hyHBERRasdyy+9EVG0G54Q8vMTIkxA//6t2Ljz\
SQx6a/+NV0TEw5sfxXg0bHAymB4CBhNw9MXXo2jt/ZZVluOIsmxgIpg+AgYTMLfyQnXAxqPY7T9u\
YCKYPgIGE9DpzldeHw368ejWP2ueBqaTgMEEtDozsXDy5b0LZRmjwVaUfkaEQxMwmIBWpxsLJ16q\
XBtu9762sQM4GAGDSSha0ZlbrFzauPNpjHa3ax4Ipo+AwQQURRHdxeOVGzl2Nu5HaSs9HJqAwYQs\
nTkfrZm5PdfLsozhYKuBiWC6CBhMyMz8cuU3sCjHsXn3Zv0DwZQRMJiQotWq3E5fjkfRW73WwEQw\
XQQMJqRodeLYubcq18bD3SjHdiLCYQgYTEpRRHfpZOXSoPfATkQ4JAGDCSmKIlpVfwOLiN6X12O4\
tV7zRDBdBAwmaPH5V2Nm/uie62U5dio9HJKAwQR1j56MVtW5iGXE9vq9+geCKSJgMEGtVucbnr5c\
xuPPHOoLhyFgMElFEYunz1cuDbd7DvWFQxAwmLClM69VXh/ubNqJCIcgYDBh7e7e46QiIrYf343d\
zUc1TwPTQ8BggoqiiM7cYuWZiMOt9Rju9BuYCqaDgMGELZw4G3PLpyrXdvu+gcFBCRhMWHtmLlqd\
mcq1B9f+WvM0MD0EDGqwcOLlyuu7/cd2IsIBCRjU4Ju20o9HwxgPBzVPA9NBwKAG3cWVyuuD3lps\
P1qteRqYDgIGNWjPzEa74kip4XYvBpuP/IwIByBgUIPu4omYWzlTuTZ2MzMciIBBDdrduejMLVau\
bd77rOZpYDoIGNSgKFrRrriZOSLi8e2PI8JPiPC0BAxqsnLuYkRUnExfllGOR7XPA9kJGNRkbvn5\
yuuj4U4Mek7kgKclYFCTojNT+QVst/84el9eq38gSE7AoCYzc4tx5LlzexfKMsbDga308JQEDGrS\
7s7H7NHqQ30HvQcRAgZPRcCgLkUr2p1u5dLjWx/byAFPScCgJkVRxPyJsxHF3o/dV4f6jhuYCvIS\
MKjR0guvRdFq77leluPY3dpoYCLIS8CgRp3ZI1EUe7cijnZ3ou9EDngqAgY1Klqd6Mwv7blejnZj\
69EXDUwEeQkY1Kg9Ox/LZ9+oXBvt9G3kgKcgYFCjomjFzMLeb2AREVsP78R4NKx5IshLwKBGRVFE\
0e5UrvW+vB7j3Z2aJ4K8BAxqduyVt6LdXdi7UJYxGgoY7JeAQc1mF09Ufgsrx6PYuPNpAxNBTgIG\
NStarcqHW5bjUWzevdnARJCTgEHNilY7ll98vXKtHI+cyAH7JGBQt6IVs8vPVS5tr9+L0WC75oEg\
JwGDmhVFEa1ON6oeDtZbvRa7W+v1DwUJCRg04Mip71eeyBFRRjn2EyLsh4BBA2aXjkd7ZnbvQhkx\
2Lhf/0CQkIBBA1rtbhStqo9fGQ9u/K32eSAjAYMmFEUsnflB5dJo0I/S05nhiQQMGnLk1Pcqr492\
tmLsRA54IgGDhnRmK46TioitB5/HYGOt5mkgHwGDBhRFETMLRyvPRBzubLoXDPZBwKAhc8fORHfx\
eOXa7vZGzdNAPgIGDWl356M1061c661er3kayKf6wUTAody4cSNWV1ef+Lre6v1Y2XsgR3xx9aO4\
PT6z7/e7cOFCLC1VPygTplVR2q8L37q333473n333Se+7ueXzsevf/HTPdev3Lwbv/rtn2IwHO3r\
/T788MO4dOnSU88JmfkGBg1aXetFRMTOeC4+334ttsdH4vjMF3F0sRcnlxfizpq/hcE3ETBo0HA8\
jv5wPv7R+1k8Gp6KiCJubf8wXl1YiXNnPhAw+D9s4oAG3b67Hn/85Hw8Gj4fX30ciyijHde3fhSb\
8UrT48F3moBBgx5vbsfqo2H876NVymjH2VPVW+yBrwgYNKgsI2aL9Yj4+l6qdjGIn7x5KoqKHYrA\
VwQMGrZ++/dxpns1WrEbEWXEqBenR3+O7vB206PBd9ozs4ljP/fkwLel3+/v+7V///RWXP7xH+Lu\
v+bjyu1+bDy8GffuXos79zdivze5rK2t+T/OgZw+fbrpEQ7smQnYe++91/QIPEOuXr2679dev/Mg\
fvmb38VoXMZoNI6D3Jj5/vvvx5UrVw7wL3nWvfPOO02PcGBuZIYJ2O+NzN8WNzLzLPI3MABSEjAA\
UhIwAFISMABSEjAAUhIwAFJ6Zu4DgzpdvHgxLl++XNv7rays1PZe8F3hPjAAUvITIgApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACn9B4wBEU4T/sSrAAAAAElFTkSuQmCC\
"
  frames[31] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKeklEQVR4nO3dzW5cZxnA8efMjMcfseM4SdOkTdtA\
o6Kipo3YZMECsWDBLreAWNAb6AWwZMEFdN01EgikSuwrFSFUQM2izRdNaOomcT7s8dgez8xhUQmp\
+JQ4duecPpPfb3neieZZZPTXjN/znqIsyzIAIJlW0wMAwEEIGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACl1mh4AplFZljHe3Y7hTj92t9ajv/bv6N/7LI69ciGW\
X34ziqJoekRIT8BgAu5/8kGsXf1L7Kzfj8Hmw4hyHBERRasdyy+9EVG0G54Q8vMTIkxA//6t2Ljz\
SQx6a/+NV0TEw5sfxXg0bHAymB4CBhNw9MXXo2jt/ZZVluOIsmxgIpg+AgYTMLfyQnXAxqPY7T9u\
YCKYPgIGE9DpzldeHw368ejWP2ueBqaTgMEEtDozsXDy5b0LZRmjwVaUfkaEQxMwmIBWpxsLJ16q\
XBtu9762sQM4GAGDSSha0ZlbrFzauPNpjHa3ax4Ipo+AwQQURRHdxeOVGzl2Nu5HaSs9HJqAwYQs\
nTkfrZm5PdfLsozhYKuBiWC6CBhMyMz8cuU3sCjHsXn3Zv0DwZQRMJiQotWq3E5fjkfRW73WwEQw\
XQQMJqRodeLYubcq18bD3SjHdiLCYQgYTEpRRHfpZOXSoPfATkQ4JAGDCSmKIlpVfwOLiN6X12O4\
tV7zRDBdBAwmaPH5V2Nm/uie62U5dio9HJKAwQR1j56MVtW5iGXE9vq9+geCKSJgMEGtVucbnr5c\
xuPPHOoLhyFgMElFEYunz1cuDbd7DvWFQxAwmLClM69VXh/ubNqJCIcgYDBh7e7e46QiIrYf343d\
zUc1TwPTQ8BggoqiiM7cYuWZiMOt9Rju9BuYCqaDgMGELZw4G3PLpyrXdvu+gcFBCRhMWHtmLlqd\
mcq1B9f+WvM0MD0EDGqwcOLlyuu7/cd2IsIBCRjU4Ju20o9HwxgPBzVPA9NBwKAG3cWVyuuD3lps\
P1qteRqYDgIGNWjPzEa74kip4XYvBpuP/IwIByBgUIPu4omYWzlTuTZ2MzMciIBBDdrduejMLVau\
bd77rOZpYDoIGNSgKFrRrriZOSLi8e2PI8JPiPC0BAxqsnLuYkRUnExfllGOR7XPA9kJGNRkbvn5\
yuuj4U4Mek7kgKclYFCTojNT+QVst/84el9eq38gSE7AoCYzc4tx5LlzexfKMsbDga308JQEDGrS\
7s7H7NHqQ30HvQcRAgZPRcCgLkUr2p1u5dLjWx/byAFPScCgJkVRxPyJsxHF3o/dV4f6jhuYCvIS\
MKjR0guvRdFq77leluPY3dpoYCLIS8CgRp3ZI1EUe7cijnZ3ou9EDngqAgY1Klqd6Mwv7blejnZj\
69EXDUwEeQkY1Kg9Ox/LZ9+oXBvt9G3kgKcgYFCjomjFzMLeb2AREVsP78R4NKx5IshLwKBGRVFE\
0e5UrvW+vB7j3Z2aJ4K8BAxqduyVt6LdXdi7UJYxGgoY7JeAQc1mF09Ufgsrx6PYuPNpAxNBTgIG\
NStarcqHW5bjUWzevdnARJCTgEHNilY7ll98vXKtHI+cyAH7JGBQt6IVs8vPVS5tr9+L0WC75oEg\
JwGDmhVFEa1ON6oeDtZbvRa7W+v1DwUJCRg04Mip71eeyBFRRjn2EyLsh4BBA2aXjkd7ZnbvQhkx\
2Lhf/0CQkIBBA1rtbhStqo9fGQ9u/K32eSAjAYMmFEUsnflB5dJo0I/S05nhiQQMGnLk1Pcqr492\
tmLsRA54IgGDhnRmK46TioitB5/HYGOt5mkgHwGDBhRFETMLRyvPRBzubLoXDPZBwKAhc8fORHfx\
eOXa7vZGzdNAPgIGDWl356M1061c661er3kayKf6wUTAody4cSNWV1ef+Lre6v1Y2XsgR3xx9aO4\
PT6z7/e7cOFCLC1VPygTplVR2q8L37q333473n333Se+7ueXzsevf/HTPdev3Lwbv/rtn2IwHO3r\
/T788MO4dOnSU88JmfkGBg1aXetFRMTOeC4+334ttsdH4vjMF3F0sRcnlxfizpq/hcE3ETBo0HA8\
jv5wPv7R+1k8Gp6KiCJubf8wXl1YiXNnPhAw+D9s4oAG3b67Hn/85Hw8Gj4fX30ciyijHde3fhSb\
8UrT48F3moBBgx5vbsfqo2H876NVymjH2VPVW+yBrwgYNKgsI2aL9Yj4+l6qdjGIn7x5KoqKHYrA\
VwQMGrZ++/dxpns1WrEbEWXEqBenR3+O7vB206PBd9ozs4ljP/fkwLel3+/v+7V///RWXP7xH+Lu\
v+bjyu1+bDy8GffuXos79zdivze5rK2t+T/OgZw+fbrpEQ7smQnYe++91/QIPEOuXr2679dev/Mg\
fvmb38VoXMZoNI6D3Jj5/vvvx5UrVw7wL3nWvfPOO02PcGBuZIYJ2O+NzN8WNzLzLPI3MABSEjAA\
UhIwAFISMABSEjAAUhIwAFJ6Zu4DgzpdvHgxLl++XNv7rays1PZe8F3hPjAAUvITIgApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACn9B4wBEU4T/sSrAAAAAElFTkSuQmCC\
"
  frames[32] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKdklEQVR4nO3du49cZxnA4ffMZWd3nc3axIltkhiw\
SJDIRREIRdAiJAqE0vEPRCgdVf4JWqrUkSgRFCgpaCNFSJAoUSLAia3YTnyP9zq7czuHIpWZMVmv\
d87JO36e8vvG3rfY1U8z5ztniqqqqgCAZFpNDwAAhyFgAKQkYACkJGAApCRgAKQkYACkJGAApCRg\
AKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAA\
pCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQk\
YACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRg\
AKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAA\
pCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQk\
YACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRg\
AKQkYACkJGAApCRgAKQkYACkJGAApNRpegBYROVkHJNBP8aD3djfvBH9W5/FeH8nnv7Zb6LVajc9\
HiwEAYMjNtzdjEvv/DH2t27GYOtWlKP9iIjorq7Hqed/HsvHTzU8ISwGAYMjVo72Y+OzD6IqJ3et\
j/qbsX31vIDBEXENDI5YZ2Utjj1xbuZeVU2iqqqaJ4LFJGBwxNrdXvTWHpu5N9zZiBAwOBICBkes\
aLWj1VmauXfn4j+iKsc1TwSLScBgDtbOPBPFjNOGk0E/qqpsYCJYPAIGc3Ds1LmZAauqMsb7Ow1M\
BItHwGAO2p3e7Hdgw/3Y/uJ8AxPB4hEwmINWtxePnH5mar0qxzHYvukkIhwBAYM5aLW797zfazLc\
dxIRjoCAwRwUrVa0u8sz9/q3L0c5HtY8ESweAYM5OfbEd2cep9+9cTHK8aCBiWCxCBjMyerJs/e8\
H2ziHRg8MAGDOWl3l6Mopv/EqrKM/Y1rDUwEi0XAYE6KVit6x09PrVflODYvfdjARLBYBAzmpGh1\
4vjZ52fuleORJ3LAAxIwmJOiKKLdW525N9zdcBIRHpCAwRytnPj2zIjt3rgYo73tBiaCxSFgMEcr\
J85Ep3dsan0y2otqPGpgIlgcAgZz1O6uzHwmYlQRg53b9Q8EC0TAYJ6KiOUZJxEjqti+6qG+8CAE\
DOaqiONnX5y5s79xzUN94QEIGMxZZ+WRmeuT4Z6TiPAABAzmqCiK6K2djO7K+tTe7s3PYrB5o4Gp\
YDEIGMxZb+2x6K4+OrVejvZj4qG+cGgCBnPW6i7f+6G+g37N08DiEDCYs6Io7vnllpuXP6p5Glgc\
AgY1WD/7wsz1/u1LTiLCIQkY1KC7On2IIyKimkyiKsc1TwOLQcCgBq12d+Z1sNHedgx37jQwEeQn\
YFCD5eOn4tjj351aH+7cjr07V+sfCBaAgEENWp1etHsrM/eqych1MDgEAYMaFEURvUefmLm3v3m9\
5mlgMQgY1GT9qeciophav3PxvYjwDgzul4BBTZYeOT6rX1GOh1GVAgb3S8CgJkW7G+3u8tR6OR7G\
eG+rgYkgNwGDmiytrsfamWen1kd7W7F742IDE0FuAgY1Kdrd6KysTa1Xk3GMBjtOIsJ9EjCoyVcn\
EU9GFNMXwka7m+EgB9wfAYMarZ35QRSt9tT6zvULUZVlAxNBXgIGNeqtfSuKYvrPbu/LK1GVkwYm\
grwEDGpUtDrRXpp+IkdVTmIy9N1gcD8EDGrUXlqJ9aefm1ovJ6PYu3OtgYkgLwGDGhWtdnSPnZha\
L0eD2L3+aQMTQV4CBjUqiiJaMw5xRERMxsOoKgc54KAEDGq2fvbFmdfBdq5/GuVo2MBEkJOAQc2W\
1x+Pot2ZWu/fvBTlWMDgoAQMala02tFZWp2xU0U5GdU+D2QlYFCzotWJ9e+8OLVelWXsffl5AxNB\
TgIGdSuK6K2dnFquynFsXPqwgYEgJwGDmhVFEa0Z18Aivrqh2UN94WAEDBrwyJlnoru6PrU+6m85\
yAEHJGDQgN7ayWjN+HLL7av/iVF/s4GJIB8BgwYUrVZ0etMnEcvxwEN94YAEDBpRxPEZJxGjihju\
btQ/DiQkYNCQ5fVTM1ar2HQSEQ5EwKABRVFE0e7O3Bts33ISEQ5AwKAhq489Fb1HH59aL0eDqDyR\
A76WgEFDlo4dj87ysan1wdbNGPW3GpgIcpl9NyXwQC5cuBDXrn39F1R2dvpR/M/aYPtW/PPv70S1\
Ov3u7F5eeOGFWFtbu88pIbei8mE7HLnXXnst3njjja993W9/9eN49Vc/umutqqr43R/ejnc/vnLg\
n/fuu+/Gyy+/fN9zQmY+QoQGvffJ1YiIGJTLcaH/Yny889O4PvxenHty+lmJwN18hAgNGgwnsTde\
ifd3fhEb4ycioohL+z+MJ78/iOJvH/iGZvg/vAODBn1+ayv+ev7Z2Bifiq/+HIuooh0rp34Z5879\
pOnx4BtNwKBBGzv7cW1jFPE/RznanW60O0vNDAVJCBg0aFJW0ak2I+Lus1TtYhjd1qCZoSAJAYOG\
bV/5S5xZOh+tGEVEGeVoOx4bvB3Lk8tNjwbfaA/NIY6D3JMDR6Xf7x/4te//+7P49U//HFc+7cXH\
l/uxdediXL7yr/hya+/A/8ft27f9jnMop0+fbnqEQ3toAvbmm282PQIPkfPnzx/4tZ988WW8+vs/\
xXhSxqQ83G2Zb731Vnz00UeH+rc83F5//fWmRzg0NzLDHBz0Ruaj4kZmHkaugQGQkoABkJKAAZCS\
gAGQkoABkJKAAZDSQ3MfGNTppZdeildeeaW2n3fixInafhZ8U7gPDICUfIQIQEoCBkBKAgZASgIG\
QEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBK\
AgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoC\
BkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIG\
QEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASv8FDW8VfMLMYMEAAAAASUVORK5CYII=\
"
  frames[33] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKSklEQVR4nO3dO49cZxnA8efMZWf2ajtrO44dEyMi\
TGIJgYRCAV+AAqVIxVegBOUrUNBTUEd0oUgJDQVFUCJQUiWG4ATHjm9re+29ze7snEOTxpoh7K53\
zvEz/v3K9xT7FDv6a855zztFVVVVAEAyraYHAICjEDAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABS6jQ9AMyacjSM7bUvY7j9KIbbj2Jv+1EMt9ejKNrxrZ/+Ilqt\
dtMjwkwQMDhmuxv345P3fhtRlU+sdxdPxUtbP4ve8mpDk8FscQsRjlm704u5xVNj6/uDzdi+f6OB\
iWA2CRgcs3Z/MRZWXx5br0bDGG4/iqqqGpgKZo+AwTFrtbvRnV+ZeG1/ZyMiBAyOg4DBMSuKIjr9\
pYnXttb+E1VZTrwGHI6AwRSsXHw9ivb4HqmtO9eiKkcNTASzR8BgCnorZ6Ioxj9eVVVFub/XwEQw\
ewQMpqDV6kQx4X2vajSM3cd3G5gIZo+AwRS05vqxfP7y2PpoOIjHNz9tYCKYPQIGU9Bqd6J/4uzE\
a+VoaCs9HAMBgykoila0Or2J1/a21m3kgGMgYDAli2deiVZnbmx948YnMdrbaWAimC0CBlMyv3px\
YsD2d7e8CwbHQMBgSjq9hf+5lX5/sNnARDBbBAympCha0Vs5M7ZelaPYuHW1gYlgtggYTEnRbsfy\
hdfGL1Rl7Dy4aSciPCUBg6kpYm7xxMQr5Wg/QsDgqQgYTElRFNHu9iMmPAcbPLprJyI8JQGDKVo6\
92rMLYx/C9u6ey2GOxsNTASzQ8BgiroLJ6LVnfxCc1S20sPTEDCYoqJoTdxKH1XE7sZa/QPBDBEw\
mKaiiJOXfjDhQhX3P/ug9nFglggYTNn8CxcmrvtdMHg6AgZTVBTFxOOkIiLK4a6IwVMQMJiy3srp\
6E7Yibi39dCRUvAUBAymrL9yJrqLJ8fWB+u3Y2/zYQMTwWwQMJiyot2NVqsz8Vo5cgsRjkrAYMqK\
oojeyumJ1waP7tY8DcwOAYManLr0w4nr6198XPMkMDsEDGrQWViZuD7a23EqPRyRgEENilYnignP\
wUbDQYx2txuYCPITMKhB/+SLsXj20tj63uaD2Fm/Vf9AMAMEDGrQ7vaj01scWx/t7cT+YNNtRDgC\
AYMaFEUx8V2wiHALEY5IwKAmKxdei4hibH3zzrX6h4EZIGBQk/6JsxPXN279MyLcQoTDEjCoSavT\
jaI94USOqopqNKp/IEhOwKAmnf5yLJ399tj6aLgbe5sPGpgIchMwqEm724u5pRfG1ofb67Fx+7MG\
JoLcBAxqUrTa0ep0J16rypGt9HBIAgY1Wn7pu1G02mPrw+1HYSMHHI6AQY3mVy9GFOMfu4ef/yOq\
0X4DE0FeAgY16vQXoyjG3wVzGgccnoBBjVrtbvRPnhtbr8oyyuFuAxNBXgIGNWp1e7F45tLYejUa\
xuCxH7eEwxAwqFFRtKI7P/7bYKPhIDZv2UoPhyFgUKOiKKIzvzRxI8dof9dzMDgEAYOarZz/XrS7\
/bH1vY37UZV2IsJBCRjUrLt0Kor2+LtgG7f/ZSMHHIKAQc2KohWt9viJHPs7m1GWDvWFgxIwqFmr\
3YmTr3x/bL2qyq9P5AAOQsCgbkUr+ideHFuuypEft4RDEDCoWVEUUUy4hRhVGTv3b9Q/ECQlYNCA\
hdMXo91bHFsv93ej8hwMDkTAoAH9k+ei01sYWx88vhejvUEDE0E+AgYNaHd6UbTGP35bd7+I4WCj\
gYkgHwGDJhQR/ZPnJ1yo3EKEAxIwaEQRJy5eGV+uInYfr9U/DiQkYNCQ7vzyhNUqHl77e+2zQEad\
pgeAWbOzsxMfffTR/z2Yd/fetegO96PXffJjuHbnq7jz/vsHPth3dXU1Ll++fOR5Iauicvw1HKur\
V6/GlStXYjT65mdZywtz8ftf/zxevfDCE+sffnozfvW7P8Xu8GDPwt5666149913jzwvZOUbGDRk\
Y3svtgfDKKsi1oYvx9reyzHX2omXX2rF+dPL8fmt9aZHhGeagEGD9ssqrg9ej6tbP44y2hFRxXL7\
OzE3/5eIEDD4JjZxQIP++LfHX8erExFFRLRiY7QaP/rJL5seDZ55AgYN+uyrx1/H60m9Cad0AE8S\
MGhSOYhuMX501Hx7s4FhIBcBgwbtbHwZJ7bfi16xFRFlRDWMzuCTWN37c9OjwTPvudnEcfv27aZH\
4Dlx7969A7/DdW99K+7e+DC68zfig38XcffBWty79XHcvHvw0zgGg4H/b47s3LlzTY9wZM9NwN55\
552mR+A5cZiAjcoqfvOHv0ZExP6ojKO8lXn9+nX/3xzZ22+/3fQIR+ZFZjhmB32R+bh4kZnnlWdg\
AKQkYACkJGAApCRgAKQkYACkJGAApPTcvAcGdVlaWoo333wzyrKs5e+98cYbtfwdeNZ4DwyAlNxC\
BCAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQED\
ICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMg\
JQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAl\
AQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUB\
AyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQED\
ICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDIKX/ArEJCsniP1HBAAAAAElFTkSu\
QmCC\
"
  frames[34] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKWElEQVR4nO3du49cZxnA4ffMzM5eY8e3eJ2ExJgQ\
LhKKaJAgVJRUkaj4A2giOpQK8Q/Qp6OLRE+BRJWONEig4CghkBgbO44TX9b2emc9148iKUjmRN7b\
nOEdP095vpH2LXb005n5zjdVKaUEACTTmvcAAHAQAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBK\
AgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoC\
BkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIG\
QEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBK\
AgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoC\
BkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIG\
QEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKnXkPAItmMh5G79bVGPbuxbB3Lwa9ezHs3Y2qasfzP/55VK32\
vEeEhSBgcMT627fjvT/8NqJMvnB9af1EnOv9NJY3Ts5pMlgsPkKEI1ZFFVU1/dYa93di9/bVOUwE\
i0nA4IgtrR2LjXMvTF2fjAYx6N2LUsocpoLFI2BwxFqd5eiu139MOO7vNjwNLC4BgyNWtVrRatd/\
vbxz60qUybjhiWAxCRjMwPpT5yNqvgd7cP2fAgZHRMBgBtZOP1e7Xb5ERPnS7kTgYAQMZqC7djyq\
qpq6XiajGO7cncNEsHgEDGaiiqgJ2GQ0iJ2bV+YwDyweAYMZaHVX4tgz35m6Xsaj6N//1FZ6OAIC\
BjPQandi5fjZ2rXxsB8hYHBoAgYzUFWtaHW6tWsPt67HZDxseCJYPAIGM7J64lxU7aWp69sf/ysm\
w4dzmAgWi4DBjKyd/tpX3oX5DgwOT8BgRpbWjtc/C1ZKDHa25jARLBYBg5mpYnojfUSZjGPn08tN\
DwMLR8BgRqp2J449+93phTKJ3TvXmh8IFoyAwYxUVStWTz1TuzYZDZyJCIckYDBD7e5q7fX+9u0Y\
24kIhyJgMCNVVUV3/WS0OstTaw8+uRTD3e05TAWLQ8BghlZPPh2dlfWalRLhVHo4FAGDGVpaPVb/\
LFiJGDiVHg5FwGCGqpoftfxMiZ2bl5scBRaOgMEsVVU88fS3apc8CwaHI2AwY+tnztden4yHMRmP\
mh0GFoiAwYzVb+KIGPXux7jfa3gaWBwCBjNUVVV0VjaitbQytda7cy3627fmMBUsBgGDGVs59lR0\
109ML5QSZWIrPRyUgMGMdVbWv/JEjmHPVno4KAGDGata7ahadefSR+zc+k/D08DiEDBowPpTF2qv\
P7jxQcOTwOIQMGjAE5vfqL1eJmNb6eGABAwa0Fk9Vnt93O/F6KFDfeEgBAwa0O4sR3tpeiPHw/s3\
o3f7ozlMBPkJGDRgaeNELD95dnqhTKKMR1FKaX4oSE7AoAGd7mosrWzUrg137zU8DSwGAYMmVK2o\
2p3apa1//y0i3IHBfgkYNKCqqlg7+Wzt2uDBnYangcUgYNCQjXPfjIjpB5pLKVFspYd9EzBoSHf9\
eF2/YjLqx6DnezDYLwGDxtS/3Ya9+7HzyaWGZ4H8BAwa0lndiPXTz08vlEmMR31b6WGfBAwa0umu\
xkrds2ARMdp9EHYiwv4IGDSlakWr3a1dunf1nSiTccMDQW4CBg2pqiqWj5+JqKZ3cvTv3/TjlrBP\
AgYNeuLci1G1ph9oLqW4A4N9EjBoUHfjRFQ1d2BlMorBg605TAR5CRg0qap/y40HD2P7+j8aHgZy\
EzBoULuzHOtnzk8vlEmM+j1b6WEfBAwa1Op0Y+3Mc7Vr434votjIAXslYNCkqor20krt0vaND2I8\
7Dc8EOQlYNCgqqpiae3JqFrtqbWHd2841Bf2QcCgYRubL0Srs1yzUqJMBAz2SsCgYd2NE1G1p+/A\
SinR3749h4kgJwGDhlVVK2p/F2w8ivvX3mt+IEhKwKBhVasV66frdiKWGO7et5Ue9kjAoGFVqxMb\
516oXRsPdh0pBXskYDAHneX12uu9Ox/FuL/T8DSQk4BBw6qqis7y+ldspf84RoPdOUwF+Uwfiw0c\
2Pb2dly8ePGRrxvv3otu1Y1WfClWJeLi39+OUefSnv7e5uZmXLhw4SCjQnoCBkfo4sWL8fLLLz/y\
dUudVvz+Nz+L588++YXrk8kkfvXLX8SfL17d09979dVX4/XXXz/QrJCdjxBhDsbjEqVETEorbvTP\
x7sPfhiXet+LYVmJH3z7mXmPBym4A4M5KFHiw+t3Y3jsJ/Fh7/tRohURJT4dfD1Ontjb3Rc87tyB\
wRyUEvHHt+PzeLXjswebW7E1OhuXhz+KpY63JjyKdwnMye3twefx+l9VnD11Kk4dW5vLTJCJgMGc\
jAY7MRl9ect8iRc3W3HmuIDBowgYzMnWzXfjRP9P0a12I6JEVQaxMXonNnbfjP7QaRzwKI/NJo4b\
N27MewQeA3fu3Nnza7e2d2O49ZcYbl2Jv16OuHHrk9i+9U5cu7kV93f29sOWvV7P/zaHsrm5Oe8R\
DuyxCdgbb7wx7xF4DFy5cmXPrx1PSvz6d29GiRLjcYnJAQ7xff/99/1vcyivvfbavEc4sKo4+hqO\
zFtvvbWnB5mPigeZeZz5DgyAlAQMgJQEDICUBAyAlAQMgJQEDICUHpvnwKAJJ0+ejFdeeaWxv/fS\
Sy819rfg/43nwABIyUeIAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQk\
YACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRg\
AKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAA\
pCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApPRfKQgF\
bsftProAAAAASUVORK5CYII=\
"
  frames[35] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKM0lEQVR4nO3dy44cVxnA8e9UX8Zje2xHtoPjGCug\
CJMgFNgBDxFFbHgBXoBFNigr/AJseACUB2ARIcEGwQIQAkJYxCSBQHAS24nHNp779KWKRVbJ1BDP\
pav52r/fsk5L/S1m9FdXn1NdmqZpAgCSqeY9AAAchoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoAB\
kJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQ\
koABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCS\
gAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKA\
AZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoAB\
kJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQ\
koABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCS\
gAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKA\
AZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoAB\
kJKAAZCSgAGQkoABkJKAAZBSf94DwCJrmiamo+0Yb69FPd6NkxeuRill3mPBQhAwOGb1dBwbd96N\
rdWbsbl6M3bX7sZo8z/R6w/jue/+MPrD5XmPCAtBwOCY7a7fi7d//uOIpv7U9WZ5JUbr96J//sqc\
JoPF4jswOGYlSuttwsn2emx+/K85TASLScDgmPWXV+LUk19qXWuiiaZpOp4IFpOAwTGr+sMYnnqi\
dW26u9XxNLC4BAyOWal60RueaF3bWn0/mnra8USwmAQMjlkpJapBe8DWb70tYHBMBAxm4MzTz0Wp\
9tvk6zswOA4CBjOwtHK+dSdi09QxHe3MYSJYPAIGM9BfOhnRFrDpJEYb9+YwESweAYOZaH9c1HS8\
E+u33ul4FlhMAgYzUA2WYuWpr7Su1dOxs2BwDAQMZqDqDeLEuUuta9Pd7QgBgyMTMJiFUqK3z0N7\
d9Y+jno67nggWDwCBjNQSolS9VrX1m+/E/XYTkQ4KgGDGTlz+Vr7geamcRIMjoGAwYwMV85H1dt7\
mLlpmpjsbM5hIlgsAgYz0hsuRykt/2JNHVurN7sfCBaMgMGMlFKianmob1NPY+PO3+cwESwWAYMZ\
KVUvzj79fPtiE86CwREJGMxKqWK4cr51aTLe9lR6OCIBgxna7yzYePNh1JNRx9PAYhEwmJFSSusT\
6SMiNu++F5Pt9Y4ngsUiYDBDpy4+E/3llT3Xm3oSTVPPYSJYHAIGMzQ8cyF6rYeZIyY7PoHBUQgY\
zFCvv7TPI6Wa2H5wu/N5YJEIGMxSiSjV3qdxREQ8fP9Gx8PAYhEwmKkS5555YZ+12lkwOAIBgxk7\
cfYLrdeno91o/KwKHJqAwQyV//G7YKONezHeXut4IlgcAgYzVvUHrd+D7a7djdHmwzlMBItBwGDG\
lp94at9HSoXvwODQBAxmrL90ev9HSrmFCIcmYDBjpdePqvUs2Ce3EYHDETCYsVJKDE6da117+MGb\
HU8Di0PAoANnLl9rvd5Mxs6CwSEJGHRguHKh9Xo9GUU93ul4GlgMAgYd2G8Tx2R3M8ZbNnLAYQgY\
dKCUElH2/ruNNu7H9n/uzGEiyE/AoANLZy7GyfNX9lltfA8GhyBg0IHecLn1hy0jwi1EOCQBgw5U\
vX5UvUHr2ubd97odBhaEgEFH9tvIsXHnHxHhFiIclIBBR85d/XpElPZF34HBgQkYdGTpzD5nwaaT\
mOxudTwN5Cdg0JGqv9T6AWyysxFb9z7ofiBITsCgI1V/EP2l03uu1+OdGK2vzmEiyE3AoCP9pdP7\
ngVrGmfB4KAEDDpS9QcxOHmmde2T3wUTMDgIAYOulCqq3rB1aefB7WjquuOBIDcBg46UUqL0+q1r\
67ffiaaedjwR5CZg0KFzz7wQpWqJWNOEW4hwMAIGHVpauRCl2vtv1zR1jLfX5zAR5CVg0KFqsNR6\
vZ6MYuf+rY6ngdwEDDpUStX6TMR6MorNe+/PYSLIS8CgQ73BiVi5/NX2RWfB4EAEDDpUer0Ynnqi\
dW28vWYrPRyAgEGnSlT99q30mx/9M+rJbsfzQF4CBh0qpcTg5Nkopbdnbefhx9FMJ3OYCnISMOjY\
6Se/HNWg/YkcTeMWIjyq9nsZwIHdv38/3nrrrc99XTPejsG03vPLKnU9ib/+6fcxHpx9pPe7evVq\
XLnS/nBgeByUxrYnOBavvfZavPjii5/7uhPDfvzs+vfi/NmTn7o+nkzjRz/9Tfzyj+8+0vtdv349\
XnnllUPNCovALUSYg9FkGtOmig93no0bG9+J97afj6hOxDeevTTv0SANtxChY6PJNH71l5vxzW99\
P27uPB9NVFGiidXRF6Mub897PEjDJzDoWF038fqtJ+PmzteiiV5ElGiiirvjq7FafTuG/b07FIG9\
BAzmYGv0SbQ+rcTlixfi9HL7DkXg0wQM5mAyWot6OvrM1Tqee7ofp08KGDwKAYM5+OiDP8TF6a9j\
UHYioonS7MbZyZ9juPnbmEycBYNH8dhs4rhz5868R2DBPXjw4JFfe39tO8ra72J782/x+r/ruLt6\
Kx6uvhkf3n0YG9uf/WTWbmNjw981R3bpUt6dr49NwF599dV5j8CCu3HjxiO/djSZxg9+8oto6iYm\
dR2HOY35xhtv+LvmyF5++eV5j3BoDjLDMXnUg8zHxUFmHne+AwMgJQEDICUBAyAlAQMgJQEDICUB\
AyClx+YcGMzapUuX4qWXXurs/a5du9bZe8H/I+fAAEjJLUQAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIw\
AFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAA\
UhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABS\
EjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFISMABSEjAAUhIwAFIS\
MABSEjAAUhIwAFISMABS+i8VLfi2pEmm0QAAAABJRU5ErkJggg==\
"
  frames[36] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJ+ElEQVR4nO3dvY5cZxnA8efMx47Xzm7imBhDIhwR\
gwiJqGkoaKgoApdAlIILSMUtcAnuIiFKpBQ0kIYiUhACJREfInFIgtd25MTe9e54Ps+hiIRwZpLY\
s3vO4Tn+/SQ37xlrnmJHf+3Oe95TVFVVBQAk02t7AADYhIABkJKAAZCSgAGQkoABkJKAAZCSgAGQ\
koABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCS\
gAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKA\
AZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoAB\
kJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQ\
koABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCS\
gAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKA\
AZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoAB\
kJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQ\
koABkJKAAZCSgAGQkoABkJKAAZDSoO0BoIuqqopyMYvlfBLlbBLL+SSqqoozT1yMoijaHg86QcCg\
Bh+9/Vp8cuVPsZzd/e+/wfZuPPfTX0R/61Tb40EnCBjU4O6ta3F4/Z171nrDUSwmhwIGJ8R3YFCD\
orf60VpOx3H31l4L00A3CRjUYOfr34mi179nrVzMYja+3dJE0D0CBjXYOnM2oljz8ao+3eABHJ+A\
QQ2Gp3fW7jZczsYtTAPdJGBQg15/a+367Oh2RFU2PA10k4BBgw6v/TOqctn2GNAJAgY1KPqDOPXo\
V1fW5+N934HBCREwqEFvMIztc0+trFcREQIGJ0LAoAZFrx/D7Z3VC1UZi+lR8wNBBwkY1KKI3mC0\
slqVy5iP91uYB7pHwKAGn3dg73I+jcOPrjQ8DXSTgEFNeoNhRHwmZFUZy+ndVuaBrhEwqMnuk8+u\
Pbi3Kpd2IsIJEDCoyXB7d+U8xIiIxeQwKjczw7EJGNSkP9qOYs15iPPxvpuZ4QQIGNRm/UaOwxtX\
opxPG54FukfAoCZFUURvzXdg5WLqPEQ4AQIGNSl6/dh98tm118rlouFpoHsEDOpSFDE8vbuyXFWV\
m5nhBAgY1KaIwanV46SqchmHH73XwjzQLQIGNSmKIoreuqcylzG5tdf8QNAxAgZASgIGNdo++2T0\
R2dW1svlPKrSTkQ4DgGDGo12zkV/uHoq/WJyGOVy3sJE0B0CBjUanDoTvf5wZX388VXPBYNjEjCo\
UVH0I9Y8WmU+vh3lfNbCRNAdAga1W3+kVIQT6eE4BAzqVBSx87Vvra5XEcvZpPl5oEMEDGo22n1i\
7frs6FbDk0C3CBjUbOvMY2tWqzi4+rfGZ4EuETCoWW/NNvqIcB4iHJOAQY2Koogv2sRRVTZywKYE\
DGo2GG1Hb7D6W1g5n0XlsSqwMQGDmo12z8fWI2dX1hfTcSzndiLCpgQMatbfWv8b2OT2tZgdftLC\
RNANAgY16w22otcfrKyXi5knM8MxCBjUrCiKL9jH4UR62JSAQROK9R+1+d07DQ8C3SFg0IDHvvG9\
tetO44DNCRg0YLT7lbXr+x++3fAk0B0CBg0Ybu+uXS9to4eNCRg04XO+A6uqKqrSRg7YhIBBAz7d\
ibj6cSvns1jO77YwEeQnYNCArUcej+3HLqysL+eTWEzHLUwE+QkYNKA/HEV/dHplfXb4cUxuX29h\
IshPwKABvcEo+lvbqxd8BwYbEzBoQNHrRdHrr71WLuceqwIbEDBo2fzodtsjQEoCBg05/fhTa9en\
d242PAl0g4BBQ04/cTHWnep7Z+/vEeFPiPCgBAwaMjzz6Np1mzhgMwIGDekPRmsfq1JVZZSLefMD\
QXICBo1Z/1CwcjGPxeSw4VkgPwGDhvS3TsVo9/zK+mJyJ45uvt/CRJCbgEFD+sNTMdo5t7JeLRex\
nBy1MBHkJmDQkKI/iMHozOdedzMzPBgBg4YUvX70hqO115YzJ9LDgxIwaEhRFFF8zkaO2dHtCL+B\
wQMZtD0AdMGNGzfi3Xff/dLXjf/1YexUVfSKe0N288pf4mr1taiK9eclftalS5fi/PnVDSHwMBEw\
OAGvvvpqvPTSS1/6uu9/96n45c9/FFvDe0N1Y++D+MnPfhjT+fK+3u/y5cvx4osvbjQrdIWAQYNu\
7o+jrKpYVoPYmz4TdxbnYmfwcWxVb3361GbgvgkYNGj/aBKLsh//OPxBXJs+E1UUUUQV2/OzsXP6\
NzGZ7bc9IqRhEwc0qCyr+GDyXOxNL0UVvYgooopeHA2ej28//+O2x4NUBAwatqgG8dljpYqiF4PB\
qXYGgqQEDBp0d7qId95/L4q4d7NGEcsY9cYtTQU5CRg0aLZYRhz8MZ45/ecYFNOIqopyMY5HJn+I\
ncWbbY8HqTw0mziuX7/e9gh02MHBwX29brEsY+/mQXxz77dx5b3fxV+vljE+vB57/34zbu7f/3mI\
BwcHfqY5ERcuXGh7hI09NAF75ZVX2h6BDnvjjTfu+7W/+v1b8evX3o5lWcay3Oz0jddffz0Wi8VG\
/xf+18svv9z2CBsrKieIwrFdvnz5vm5kPsn3cyMzDzvfgQGQkoABkJKAAZCSgAGQkoABkJKAAZDS\
Q3MfGNTp6aefjhdeeKGx97t48WJj7wX/r9wHBkBK/oQIQEoCBkBKAgZASgIGQEoCBkBKAgZASgIG\
QEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBK\
AgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoC\
BkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIG\
QEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASv8Bf2PoeOclQKYAAAAASUVORK5CYII=\
"
  frames[37] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJaElEQVR4nO3dT4skdxnA8aeqe6ZndpLJmixhCRqV\
LChxUW9eEgTBYy6+Ac++gOCL8BXoNUc9eBVFMXiVFTSQREkCms3ubMy6m56Z/ldVHhbx0D2bTfdU\
FU/l84G9VC/0c9jmy3b/6qmiaZomACCZsu8BAGAbAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBK\
AgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoC\
BkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIG\
QEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBK\
AgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoC\
BkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIG\
QEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZASgIGQEoCBkBKAgZA\
SgIGQEoCBkBKAgZASgIGQErjvgeAIWqaJupqGfVyFtVyHvVyHtVyHpOnn4v9o6t9jweDIGDQgpO/\
/T4+ee/PUS1nj+K1mEW1nMVXvvejeP7mD6Ioir5HhPQEDFpwfv+jmN75x9r15WwaEU1ECBjsym9g\
0KFqfhrRNH2PAYMgYNCGC74iPPvkw2jquuNhYJgEDFpw/MI3oihHa9fn/7kbTSNgcBkEDFqwd/Sl\
iMLHC9rkEwYtGB8cPeakod/A4DIIGLRgtH+48XrTNFEvFx1PA8MkYNCC4oJj8k1Tx2p+1vE0MEwC\
Bh1q6ipW89O+x4BBEDBoQVGWMT48XrteL+cx/ejvPUwEwyNg0IJyvB9H117c8MqjHYnA7gQM2lCU\
MT446nsKGDQBgxYUZRmj/SsXvNpEY50U7EzAoBVFlOP9ja9Ui5l9iHAJBAxa8LjHpazmp9ZJwSUQ\
MGjLBQ2r5mcW+sIlEDBoyZVrL278GnF68l7Uq3kPE8GwCBi0ZP/K1Y0b6evlPMJXiLAzAYOWjCZH\
UdhID63x6YKWjCdXIsrNHzG/gcHuBAxaUpSjjUt9m6aJ1cJCX9iVgEHnmljNLPSFXQkYdKyp65g/\
POl7DEhPwKAlRTmKo+e/vv5CU8fpyfvdDwQDI2DQkqIoY/LM832PAYMlYNCWoojxxEZ6aIuAQYvG\
k80b6Zu6tpEediRg0JKiKC68D6xanEdTrzqeCIZFwKAH1XIWTVX1PQakJmDQotF4ErFhndTy7EHU\
q0UPE8FwCBi06PC5L8f44Km16+f3b8dqbhsH7ELAoEWj/cMoR+O+x4BBEjBo0WjvYOMjVR5xChF2\
IWDQotHe5MKA1dWy42lgWAQMWrW+jT4iIpqI1Wza7SgwMAIGPbGRHnYjYNCmIqIcTza80MSnH73b\
+TgwJAIGrSrimRe/tfGV5dmDjmeBYREwaNnewdN9jwCDJGDQsk03Mv+Phb6wPQGDlpV7m34Di2iq\
ZTS1fYiwLQGDFhVFERcdpa+W82jcCwZbEzDoSb2cu5kZdiBg0LK9K8cx2vBk5vn037E8e9jDRDAM\
AgYt2z+6GnsbDnJU87OolvMeJoJhEDBoWTmeRDHe63sMGBwBg5aVe5MoRxcEzDF62JqAQcvKchRF\
ufmjVi3PO54GhkPAoEer80/7HgHSEjDoQFFufiqzfYiwPQGDDhy/8M2N1x/8862OJ4HhEDDowPhw\
80LfprFKCrYlYNCBvQsCBmxPwKAD4w2bOCIiorGRHrYlYNCFcrTxcl0to14tOh4GhkHAoEf1ahm1\
dVKwFQGDDhRFxKbHqtTVIqrlrPN5YAgEDDqwf/RsHFy9vnZ9Mf0kZg/u9jAR5Cdg0IFybxKjyeH6\
C00TTV13PxAMwOb1AMATWSwWcevWraiqx9/P1dSrGE3PYtNRjnfefjvq26dP9H7Hx8dx8+bNLSaF\
4SkaZ3hha3fv3o2XXnopTk8/O0A/+8kP4/vf+dra9Z/+/Lfxh1sfPNH7vfrqq/Hmm29+zilhmPwP\
DDrSNI/+3F9dj7vzr8a4WMULB+/G8ZVJ36NBSgIGnSni9vxGvHX6SlTNo+eD3Z7fiP2jv0TEO/2O\
Bgk5xAEd+eDjMt6avhJVsx+PjtQXcVY/E9du/DiKYv2IPfB4AgYd+ev7H8eqWf/SYzTecDoR+EwC\
Bh15eDqN/WL9CcwHIw+1hG0IGHRkfvZxfPfp38Vh+TAi6iiiimf3PoxvP/XHiHAYGD6vL8whjjt3\
7vQ9AgN07969J94m/697D+MXv/xVnNe/ifvL61EWq7i292Gczz6NJ72ZZbFY+LfMpbp+fX1DTBZf\
mIC98cYbfY/AAE2n01itVk/2d88X8es/vb3T+52cnPi3zKV6/fXX+x5ha25khh18nhuZL4MbmeH/\
/AYGQEoCBkBKAgZASgIGQEoCBkBKAgZASl+Y+8CgDZPJJF577bWYzWadvN/LL7/cyftABu4DAyAl\
XyECkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCS\
gAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKA\
AZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoAB\
kJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQ\
koABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCS\
gAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQ0n8BR425kw9Q6ysAAAAASUVO\
RK5CYII=\
"
  frames[38] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAIw0lEQVR4nO3dz28cZx3H8e/Mem0naUrVUnD5UWiF\
egCJ3kAqQnDj2it/Qjlx6Z/BCan/QK/cKiSEkJBAokWc2iIhhQOkpTRpihvixo7t3R0OuaB60yZZ\
Pzv6bF+v4+za/h48envGO8/TDcMwFACE6cceAAAehoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZA\
JAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQM\
gEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJ\
GACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABE\
EjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkY\
AJG2xh4ANtHRzWt1evvmmeO7j+3V9qXHRpgINo+AQQPX3/pd3fjbH84cf/oHP60vfefH1XXdCFPB\
ZnELEdZofnpUVcPYY8BGEDBYo8XJsX7BOREwWCNXYHB+BAwamF64XFVn/89159aNGobF+geCDSRg\
0MDFJ79RXX/29Dr68N0aFgIG50HAoIHJdLfKJw2hKQGDBvrpbi27hQicHwGDBibbu571gsYEDBqY\
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\
AAAASUVORK5CYII=\
"
  frames[39] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAH/0lEQVR4nO3dP49cVx3H4d+d/cPacSBy+GMLCSGE\
LBoj/A4iuaNLxxtxQUHHK6CmQS4sOrpIkIISJBc4hRusCKIosiMIcmKv7Z2deyicxtm11t69546+\
4+eRXOz17O5pjj57zj13ZmittQKAMIt1DwAATkPAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkY\
AJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQS\
MAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACI\
JGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoAB\
EEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQB\
AyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBI\
2+seAGyicXVYjx58XNXGF64PW9t14fs/qWHhb0c4KwGDDlbP9uufH/yuxsODF65vn/t2/fxXv62t\
3b01jQw2hz8DYWbtG6sy4HQEDGbVBAwmImAwp1ZVo4DBFAQMZmUFBlMRMJhZG1frHgJsBAGDObX2\
/B9wZgIGM2rlFCJMRcBgZrYQYRoCBrNq1WwhwiQEDObUqqpZgcEUBAxm1ap5DgwmIWAwM4c4YBoC\
BnNqzTtxwEQEDGbkGD1MR8BgZgIG0xAwmJVDHDAVAYMOFts7df67PzpyfVwe1P5/P1nDiGDzCBj0\
MCxqa/fcMf/RalweHHMdeF0CBj0MQw2D6QU9mWHQwVBVtTC9oCczDDqxAoO+zDDoYqjBCgy6MsOg\
h6FqGLbWPQrYaAIGXQxVw7DuQcBGEzDoZFhYgUFPAgadOMQBfZlh0MEwDFUCBl2ZYdCJU4jQlxkG\
XXgnDujNDIMehrKFCJ2ZYdCJLUToywyDLmwhQm9mGHTy8hVYq9barGOBTSRg0MEwDPX1e9IfoV0w\
DQGDubVx3SOAjSBgMLMmYDAJAYOZtXGsKvuIcFYCBnNro37BBAQMZtbGVSkYnJ2AwczcA4NpCBjM\
rLXRc2AwAQGDuY1WYDAFAYOZ2UKEaQgYzOx5wGwhwlkJGMxtdIwepiBgMDMrMJiGgMHM3AODaQgY\
zG10jB6mIGAwMyswmIaAQSfDYuvY6+NqWe6BwdkJGHTy1vd+XIvtbx25/vjzf9W4PFjDiGCzCBh0\
MmxtVQ3HfCqzU4gwCQGDTobB9IKezDDoZFgs6pj1FzARAYNehsXxW4jAJAQMOhmG408hAtMQMOhk\
WCyqbCJCNwIGvQwL/YKOBAw6eX4KUcGgl+11DwCStNbqzp07tb+/f/KLn31Z24eHRxK2Wo11+/bt\
atvnTvwRu7u7de3atdracj8Nvmlo3lUUXtk4jnX16tW6e/fuia+9/O6F+sOv3693Luy9cP3Js2W9\
/5s/1hdfPjn5Z1y+XPfu3avz58+fesywqazAoJPV2Kq1Vl8dvlOfPftptVrU5d2Pa7fur3tosBEE\
DDpZrVr9b/mD+vfyl/V0fKuqqj59+rO6svvnNY8MNoNDHNDJclzUR1+9V0/HC/X8MMdQy7ZXHz16\
rw7GvZO+HTiBgEEnq7HVsu0cvd52qjmdCGcmYNDJOI61t3h05Pre4nEtyodawlkJGPTSlvWLtz+s\
72x/XkOtaqixLmx9Udfe/kvtLJ6te3QQ7405xHH/vpNfnN04jnV4ePhKrz1Yrur3f/qwFtt/q/8s\
f1itDfXuzmf113pUj5+82gdajuNYDx48qHPnTn5mDE7j0qVL6x7Cqb0xAbt58+a6h8AGaK3Vw4cP\
X+m1q7HVB3+/9/VX/zjV79vf369bt27Vzs7Re2kwhRs3bqx7CKfmQWZ4Da/zIPMUPMgML+ceGACR\
BAyASAIGQCQBAyCSgAEQScAAiPTGPAcGU7l+/XpduXJllt918eJFH2YJL+E5MAAi2UIEIJKAARBJ\
wACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMg\
koABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIG\
QCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEE\
DIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAi\
CRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAA\
RBIwACIJGACRBAyASAIGQCQBAyDS/wGZsks7OtqTFAAAAABJRU5ErkJggg==\
"
  frames[40] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAH/klEQVR4nO3dv69UaR3H8e+ZuYDsFQsKlnUbf8Qt\
NARtLKhstKb1PzD0/Ak2llrwB9CbWEksjcSQGI0mGxQiiawIm+zeTQxc+TFzHotNTHYZ4uXuc87k\
M7xe5bnDzFNw8p7nOc85M7TWWgFAmMW2BwAAxyFgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnA\
AIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCS\
gAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZA\
JAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQM\
gEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJ\
GACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABE\
EjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAA\
iCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkCk\
vW0PAHbRuF7V4w/vVbXxM8eH5V59+dw3alj47ghflIDBBFZPH9fdX/+8xtXzzxw/cfordeHHP63l\
4tSWRga7w9dAmEAb19seAuw8AYMJtPVq20OAnSdgMIFxXFdrbdvDgJ0mYDCBNpqBwdQEDCbgGhhM\
T8BgAm29ripLiDAlAYMJWEKE6QkYTKCNaxMwmJiAwQRGMzCYnIDBBFwDg+kJGEzgyYf3Nu5E3D/3\
9RoWyy2MCHaPgMEE1i+ebjy+PLVfNQwzjwZ2k4DBjBaLZckX9CFgMKNh6QcgoBcBgxm5/gX9CBjM\
6NOAWUSEHgQMZmQJEfoRMJjRsFiagEEnAgYzWiz2SsGgDwGDGQ1LmzigFwGDGdmFCP0IGHTW2quf\
gTgsbOKAXgQMJtDauPkPQ9XgUVLQhYBBb61tfJAv0JeAQWetBAzmIGDQW2vVxlcsIQLdCBj01lq1\
ZgYGUxMw6KxVqzIDg8kJGPTWRtfAYAYCBr3ZhQizEDDorLX26vvAgG4EDLozA4M5CBj01lq19Wrj\
nwZPooduBAw6Wz17UocHD146vjy1X6fPvruFEcFuEjDorLVWbXx5BjYsFrXYO7mFEcFuEjCYzeDn\
VKAjAYOZDMNQw8IpB704m2Auw1DDYAYGvQgYzGSwhAhdCRjMZRiqLCFCN84mmMtgBgY9CRjMZBiG\
GganHPTibILZmIFBTwIGcxkWAgYdCRj01trGw8NQlhChI2cTdOZJ9DAPAYPOBAzmIWDQ2at+SgXo\
S8Cgs3HDk+iB/gQMOmvjumrzPg6gIwGDziwhwjwEDDqziQPmIWDQ2WgGBrMQMOis2cQBsxAw6Kyt\
LSHCHAQMOluvntWmbYjDYq9qmH88sKsEDDp7/PDuxuNn3nmvFAz6ETDorLVx4/HF3smZRwK7TcBg\
Jn5KBfra2/YAIMG9e/fq0aNHR3rt8uBg4zfDDx78q+6vfn+k97hw4UKdOXPmNUYIb56htVf8eBHw\
P1euXKlr164d6bU/+8kP6wff/dpLx3/xy1t1/Td/OdJ73Lx5sy5duvQ6Q4Q3jhkYTOD5+KX659P3\
6um4X2dPPKxzJ+/Xi9Xma2PA8QgYdPZsfKv++O8f1Serc1U11P2n365vvvWnerH+3baHBjvFJg7o\
7G9Pvl+frN6uT0+voVot6++H36uHh+9se2iwUwQMOlu1E/X5+71aLevZyj1g0JOAQWenF4/r80/i\
WA7PaxgPtzMg2FECBp19a/8P9dVTd2tZL6qq1cnhP/Wd/Zt1Zniw7aHBTnljNnEc9R4e2OTw8Oiz\
p1/99s/19vv/qI9fvFvPx9N1Zu+jurX8uP76wUdHfo+DgwP/Z5nF+fPntz2EY3tjAnb9+vVtD4Fg\
d+7cOfJrb91+UFUPqur9Y3/ejRs36vbt28f+93BUV69e3fYQjs2NzHAEr3Mjcw9uZIb/zzUwACIJ\
GACRBAyASAIGQCQBAyCSgAEQ6Y25Dwy+iIsXL9bly5dn+7yzZ8/O9lmQyn1gAESyhAhAJAEDIJKA\
ARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAk\
AQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyA\
SAIGQCQBAyCSgAEQScAAiCRgAEQSMAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkY\
AJEEDIBIAgZAJAEDIJKAARBJwACIJGAARBIwACIJGACRBAyASAIGQCQBAyCSgAEQScAAiCRgAEQS\
MAAiCRgAkQQMgEgCBkAkAQMgkoABEEnAAIgkYABEEjAAIgkYAJEEDIBIAgZAJAEDIJKAARBJwACI\
JGAARBIwACIJGACRBAyASAIGQKT/AvUZN0qjufplAAAAAElFTkSuQmCC\
"
  frames[41] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAJMklEQVR4nO3dz28cZxnA8Wdmdr0bO2lpRZsUahCi\
QkEVtwrBgT+hUjnTf6MXDpX6Z3ClZ8SlRzihSghxCQKlqAglQm2dNmriOI5/7OwMhwKCZp2E9N0Z\
PevPR/LlnZX9HCx9NTs/3qrv+z4AIJl67AEA4GkIGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKU3GHgA21cm927E42o96MotmOo96OotmOouqmUZVVWOPB+kJGKxB3/fx\
2Qe/i70//SaarXk00y9+6q15vHD1J/H17/1o7BEhPQGDdei7aE8eRL9cRHu0iPbo4D+HnvnG1REH\
g83hGhisQd91sTw9GnsM2GgCBmvQLRdxfPeT1Qdd/oIiBAzWoF+2cXz31kPr9WQrdl74zggTweYR\
MBhQVTcxvXBp7DFgIwgYDKmqo9m6MPYUsBEEDNag7/uV61VdR7O1PfA0sJkEDNagW6y+A7GKKurJ\
dOBpYDMJGKxBe/LgzLMwoAwBgzVoTx5ECBislYDBGhx8/EH0XfvQejPf8R5EKETAYA2Wp8cr1y9d\
fiWquhl4GthMAgYDambb4VUcUIaAQWGPunljMtuO8BUiFCFgUFwffbdcecRDzFCOgEFhfbeM5WL1\
NbCoazdxQCECBoU9MmBAMQIGhXXtItoHB4//IPCVCBgU1p4cxoPP//HQej2dx/yZF0eYCDaTgMFA\
muksti4+P/YYsDEEDAZS1U0009nYY8DGEDAo7cytVCZRT91GD6UIGBS2XJysXK+qKupmMvA0sLkE\
DApbnhxGrDoJ8/gXFCVgUFh7fDj2CHAuCBgUdvjZjVh1ClbVE2dhUJCAQWEnB7dXrj/z8quhYFCO\
gMFAphcujT0CbBQBg4FMZjtjjwAbRcCgoL7vou+7lce+2MwSKEXAoKBu2UbXnq48VlWVrVSgIAGD\
gvrl4syAAWUJGBTUtW10Z7yJI5x9QVECBgWdHn4ex3c+eWh9uvO1mD9rKxUoScCgpG71TRz1ZCsa\
L/KFogQMBlA306inW2OPARtFwKCg1Rup/CtgE3uBQUkCBgV17RlbqdR1VHUz8DSw2QQMCmpPvIke\
hiJgUNDSViowGAGDgo7vfbpyfTL3Il8oTcCgoMNPb6xcv/jSK8MOAueAgMEAvIkeyhMwGMBkdnHs\
EWDjCBgU0vddRL/6SbBm5i0cUJqAQSFdu4hu2a48VlWNrVSgMAGDQrr2NPrlYuwx4NwQMCika0+j\
EzAYjIBBIScHt2PxYP+h9enOczG94DkwKE3AoJDu9HjlbszTC5eimW2PMBFsNgGDNasnW1E307HH\
gI0jYFBAf8bt8xER9WQWVTMZcBo4HwQMCjnrBo66aWylAmsgYFBI+4g30XsGDMoTMCjEXmAwLAGD\
QhZH91auu/4F6yFgUEQfBx//deWRZ19+deBZ4HwQMCik77oVq1VM5t5ED+sgYLBmjb3AYC18OQ9n\
uHPnTly/fv2JPltVEc3RUXz5XsM++vjz9Q8jbtx+7O/Y3d2N3d3dp5gUzicBgzO8//778frrrz/R\
Z7dn0/jlz38a37r87P+sd10XP3vzzbixd/exv+Odd96Jt99++6lmhfNIwKCA7fk0mmYaHx1/N/bb\
y7Hd7Mc3Zx9GHcdjjwYbS8CggJ35PP5+8uM4jNeijzqq6OP26W784OJvxx4NNpaAQQHPvfTDOJi+\
FlV88cqoPqr4bLEbf7j1/ThZrN6lGfhq3IUIBdTN1or3HVbx4Uf3497hySgzwaYTMChgqz6OOr58\
ptVHt7gTi+Wq58OAr0rAoIAXt27G1Z3fx7Q6jog+muo0vj3/S1yu/hhtK2CwDufmGtje3t7YI5DM\
3buPv/X93679bS/qX/0i9ttfx8HyubhQ34/np5/Ezb3Po3vEXmH/7f79+/5PGdyVK1fGHuGpnZuA\
vfvuu2OPQDJP+hBzRMTNW/tx89Z+RHzw1H/v2rVr/k8Z3FtvvTX2CE+t6h+1lSycY++9994TP8hc\
ggeZ4f/jGhgAKQkYACkJGAApCRgAKQkYACkJGAApnZvnwOD/deXKlXjjjTcG+3tXr14d7G/BJvAc\
GAAp+QoRgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQE\
DICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQM\
gJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyA\
lAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICU\
BAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQE\
DICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlP4J+q93TibbI+8AAAAA\
SUVORK5CYII=\
"
  frames[42] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKLklEQVR4nO3dv48cZxnA8Wdmf9yeffbFTkyMk5CI\
iGACFLRQIIU+SkQFZUr6iCL/AP8BfdIjpQtISC4oIqJIBikFEIVwcbBjJz47d+e72x8zFHGBc3PI\
vp3d5dn9fCQ3M2vdI51HX3n2fWeKuq7rAIBkykUPAAAnIWAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQk\
YACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRg\
AKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAA\
pCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQk\
YACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRg\
AKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAA\
pCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKTUXfQAsKzqqoq9W/+MouxE2RtEp7cWZXctOr21KMrOoseD9AQM\
ZmR8sBMf/v63UU1G9wM2iE5/EN3BRnz7pdeiu3Z60SNCagIGMzK8dzeqySgmw/2YDPdjdP942e1H\
PRkvdDZYBr4Dgxm59/lWVOPhkeNf3T4s5j8QLBkBgxkZ7m5HXU2OHD/71OXo9AcLmAiWi4DBDNR1\
HXVdNZ7rb5yPonT3HqYlYDADdTWJ8f5O47lOfz2icAsRpiVgMAN1NY6DL28dc7aIQsBgagIGM1CN\
hrH32UdHjhedXqyfv7SAiWD5CBjMRH3/z4PKbj/WzwkYtEHAYAYmo4PG40VZRu/U2TlPA8tJwGAG\
hrvbjasQiyisQISWCBjMwN1PPmh82kZncNoCDmiJgMEMTIb7jcfPfPO7HuQLLREwaNn/3sR8LqJw\
2UEbXEnQsnoyivHhXuM5T6CH9ggYtGx8eC/2b3/afLII34FBSwQMWjYZHcRhw1M4ik4veoMzC5gI\
lpOAwZx0Bxtx6sKzix4DloaAQcuOW4FYdnvRG2zMeRpYXgIGLRvu3m56ilQUhU3M0CYBg5bt3vgw\
mgpWlF0vYoYWCRi07HDni8bjm9/6YSgYtEfAoEV1XUfUDfcPI2LtzONzngaWm4BBi6rx4bGLOHrr\
ltBDmwQMWjQ+uBfDe3ePOetNzNAmAYMWjfa24/DuZ0eOdwcb0d84v4CJYHkJGMxBd7AR/dPnFj0G\
LBUBgxYd9ybmTm8QnbVTc54GlpuAQYsOd283Hi/K0nvAoGUCBi268/HVRY8AK0PAoEXHvon50uU5\
TwLLT8CgJXVdHbuJ+dTjT895Glh+AgYtmQwPYjI+bDzXWz8752lg+QkYtGR8uBeTYfMqxChKm5ih\
ZQIGLTm489lXr1L5mk5/PTq9tQVMBMtNwKAt1SSiro4cHmw+Gf0Nm5ihbQIGLajrOsbHbWJeOxVl\
bzDniWD5CRi0pOn2YURE2enaxAwzIGDQkt3r/2g8XpRdCzhgBgQMWlEf+ybms0+/OOdZYDUIGLSg\
ruuIaNrEXMTamSfmPQ6sBAGDFkwO96KajI+eKCJ6pzfnPxCsAAGDFhzu3I7qmFWIReEyg1lwZUEL\
9r+4FuOD3SPHy05PwGBGXFkwpeO//4rYePL56J3yHESYBQGDFkxGzQ/x7a6fibLbn/M0sBoEDKZV\
1zHc2248VXb7EW4hwky4smBKdTWOu//6a+O5IgqbmGFGBAymVNd1TJpWIBZlbFx8fv4DwYoQMJjW\
MW9hLsoyBucuzXkYWB0CBlMaHexEXU2OHC+KMvpWIMLMCBhMabR3p/kpHBERnkIPMyNgMKXdGx82\
PoWju37Wa1RghgQMplQ13D6MiDh94bno2AMGMyNgMIW6rqI6ZhNz//RmFGV3zhPB6hAwmEJdTWJ0\
707juU5/PYrSJQaz4uqCKVSjYeze/HjRY8BKEjCYQl1NYrR39H9gRdmNtbMXFjARrA4BgynUdRVN\
T6Ive/04feG5uc8Dq0TAYAqj/S/vv07lQUXR8RoVmDFLpOBrrl27FltbWw/12c3q84i6OnJ8NB7F\
e++9H/VDrEK8fPlynD9//pHnhFUnYPA1b775ZrzxxhsP9dlf/+In8fOfvnjk+KfXb8Yvf/WzOBge\
84SO//L222/Hyy+//MhzwqoTMJjSqOrFvw+/E3uTx2KzezMurn0U737wSYzGzRucgXYIGJxQWRZR\
dk/FX3Zeis9Hz0QdRRTxvbg7/kZc3/5zVFXzU+qBdljEASe01u3EdvfHcWv0raijjIgi6ujE1sGL\
8fftZxrWJgJtEjA4ofNn1+MHzz8dEQ++cbmOMia1mxswawIGJ9Qpy3hscBgRD65CnEyGMTq8u5ih\
YIUIGJzQaDKJ/t6fYnP8fhT1YUTU0SsO4sL4SlzfenfR48HSW5n7HDdu3Fj0CCSxs7PzUJ+7/sVu\
vPab38VTF/4Ym098Py48cSl+9GwZm6e3Y3t3/6F/3vb2tn+fLMzFixcXPcKJrUzA3nrrrUWPQBJX\
r1596M/u7g/jb1u3IrauRFFE/KEsoyiLGI2Pbm4+zpUrV+LmzZsnGRWm9vrrry96hBNbmYBl/iUx\
X6PRKN55551H/nt1HTGaVBGPuP3r1VdftZEZTsB3YACkJGAApCRgAKQkYACkJGAApCRgAKS0Msvo\
4WG98MIL8corr8zt52XeSAqLVNRN70MHgP9zbiECkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKA\
AZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoAB\
kJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQ\
koABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCS\
gAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKA\
AZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoAB\
kJKAAZCSgAGQ0n8AZ3HVSCWhsCMAAAAASUVORK5CYII=\
"
  frames[43] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKbElEQVR4nO3dy45cV7nA8W/vquqb22lfmiY3kxzE\
JYgDRIA8QmGExANkyBjxAnkI4DEyYhAdBUZIvAAIHR0BOeBEnAPtOA5JO4673d3V3VW1GIQBorbB\
rtpVxVf1+w3Xaqu/gbf+Ktfay1UppQQAJFMvegAAmISAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKA\
AZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoAB\
kJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQ\
koABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCS\
gAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKA\
AZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoAB\
kJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQ\
koABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCS\
gAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKAAZCSgAGQkoABkJKA\
AZCSgAGQkoABkJKAAZCSgAGQkoABkFJ30QPAsiqlxIP938TBO7+Mta2d6G1did7WTqxt7cT6lU/H\
+va1RY8IqQkYzEopcXT3nbj/x1+PbX3qS6/Ei698bwFDwfLwT4gwI2U0jKO77zTuVbVHD6blKYIZ\
KWUU50f3xtaruhtX/+PrC5gIlouAwYwM+g+jlNHYelXXsb6zt4CJYLkIGMzIyb13YzQ4b9yr686c\
p4HlI2AwIycH+40Bu7T32ai76wuYCJaLgMEMlDJ65Kevrd3no+6uzXkiWD4CBjNQhoM4O/ygca/T\
24ioqjlPBMtHwGAGhhf9OLzzh7H1qu7E2vb1qAQMpiZgMANlNIwyGo6t172NuPzM5xcwESwfAYMZ\
ODv6KEopY+tV3Yne5lMLmAiWj4DBDDzY/22U4WBsvbO2EeEWDmiFJwlaVkqJQf8oIsY/ge3c+Ip3\
wKAlAgYtK6NhDM5PG/c2nvpUROWxgzZ4kqBlg/7DODnYb9yruz0nEKElAgYtG1704+zww7H17sZ2\
bO1+ZgETwXISMGjZI+8/7G3E+uXdOU8Dy0vAoGUnH+43nd+IutOJztrm/AeCJSVg0LKP938TTQWr\
u+sRvv6C1ggYtKiUEtHwAnNExLXP3QwFg/YIGLRoeNGPwdlx49765etzngaWm4BBi84+/kscf/D/\
jXtV1XGEHlokYNCiR13iu7Z9PTavPbuAiWB5CRi0aNB/2Lje3dyOtUtX5zwNLDcBgxYd37vduF7X\
nag63TlPA8tNwKBFh3d+37i+ef3GnCeB5Sdg0JLRaBhlOP79V0TEU89+cc7TwPITMGjJ4PQohucn\
jXu9rR0nEKFlAgYtOb3/XuMlvnV3Peru2gImguUmYNCCUkqMLs4aj9BvXnvWJb4wAwIGLTk/vt+4\
3t245BJfmAEBg5Y8uP2/jet1Zy2q2qMGbfNUQStK4/dfERE7N74851lgNQgYtGD4iO+/IqrYuOoK\
KZgFAYMW9O/fbb6Fvororm/NfyBYAQIGLTj96L0YNgSs09uIqu4sYCJYfgIGUyqlxGh40bh3+Zkv\
xNqlK3OeCFaDgMG0yij6D/7SuNXdvBxVpzfngWA1CBhMaTQaxsd/+p/GvaqqXSEFMyJgMK1SGk8g\
VnUnrrzw1QUMBKtBwGBKF6eHzQGr6ti4+swCJoLVIGAwpQf7v4vBWcMt9FXlEl+YIQGDKZRSPnn/\
q4zG9rau34iOgMHMCBhMo5QYXfQbt7Z2P+MTGMyQgMEUymgQ/QcfNO5117ciKo8YzIqnC6YwvDiL\
o7tvj61XVR29S1ccoYcZEjCYQhkOogwHY+t1byO2n/78AiaC1SFgMIWzo4MopYytV52OK6RgxgQM\
pnB45/fNn8C6ay7xhRkTMJhQKSUuTg4jYvwT2M7zX466053/ULBCBAwmVIaDGPQfNu6t7+w5gQgz\
5gmDCQ3OT+L0ozuNe53euhOIMGMCBhManZ9G/3D8HbCq043uxvYCJoLVImAwocF58w0cvc2n4rIj\
9DBzAgYTOrl3u+n8RtSdXnQ3Ls1/IFgxAgYTOrz9u2gqWNXpRoTvv2DWBAwmUMooymj8BvqIiKuf\
/UaEAxwwcwIGExie9+PiEUfoN3b25jwNrCYBgwmcHX4YJwd/btyrOz1H6GEOXBUAf3Pr1q24d+/e\
Y/1sfXoQ9XAw9k3XoFqP375zO7p3T//pn6+qKl5++eXY3NyccFqgKk03kcIKevXVV+ONN954rJ/9\
9tdeiB/94Dtjn7T+sH8Q3//xz6J/MX4/4t+r6zreeuuteOmllyaeF1adT2AwgVe+9kKcjS7Fu2df\
iPPRZuyuvRu7vXfjfDD8l/EC2iFgMIFr156PXx9+N46G1yKiitv9L8UXL/0q/vz+rUWPBivDIQ54\
Qt1OHW/3vxVHw9345BGqYhTduHV8M37x1sWix4OVIWDwhJ7bvRzXdsb/s8pRdOPuvX9+eANoj4DB\
E7qxtxMv7kb84y0cw4vjGAxOFjITrCIBgyd0cnYRV/o/j83B21GVi4gYxXp1HJcf/lc8/Lj53TCg\
fStziOP9999f9Aj8m+v3m2+X/0f//fbd+P4PfxLP7f0itq//Zzy3txfffHEYx8fvxdHJ2WP/voOD\
A38vWbinn3560SNMbGUC9vrrry96BP7N7e/vP/bP3n/Yj/sP70T8352oqyp+2vnkfbDh6PFeqyyl\
xJtvvhl7e66dYrFee+21RY8wMS8yw988yYvM0/IiM0zPd2AApCRgAKQkYACkJGAApCRgAKQkYACk\
tDLvgcG/cvPmzRgOh3P5XXVdx/b29lx+Fywr74EBkJJ/QgQgJQEDICUBAyAlAQMgJQEDICUBAyAl\
AQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUB\
AyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQED\
ICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMg\
JQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAl\
AQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUBAyAlAQMgJQEDICUB\
AyAlAQMgJQEDICUBAyClvwIMOfZwERUGTAAAAABJRU5ErkJggg==\
"
  frames[44] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0\
dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAKl0lEQVR4nO3dy29cZxnA4ffM1R43jkOuJXVaoEXh\
UgmEBBtWiGUXlfg32HaJ2n+BDXtUiTVCKoIFQqpYUEBCLUgJBIU2UNM2SR07vs7tsOBW8ImI7TMz\
fWeeZ+fvs+x34dFPnvOdM0VZlmUAQDKNWQ8AACchYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQk\
YACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRg\
AKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAA\
pCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQk\
YACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRg\
AKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAA\
pCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACk\
JGAApCRgAKQkYACkJGAApCRgAKTUmvUAMK/KsowH77wVW3d+F70L16J3YT3ay6vR6vai0V6Koihm\
PSKkJmAwKeU4tv7y+7h74/V/fF00orNyLrqrF+L8c1+Li9e/Ptv5IDkBgwkZj4bx4O03/7NQjqO/\
cz/6O/dj+dyTsxsM5oRrYDAh5WgYZTk6sl40WnF2/fkZTATzRcBgQg4f3otyNDyyXjQa/gODGggY\
TMjDjT/EqH9Quddotac8DcwfAYMJKMtxDA52IqI8snfmyc9Go700/aFgzggYTMB4NIyDzb9V7vUu\
rEej1ZnyRDB/BAwmYNw/iO2Nm5V7jVbHPWBQAwGDCRgNDyPKo28fNjvLTiBCTQQMJuDh325FOa44\
Qt9sx9LZizOYCOaPgMEE7L5/uzJgraUnomg0ZzARzB8Bg5qV41GMBoeVe2vXno+i4QE4UAcBg5qN\
+vtxuP1B5V73zPkIBzigFgIGNRsc7MTu3Xcqdooomm0nEKEmAgY1Gx7sRtUNzO2VtVi9en36A8Gc\
EjCo2cONP1QfoW91orOyNoOJYD4JGNRs9+7bleut3qrrX1AjAYMajYf9GA+qH+D7iU9/JSIEDOoi\
YFCjg+27sXvvTuVeZ+XclKeB+SZgUKNRfz9Gh3tH1pvdlWivrDmBCDUSMKjRYG+rcr2zci6Wzl6e\
8jQw3wQMavRw44+V6412J5qd5SlPA/NNwKAmZVnG3v2/VO51eo7PQ90EDGoy6u/H6FEnEJ/96pSn\
gfknYFCT/Q//Godbdyv32surDnBAzQQMalCWZQwP92I8PPoU+vbKWrSWn5jBVDDfBAxqcrh9r3J9\
ae1KdJ84P+VpYP4JGNSijPt/eqNyp9nqRNH0GWBQNwGDOpRljB/xIZZnrn7O9S+YAAGDGgz2tmI8\
7FfsFLFy8ZlpjwMLQcCgBrt3347B/vbRjSKi1e1NfyBYAAIGp1SWZQz2H0Y5Gh7ZWzp72RM4YEIE\
DE6rLONg64PKrZWLT0d7+cyUB4LFIGBwSuV4GA/+/NvKvUarE0WjOeWJYDEIGJzSeDSsPMBRNJqx\
+tTnZzARLAYBg1Pa/3CjOmBFI3oXrs1gIlgMAgantL1xs/IhvkWrE81WdwYTwWIQMDiFsiyjHA0q\
91avXo9mZ2nKE8HiEDA4hXI0jINHPIG+u3oximZ7yhPB4hAwOIXR4CC2371RuddotjxCCiZIwOAU\
xoPDyrcQG+0lJxBhwgQMTmH33jtRjkdH1hvNdiyf++QMJoLFIWBwCtvv3qwMWLPddQMzTJiAwQmV\
49EjnkAfsfbMl6LhM8BgogQMTmhwsBO779+u3OuuXowovLxgkrzC4ITG/f042D76EN+i2Yp2b9UJ\
RJgwAYMTGvb3K9dbS2di5eKnpjwNLB4BgxPaff92RFkeWW+02tHurc5gIlgsAgYnUJZlbG/8sXKv\
2V6OwvUvmDivMjiBcjSM8fCwcu/8c1+NcP0LJk7A4AT6Ox/G/ocblXvdM+enPA0sJgGDExge7sRg\
78GR9Ua7G83uE04gwhQIGJzAYG+7cr175kL0zl+d8jSwmAQMTuD+rTcq1xutbjQ7vSlPA4tJwOCY\
yrKMwX71f2Arl56Z7jCwwAQMjmnU34/xoPoE4pkrz055GlhcAgbHdLD1fhzu3K/ca/fOOsABUyJg\
cAz/evtwdLh3ZK/ZXYlmZ3kGU8FiEjA4psHOZuV67/xT0T17acrTwOLygUUsvM3Nzbhx48ZjfW8R\
Ed07r1fubT3cizd+9Zv/+zPW19djfX39OCMCFYqyrHgaKSyQ1157LV544YXH+t5Go4gffOdb8ekn\
z/3XelmW8b0f/jq+/9M3/+/PePnll+OVV145yajAR/gPDI7h0lovljrd2Dj4TDwYXo5ecyuudm9F\
Mw7jt7fem/V4sFAEDI7hy599Kh50vxF3dr4QZTSiiDLu9tfjiys/i+296qP1wGQIGBzD/dGzcefg\
i1H+8/xTGUXcG6zHazevxebD6g+4BCbDKUR4TEURcfkTa/+O10d24p0PDmJnvz+TuWBRCRg8pnaz\
Gd/80uVoxPB/dsqI0YMYjZ2HgmkSMDiG3uhWXBr9PGK8FxFlNIt+XG3/Lrb/+pNZjwYLZ2Gugb33\
nhNiVNvcrL4x+X/1h6P49nd/HFcv/iLOX/pRrJ17Op6/1o2nn9qLN//07mP/vp2dHX+PfGxcuXJl\
1iOc2MIE7NVXX531CHxM3bx587G/93Awitsbm3F745dRFL+Mnzca0WwUcTgYPfbPeOutt/w98rHx\
0ksvzXqEE3MjMwvvODcy18GNzFAP18AASEnAAEhJwABIScAASEnAAEhJwABIaWHuA4NHuXLlSrz4\
4otT+33Xr1+f2u+CeeY+MABS8hYiACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
KQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAAp\
CRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJ\
GAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkY\
ACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgAKQkYACkJGAApCRgA\
Kf0d5wf8q3x1+JcAAAAASUVORK5CYII=\
"


    /* set a timeout to make sure all the above elements are created before
       the object is initialized. */
    setTimeout(function() {
        anime1d6ab7d88ab4343aa55e9874d6f8be1 = new Animation(frames, img_id, slider_id, 40.0,
                                 loop_select_id);
    }, 0);
  })()
</script>
<br>

We can see from the animation that over the course of one episode the pole increasingly tilts more to the left and right. Eventually it would fall out of frame. 

Let’s try using a neural network policy in the next segment (which will prob be in a few days)! Byeee. 





