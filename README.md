
# Lunar Lander - Gymnasium - DQN

The goal of this project is training an agent to play LunarLander game. We have used [Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/) project to reporesent the game environment to the agent. In addition, we have implemented the agent by Deep Q Learning model [details](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc) and [PyTorch](https://pytorch.org/). 

-----

## How to run
Since this project includes some not common libraries, first install all packages by running this command: 
```
pip install -r requirements.txt
```
You can see the packages inside ```requirements.text``` file. 

-----
The project includes two files: ```agent.py``` and ```train and visualize agent.ipynb```.
The DQN agant is located inside the agent.py.
To run the project, open ```train and visualize agent.ipynb``` file and run code cells one by one. 
Generally the code includes 4 sections: 
1- Initialize environment and agent
2- Train DQN agent
3- Plot Training reports
4- Create video from trained agent playing

you can watch the result of the trained agent on the last step. A new window will be opened to show the game. Since Jupyter Notebook is not prefectly compatible with that window, when the game will over, it is not going to close the window by it self. Make sure you will close that window and reset the Kernel. 
To prevent reset Kernel step, you can just export the ```ipynb``` code as a ```py``` code then run the code. 
