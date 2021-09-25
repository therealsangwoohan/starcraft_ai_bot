# starcraft_ai_bot

Made an AI bot that can beat Starcraft II in "Hard" difficulty. It started as a reinforcement learning project but pivoted to evolutionary learning to make it computationally less expensive.

## Early stage demo

https://user-images.githubusercontent.com/46112193/132957900-58f9bf4a-5012-4930-aa8e-624c46317864.mp4

In the early stage, the bot just creates many workers, but as the game progresses, the bot uses unconventional strategies to win.

## How the code runs

![](https://user-images.githubusercontent.com/46112193/134777935-5d8a691a-e5d9-48f4-b4d6-6ff7ff138b40.png)

The Starcraft API allows the game to run visually in real-time for testing purposes, but it must be running off-screen and faster for training.

## Tensorboard

<img width="1027" alt="stage2-model-train (2)" src="https://user-images.githubusercontent.com/46112193/132957885-eb159c77-d44a-425f-abdd-3d37169b2bf3.png">

Multiple versions of the model were trained on the cloud (Paperspace) for weeks. Even after three days, the best model could only win 40% of games in "Hard" difficulty. The best model can now win 95% of games in "Hard difficulty".
