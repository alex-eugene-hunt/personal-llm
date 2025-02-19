
docker build -t personal-llm .
docker run --gpus all -it --rm personal-llm


./fine_tuned_model is where the AlexAI is stored.


1. python train.py
2. python run_chat.py

