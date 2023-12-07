# CarRacingv2
Project in ECE7606J
# Installation
ˋˋˋ
   conda create -n CarRacing python=3.8
   conda activate CarRacing
    
   pip3 install -r requirements.txt
ˋˋˋ

# Train
ˋˋˋ
   cd CarRacingv2
   python3 -m example.train
ˋˋˋ

# Test
ˋˋˋ
  cd CarRacingv2
  python3 -m example.test --model_path /path/to/best_model.pth
ˋˋˋ
