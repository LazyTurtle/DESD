
import torch, ultralytics, os

from ultralytics import YOLO

def check_installation():
  print(torch.utils.collect_env.main())
  home = os.getcwd().strip()
  print(home)
  ultralytics.checks()

def check_model(model_file = "yolo26n.pt"):
  model = YOLO(model_file)
  results = model('https://upload.wikimedia.org/wikipedia/commons/0/0a/Dave_Megarry_Dungeon_Gary_Con_2018_1.jpg')
  for result in results:
    print(result)

if __name__ == "__main__":
  check_installation()
  check_model()