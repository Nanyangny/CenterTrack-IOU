import json

data_path = "/home/students/acct1001_05/Dataset/mot17/annotations/val_half.json"

with open(data_path, "r") as content:
  half_train = json.loads(content.read())


print(half_train.keys())
count = len(half_train['images'])
print(f'images counts==>{count}')