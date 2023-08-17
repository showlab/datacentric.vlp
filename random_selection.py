import os
import json
import numpy as np


# metadata/cc3m/train_success_align_269.json
# metadata/cc12m/train_success.json
# metadata/yfcc15m/train_success.json
# metadata/laion40m/train_success.json
with open('metadata/laion40m/laion40m.json','r') as f:
    metadata = json.load(f)


success_data = metadata[:len(metadata)//4]
out_json = "/metadata/laion40m/random_selection_1_of_4.json" # 672689

print("{} samples".format(len(success_data)))
with open(out_json, 'w') as outfile:
    json.dump(success_data, outfile)
