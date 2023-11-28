import os
import shutil
from sklearn.model_selection import train_test_split

org_dataset_path = './licenseNums_archive'
train_directory = './train_dataset'
test_directory = './test_dataset'

# licenseNums_archive内の全てのファイルをリストアップ
all_files = [f for f in os.listdir('licenseNums_archive') if os.path.isfile(os.path.join('licenseNums_archive', f))]

train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

for file in train_files:
    shutil.move(os.path.join(org_dataset_path, file), train_directory)

for file in test_files:
    shutil.move(os.path.join(org_dataset_path, file), test_directory)