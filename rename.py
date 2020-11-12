import os

DIR_TO_RENAME_CONTENTS_OF = "./data_set/testing/style"

count = 0
for filename in os.listdir(DIR_TO_RENAME_CONTENTS_OF):
    os.rename(
        f"{DIR_TO_RENAME_CONTENTS_OF}/style{count + 85}.jpg",
        f"{DIR_TO_RENAME_CONTENTS_OF}/style{count}.jpg",
    )
    count += 1
