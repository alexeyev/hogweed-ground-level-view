# coding: utf-8

import json
import os
import shutil
import pandas as pd


def get_df(segment: str = "train"):
    """
        Preparing classification datasets based on `ann_coco_*.json`

        Example:

            id,has_hogweed
            2840ae55c87931b629a06e0d03bd2719,1
            07c0c2d926d68fe893e7586bc9aa7425,1
            93e057192c8c80076f6ff03612e54bad,0
            ...

        1 stands for 'has_hogweed', 0 for 'no_hogweed'

    :param segment: "train" or "test"
    :return: pandas dataframe with ids and labels
    """

    j = json.load(open(f"prepared_data/ann_coco_{segment}.json", "r", encoding="utf-8"))
    file_list = [l.strip() for l in open(f"prepared_data/{segment}_file_list.txt", "r") if l.strip()]
    image_has_anno = []

    for anno in j["annotations"]:
        image_has_anno.append(anno["image_id"])

    image_has_anno = set(image_has_anno)
    ids = [el.split("/")[-1].split(".")[0] for el in file_list]
    has_anno = [1 if id in image_has_anno else 0 for id in ids]
    df = pd.DataFrame({"id": ids, "has_hogweed": has_anno})

    return df


def do_if_possible(action):
    try:
        action()
    except Exception as e:
        pass


if __name__ == "__main__":

    from torchvision.datasets import ImageFolder
    from torchvision import transforms

    TEST_AVAILABLE = False

    train_df = get_df("train")
    train_df.to_csv("prepared_data/train.csv", index=None)

    do_if_possible(lambda: os.mkdir("prepared_data/images_train"))
    do_if_possible(lambda: os.mkdir("prepared_data/images_train/has_hogweed"))
    do_if_possible(lambda: os.mkdir("prepared_data/images_train/no_hogweed"))

    train_has_hogweed = set(train_df[train_df["has_hogweed"] == 1]["id"])
    train_no_hogweed = set(train_df[train_df["has_hogweed"] == 0]["id"])

    if TEST_AVAILABLE:
        test_df = get_df("test")
        test_df.to_csv("prepared_data/gold.csv", index=None)
        do_if_possible(lambda: os.mkdir("prepared_data/images_test"))
        do_if_possible(lambda: os.mkdir("prepared_data/images_test/has_hogweed"))
        do_if_possible(lambda: os.mkdir("prepared_data/images_test/no_hogweed"))

        test_has_hogweed = set(test_df[test_df["has_hogweed"] == 1]["id"])
        test_no_hogweed = set(test_df[test_df["has_hogweed"] == 0]["id"])

        test_df["has_hogweed"] = 1
        test_df.to_csv("prepared_data/sample_submission.csv", index=None)

        test_df.drop("has_hogweed", axis=1, inplace=True)
        test_df.to_csv("prepared_data/test.csv", index=None)

    prefix_images = "prepared_data/images"

    for f in os.listdir(prefix_images):

        if not f.endswith(".jpg"):
            continue

        id = f.replace(".jpg", "")

        if id in train_has_hogweed:
            shutil.move(prefix_images + "/" + f, prefix_images + "_train/has_hogweed")
        elif id in train_no_hogweed:
            shutil.move(prefix_images + "/" + f, prefix_images + "_train/no_hogweed")
        elif TEST_AVAILABLE and id in test_has_hogweed:
            shutil.move(prefix_images + "/" + f, prefix_images + "_test/has_hogweed")
        elif TEST_AVAILABLE and id in test_no_hogweed:
            shutil.move(prefix_images + "/" + f, prefix_images + "_test/no_hogweed")
        elif TEST_AVAILABLE:
            raise Exception("Something's wrong.")

    # example how one can read a dataset we have prepared here
    train_dataset = ImageFolder(root="prepared_data/images_train",
                                transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(750)]))
    print(train_dataset.__getitem__(2))
