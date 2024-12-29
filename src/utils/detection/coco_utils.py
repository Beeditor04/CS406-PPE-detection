from pycocotools.coco import COCO

def get_coco_api_from_dataset(dataset):
    coco = COCO()
    coco.dataset = dataset.coco
    coco.createIndex()
    return coco