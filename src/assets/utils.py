# src/data/utils.py

def collate_fn(batch):
    """
    DataLoader가 배치(batch)를 구성할 때 호출되는 함수.
    이미지와 타겟을 각각의 튜플로 묶어 반환한다.
    """
    return tuple(zip(*batch))