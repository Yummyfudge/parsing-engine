


def keep_tables_only(label, score):
    return label == 3 and score >= 0.8

def construct_labelled_block(box, score, label):
    return {
        "label": label,
        "bbox": [box[0], box[1], box[2], box[3]],
        "score": score
    }