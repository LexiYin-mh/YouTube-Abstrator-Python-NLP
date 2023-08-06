import numpy as np

def ids2str(decoder_and_encoder, ids, reserved_tokens):
    """Decode ids, if ids > 1 and ids < reserved_tokens, we output <%d>."""
    if not reserved_tokens:
        return decoder_and_encoder.decode(ids.flatten().tolist())
    # 1000 1001 89 23 1 --> Good morning <89> <23>.
    # 1000 89 1001 23 1 --> Good <89> morning <23>.
    tokens = np.where(ids < reserved_tokens)[0]
    if tokens.size <= 0:
        return decoder_and_encoder.decode(ids.flatten().tolist())
    else:
        # tokens -> [89, 23] [2, 3]
        split_locations = np.union1d(tokens, tokens + 1)
        # [2, 3] [3, 4] -> [2, 3, 4]
        # tokens -> [89, 23] [1, 3]
        # [1, 3] [2, 4] -> [1, 2, 3, 4]
        ids_list = np.split(ids, split_locations)
        text_list = [
            "<%d>" %
            i if len(i) == 1 and i < reserved_tokens and i > 1
            else decoder_and_encoder.decode(i.tolist())
            for i in ids_list
            ]
        return " ".join(text_list)