"""Bind a cached crop response to the exact image, query, crop and plan."""
FIELDS=('topic','frame','tile_index','bounds','source_image_sha256','crop_png_sha256','query','plan_sha256')

def check_crop_receipt(receipt,expected):
    if set(expected)!=set(FIELDS) or any(receipt.get(k)!=expected[k] for k in FIELDS):
        raise ValueError('Crop response does not match the frozen input and query')
    result=receipt.get('response')
    if not isinstance(result,dict) or not isinstance(result.get('results'),list) or result.get('status','ok') not in ('ok','success'):
        raise ValueError('Crop response is not a successful segmentation result')
