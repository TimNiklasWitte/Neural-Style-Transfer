from tbparse import SummaryReader


def load_dataframe(log_dir):

    reader = SummaryReader(log_dir, pivot=True)

    df = reader.tensors

    # preprocess images (content tensor to image)
    image_dict_arr = df['Image'].apply(SummaryReader.tensor_to_image)
    df['Image'] = image_dict_arr.apply(lambda x: x['image'])
    
    return df