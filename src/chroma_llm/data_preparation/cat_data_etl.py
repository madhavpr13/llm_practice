import os
from pathlib import Path
import csv
from datetime import datetime
from typing import Union, Dict, Generator
from dateutil.parser import parse
from dateutil import tz
from car_data_model import CarReview
from pydantic import ValidationError

def parse_datestr(datestr: str) -> datetime:
    date_split_list = datestr.strip().split()[1:-2]  ## ignore the timezone for now, assume purely UTC
    return parse(" ".join(date_split_list)).astimezone(tz.tzutc())

def unique_id_generator(start: int = 1) -> Generator[int, None, None]:
    current_id = start
    while True:
        yield current_id
        current_id += 1

def make_car_reviews(file_path: Union[Path, str],  id_gen: Generator[int, None, None]) -> Generator[Dict,None,None]:
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        columns = next(reader)
        columns = columns[1:] + ['id_']
        for row in reader:
            try:
                review_id = next(id_gen)
                row = row[1:] + [review_id]
                data_dict = dict(zip(columns, row))
                data_dict['Review_Date'] = parse_datestr(data_dict['Review_Date'])
                try:
                    yield CarReview.model_validate(data_dict)
                except ValidationError as ex:
                    pass
            except Exception as ex:
                pass

def get_car_reviews(base_dir: Union[Path, str]) -> Generator[Dict,None,None]:
    file_paths = (os.path.join(base_dir, f) for f in os.listdir(base_dir))
    id_gen = unique_id_generator()  # Create a unique ID generator
    for file_path in file_paths:
        print(f'In file_path: {file_path}')
        yield from make_car_reviews(file_path, id_gen)



if __name__ == "__main__":
    data_path = Path("D:/udemy/transformers_for_nlp/data/car_reviews")
    reviews = get_car_reviews(data_path)
    reviews_list = list(reviews)
    print(f'Number of reviews found: {len(reviews_list)}')
