from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel
from datetime import datetime

class CarReview(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    review_date: datetime | None = Field(default=None, validation_alias='Review_Date',
                                         serialization_alias='reviewDate')
    author_name: str | None = Field(default=None, validation_alias='Author_Name',
                                    serialization_alias='authorName')
    vehicle_title: str = Field(default='Unknown Vehicle', validation_alias='Vehicle_Title',
                               description='The title of the vehicle being reviewed.',
                               serialization_alias='vehicleTitle')
    review_title: str = Field(default='Blank review title', validation_alias='Review_Title',
                              description='The title of the review.',
                              serialization_alias='reviewTitle')
    review: str | None = Field(default='Blank review', validation_alias='Review',
                               serialization_alias='reviewText', description='The review text.')
    rating: float = Field(ge=0, le=5, default=3, validation_alias='Rating', serialization_alias='Rating')
    id_: int = Field(gt=0, validation_alias='id', serialization_alias='id')

