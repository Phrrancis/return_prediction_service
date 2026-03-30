from pydantic import BaseModel
from typing import List

class Product(BaseModel):
    product_id: str
    price: float
    category: str
    size: str

class PredictRequest(BaseModel):
    user_id: str
    cart: List[Product]