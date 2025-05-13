from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import List, Optional
import re

class CreateUserRequest(BaseModel):
    username: str = Field(..., max_length=80, min_length=3)
    email: EmailStr
    password: str = Field(..., min_length=8)

    @field_validator('password')
    def validate_password(cls, v):
        if not re.search(r'[A-Za-z]', v):
            raise ValueError('Password must contain at least one letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.match(r'^[A-Za-z\d@#$%^&+=]{8,}$', v):
            raise ValueError('Password contains invalid characters')
        return v

class LoginUserRequest(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ProductsBase(BaseModel):
    name: str = Field(..., max_length=100)
    brand: str = Field(..., max_length=50)
    cost: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    image_url: str = Field(..., max_length=255)
    stock_quantity: int = Field(..., ge=0)
    barcode: str = Field(..., max_length=50)
    description: Optional[str] = Field(None, max_length=65535)
    category_id: int = Field(..., gt=0)

    @field_validator('barcode')
    def validate_barcode(cls, v):
        if not re.match(r'^[A-Za-z0-9\-]{1,50}$', v):
            raise ValueError('Invalid barcode format')
        return v

class CartItem(BaseModel):
    id: int = Field(..., gt=0)
    quantity: float = Field(..., gt=0)

class CartPayload(BaseModel):
    cart: List[CartItem]

class TokenVerifyRequest(BaseModel):
    token: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=8)

    @field_validator('new_password')
    def validate_new_password(cls, v):
        if not re.search(r'[A-Za-z]', v):
            raise ValueError('Password must contain at least one letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.match(r'^[A-Za-z\d@#$%^&+=]{8,}$', v):
            raise ValueError('Password contains invalid characters')
        return v

class TokenVerificationResponse(BaseModel):
    username: str
    tokenverification: str

class UpdateProduct(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    brand: Optional[str] = Field(None, max_length=50)
    price: Optional[float] = Field(None, gt=0)
    cost: Optional[float] = Field(None, gt=0)
    image_url: Optional[str] = Field(None, max_length=255)
    stock_quantity: Optional[int] = Field(None, ge=0)
    barcode: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = Field(None, max_length=65535)
    category_id: Optional[int] = Field(None, gt=0)

    @field_validator('barcode')
    def validate_barcode(cls, v):
        if v and not re.match(r'^[A-Za-z0-9\-]{1,50}$', v):
            raise ValueError('Invalid barcode format')
        return v