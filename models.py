from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Numeric
from database import Base
from datetime import datetime

class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Categories(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True)  # Added code
    name = Column(String, index=True)
    description = Column(String, nullable=True)  # Added description

class Products(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    brand = Column(String)
    cost = Column(Numeric(15, 2))
    price = Column(Numeric(15, 2))
    image_url = Column(String)
    stock_quantity = Column(Integer)
    barcode = Column(String, unique=True)
    description = Column(String, nullable=True)
    category_id = Column(Integer, ForeignKey("categories.id"))
    user_id = Column(Integer, ForeignKey("users.id"))

class Orders(Base):
    __tablename__ = "orders"
    order_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    total = Column(Numeric(15, 2))
    datetime = Column(DateTime, default=datetime.utcnow)

class OrderDetails(Base):
    __tablename__ = "order_details"
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.order_id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Numeric(15, 2))
    total_price = Column(Numeric(15, 2))

class Stock(Base):
    __tablename__ = "stock"
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer)
    updated_at = Column(DateTime, default=datetime.utcnow)