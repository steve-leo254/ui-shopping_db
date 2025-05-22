from sqlalchemy import Column, Integer, String, func, DateTime, Numeric, ForeignKey, Enum, Boolean
from database import Base
from sqlalchemy.orm import relationship
import enum

class Role(enum.Enum):
    ADMIN = "admin"
    CUSTOMER = "customer"

class OrderStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class Users(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(200), unique=True, index=True)
    hashed_password = Column(String(256))
    role = Column(Enum(Role), default=Role.CUSTOMER, nullable=False)
    created_at = Column(DateTime, default=func.now())
    orders = relationship("Orders", back_populates="user")
    products = relationship("Products", back_populates="user")
    addresses = relationship("Address", back_populates="user")

class Categories(Base):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(200))
    products = relationship("Products", back_populates="category")

class Products(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    cost = Column(Numeric(precision=14, scale=2), nullable=False)
    price = Column(Numeric(precision=14, scale=2), nullable=False)
    img_url = Column(String(200), nullable=True)
    stock_quantity = Column(Numeric(precision=14, scale=2), nullable=False)
    description = Column(String(200), nullable=True) 
    created_at = Column(DateTime, default=func.now())
    barcode = Column(Numeric(precision=12), unique=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    category_id = Column(Integer, ForeignKey('categories.id'))
    brand = Column(String(100), nullable=True)
    user = relationship("Users", back_populates="products")
    category = relationship("Categories", back_populates="products")
    order_details = relationship("OrderDetails", back_populates="product")



class Address(Base):
    __tablename__ = 'addresses'
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(20), nullable=False)
    street = Column(String(200), nullable=False)
    city = Column(String(100), nullable=False)
    postal_code = Column(String(20), nullable=False)
    country = Column(String(100), nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=func.now())
    
    user = relationship("Users", back_populates="addresses")
    orders_delivery = relationship("Orders", foreign_keys="Orders.delivery_address_id", back_populates="delivery_address") 
    orders_billing = relationship("Orders", foreign_keys="Orders.billing_address_id", back_populates="billing_address")  


class Orders(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    order_number = Column(String(50), unique=True, nullable=False)  
    subtotal = Column(Numeric(precision=14, scale=2), nullable=False)  
    shipping_fee = Column(Numeric(precision=14, scale=2), nullable=False)  
    total = Column(Numeric(precision=14, scale=2), nullable=False)  
    datetime = Column(DateTime, default=func.now(), index=True)
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING, nullable=False)
    payment_method = Column(String(100), nullable=False)  
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    delivery_address_id = Column(Integer, ForeignKey('addresses.id'), nullable=False)  
    billing_address_id = Column(Integer, ForeignKey('addresses.id'), nullable=False)  

    user = relationship("Users", back_populates="orders")
    order_details = relationship("OrderDetails", back_populates="order")
    delivery_address = relationship("Address", foreign_keys=[delivery_address_id], back_populates="orders_delivery")  
    billing_address = relationship("Address", foreign_keys=[billing_address_id], back_populates="orders_billing")

class OrderDetails(Base):
    __tablename__ = "order_details"
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Numeric(precision=15, scale=2), nullable=False)
    unit_price = Column(Numeric(precision=14, scale=2), nullable=False) 
    total_price = Column(Numeric(precision=15, scale=2), nullable=False) 
    
    product = relationship("Products", back_populates="order_details")
    order = relationship("Orders", back_populates="order_details")
