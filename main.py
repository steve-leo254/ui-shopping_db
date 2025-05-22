from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from pydantic_model import (
    ProductsBase, CartPayload, CartItem, UpdateProduct, CategoryBase, CategoryResponse,
    ProductResponse, OrderResponse, OrderDetailResponse, Role, PaginatedProductResponse,
    ImageResponse, AddressCreate, AddressResponse,OrderCreate
)
from typing import Annotated, List
import models
from database import engine, db_dependency
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import select, func, update
from sqlalchemy.orm import joinedload
import auth
from auth import get_active_user
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
from dotenv import load_dotenv
import os
from decimal import Decimal
from math import ceil
import uuid
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import random

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(auth.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

user_dependency = Annotated[dict, Depends(get_active_user)]

def require_admin(user: user_dependency):
    if user.get("role") != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    logger.info("Database tables created successfully")

@app.on_event("startup")
async def on_startup():
    await init_db()

@app.post("/upload-image", response_model=ImageResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(user: user_dependency, file: UploadFile = File(...)):
    require_admin(user)
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are allowed")
        max_size = 5 * 1024 * 1024
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(status_code=400, detail="File size exceeds 5MB limit")
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["jpg", "jpeg", "png", "gif"]:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        with file_path.open("wb") as f:
            f.write(content)
        img_url = f"/uploads/{unique_filename}"
        logger.info(f"Image uploaded: {unique_filename} by user {user.get('id')}")
        return {"message": "Image uploaded successfully", "img_url": img_url}
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail="Error uploading image")

@app.get("/public/products", response_model=PaginatedProductResponse, status_code=status.HTTP_200_OK)
async def browse_products(db: db_dependency, search: str = None, page: int = 1, limit: int = 10):
    try:
        skip = (page - 1) * limit
        query = select(models.Products).options(joinedload(models.Products.category))
        if search:
            query = query.filter(models.Products.name.ilike(f"%{search}%"))
            logger.info(f"Product search query: {search}")
        
        count_query = select(func.count()).select_from(models.Products)
        if search:
            count_query = count_query.filter(models.Products.name.ilike(f"%{search}%"))
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        products = result.scalars().all()
        
        total_pages = ceil(total / limit)
        return {
            "items": products,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": total_pages
        }
    except SQLAlchemyError as e:
        logger.error(f"Error fetching products: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching products")

@app.get("/public/products/{product_id}", response_model=ProductResponse, status_code=status.HTTP_200_OK)
async def get_product_by_id(product_id: int, db: db_dependency):
    try:
        result = await db.execute(
            select(models.Products)
            .filter(models.Products.id == product_id)
            .options(joinedload(models.Products.category))
        )
        product = result.scalars().first()
        if not product:
            logger.info(f"Product not found: ID {product_id}")
            raise HTTPException(status_code=404, detail="Product not found")
        return product
    except SQLAlchemyError as e:
        logger.error(f"Error fetching product by ID {product_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching product")

@app.get("/public/categories", response_model=List[CategoryResponse], status_code=status.HTTP_200_OK)
async def browse_categories(db: db_dependency):
    try:
        result = await db.execute(select(models.Categories))
        categories = result.scalars().all()
        return categories
    except SQLAlchemyError as e:
        logger.error(f"Error fetching categories: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching categories")

@app.post("/categories", response_model=CategoryResponse, status_code=status.HTTP_201_CREATED)
async def create_category(user: user_dependency, db: db_dependency, category: CategoryBase):
    require_admin(user)
    try:
        db_category = models.Categories(**category.dict())
        db.add(db_category)
        await db.commit()
        await db.refresh(db_category)
        return db_category
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error creating category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/products", status_code=status.HTTP_201_CREATED)
async def add_product(user: user_dependency, db: db_dependency, create_product: ProductsBase):
    require_admin(user)
    try:
        add_product = models.Products(
            **create_product.dict(),
            user_id=user.get("id")
        )
        db.add(add_product)
        await db.commit()
        await db.refresh(add_product)
        return {"message": "Product added successfully"}
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error adding product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products", response_model=PaginatedProductResponse, status_code=status.HTTP_200_OK)
async def fetch_products(user: user_dependency, db: db_dependency, search: str = None, page: int = 1, limit: int = 10):
    require_admin(user)
    try:
        skip = (page - 1) * limit
        query = select(models.Products).filter(models.Products.user_id == user.get("id")).options(joinedload(models.Products.category))
        if search:
            query = query.filter(models.Products.name.ilike(f"%{search}%"))
            logger.info(f"Admin product search query: {search}")
        count_query = select(func.count()).select_from(models.Products).filter(models.Products.user_id == user.get("id"))
        if search:
            count_query = count_query.filter(models.Products.name.ilike(f"%{search}%"))
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        products = result.scalars().all()
        total_pages = ceil(total / limit)
        return {
            "items": products,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": total_pages
        }
    except SQLAlchemyError as e:
        logger.error(f"Error fetching products: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching products")

@app.put("/update-product/{product_id}", status_code=status.HTTP_200_OK)
async def update_product(product_id: int, updated_data: UpdateProduct, user: user_dependency, db: db_dependency):
    require_admin(user)
    try:
        result = await db.execute(
            select(models.Products)
            .filter(models.Products.id == product_id, models.Products.user_id == user.get("id"))
            .options(joinedload(models.Products.category))
        )
        product = result.scalars().first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        update_dict = updated_data.dict(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(product, key, value)
        await db.commit()
        await db.refresh(product)
        return {"message": "Product updated successfully"}
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error updating product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-product/{product_id}", status_code=status.HTTP_200_OK)
async def delete_product(product_id: int, db: db_dependency, user: user_dependency):
    require_admin(user)
    try:
        result = await db.execute(
            select(models.Products)
            .filter(models.Products.id == product_id, models.Products.user_id == user.get("id"))
        )
        product = result.scalars().first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        result = await db.execute(select(models.OrderDetails).filter(models.OrderDetails.product_id == product_id))
        order_details = result.scalars().first()
        if order_details:
            raise HTTPException(status_code=400, detail="Cannot delete product with existing orders")
        await db.delete(product)
        await db.commit()
        return {"message": "Product deleted successfully"}
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error deleting product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders", response_model=List[OrderResponse], status_code=status.HTTP_200_OK)
async def fetch_orders(user: user_dependency, db: db_dependency, skip: int = 0, limit: int = 10):
    try:
        result = await db.execute(
            select(models.Orders)
            .filter(models.Orders.user_id == user.get("id"))
            .options(joinedload(models.Orders.order_details).joinedload(models.OrderDetails.product))
        )
        orders = result.scalars().all()
        for order in orders:
            for detail in order.order_details:
                if detail.product is None:
                    logger.warning(f"Order {order.order_id} has order_detail {detail.order_detail_id} with missing product (product_id={detail.product_id})")
        return orders
    except SQLAlchemyError as e:
        logger.error(f"Error fetching orders: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching orders")

@app.get("/dashboard", status_code=status.HTTP_200_OK)
async def dashboard(user: user_dependency, db: db_dependency):
    require_admin(user)
    try:
        id = user.get("id")
        today = datetime.utcnow().date()
        
        result = await db.execute(
            select(func.sum(models.Orders.total)).filter(models.Orders.user_id == id)
        )
        total_sales = result.scalar() or 0
        result = await db.execute(
            select(func.count(models.Products.id)).filter(models.Products.user_id == id)
        )
        total_products = result.scalar() or 0
        result = await db.execute(
            select(func.sum(models.Orders.total)).filter(
                models.Orders.user_id == id, func.date(models.Orders.datetime) == today
            )
        )
        today_sale = result.scalar() or 0
        
        return {
            "total_sales": float(total_sales),
            "total_products": total_products,
            "today_sale": float(today_sale),
        }
    except SQLAlchemyError as e:
        logger.error(f"Error fetching dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching dashboard data")

@app.post("/addresses", response_model=AddressResponse, status_code=status.HTTP_201_CREATED)
async def create_address(user: user_dependency, db: db_dependency, address: AddressCreate):
    try:
        user_id = user.get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user authentication")

        # Unset other default addresses if this is default
        if getattr(address, "is_default", False):
            stmt = update(models.Address).where(
                models.Address.user_id == user_id,
                models.Address.is_default == True
            ).values(is_default=False)
            await db.execute(stmt)
            await db.commit()

        # Create new address
        db_address = models.Address(
            **address.dict(),
            user_id=user_id,
            created_at=datetime.utcnow()
        )
        db.add(db_address)
        await db.commit()
        await db.refresh(db_address)

        logger.info(f"Address created for user {user_id}: Address ID {db_address.id}")
        return db_address
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"Integrity error creating address: {str(e)}")
        raise HTTPException(status_code=400, detail="Address already exists or invalid data")
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Database error creating address: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/addresses", response_model=List[AddressResponse], status_code=status.HTTP_200_OK)
async def get_addresses(user: user_dependency, db: db_dependency):
    try:
        result = await db.execute(
            select(models.Address).filter(models.Address.user_id == user.get("id"))
        )
        addresses = result.scalars().all()
        if not addresses:
            logger.info(f"No addresses found for user {user.get('id')}")
            return []
        logger.info(f"Retrieved {len(addresses)} addresses for user {user.get('id')}")
        return addresses
    except SQLAlchemyError as e:
        logger.error(f"Error fetching addresses: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching addresses")

@app.delete("/addresses/{address_id}", status_code=status.HTTP_200_OK)
async def delete_address(address_id: int, user: user_dependency, db: db_dependency):
    try:
        result = await db.execute(
            select(models.Address).filter(
                models.Address.id == address_id,
                models.Address.user_id == user.get("id")
            ).options(joinedload(models.Address.orders))
        )
        address = result.scalars().first()
        if not address:
            logger.info(f"Address not found: ID {address_id} for user {user.get('id')}")
            raise HTTPException(status_code=404, detail="Address not found")
        
        if address.orders:
            logger.info(f"Cannot delete address {address_id}: used in {len(address.orders)} orders")
            raise HTTPException(status_code=400, detail="Cannot delete address used in orders")
        
        await db.delete(address)
        await db.commit()
        logger.info(f"Address {address_id} deleted by user {user.get('id')}")
        return {"message": "Address deleted successfully"}
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error deleting address {address_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting address")

def generate_order_number():
    timestamp = int(datetime.utcnow().timestamp())
    random_digits = random.randint(1000, 9999)
    return f"#{timestamp}{random_digits}"


@app.post("/orders/", status_code=status.HTTP_201_CREATED)
async def create_order(order_data: OrderCreate, db: db_dependency, user: user_dependency):
    try:
        # Validate delivery address
        delivery_address_result = await db.execute(
            select(models.Address).filter(
                models.Address.id == order_data.delivery_address_id,
                models.Address.user_id == user.get("id")
            )
        )
        delivery_address = delivery_address_result.scalars().first()
        if not delivery_address:
            raise HTTPException(status_code=404, detail="Delivery address not found")

        # Validate billing address
        billing_address_result = await db.execute(
            select(models.Address).filter(
                models.Address.id == order_data.billing_address_id,
                models.Address.user_id == user.get("id")
            )
        )
        billing_address = billing_address_result.scalars().first()
        if not billing_address:
            raise HTTPException(status_code=404, detail="Billing address not found")

        # Validate payment method
        allowed_payment_methods = ["credit_card", "payment_on_delivery", "mpesa"]
        if order_data.payment_method not in allowed_payment_methods:
            raise HTTPException(status_code=400, detail="Invalid payment method")

        # Start transaction
        total_cost = Decimal(str(order_data.total))
        db_order = models.Orders(
            order_number=generate_order_number(),
            user_id=user.get("id"),
            delivery_address_id=order_data.delivery_address_id,
            billing_address_id=order_data.billing_address_id,
            payment_method=order_data.payment_method,
            subtotal=Decimal(str(order_data.subtotal)),
            shipping_fee=Decimal(str(order_data.shipping_fee)),
            total=total_cost,
            status="pending",
            datetime=datetime.utcnow()
        )
        db.add(db_order)
        await db.flush()

        # Create and validate OrderDetails
        for item in order_data.items:
            product_result = await db.execute(
                select(models.Products).filter(models.Products.id == item.product_id)
            )
            product = product_result.scalars().first()
            if not product:
                await db.rollback()
                raise HTTPException(status_code=404, detail=f"Product ID {item.product_id} not found")
            quantity = Decimal(str(item.quantity))
            if product.stock_quantity < quantity:
                await db.rollback()
                raise HTTPException(status_code=400, detail=f"Insufficient stock for product {product.name}")

            db_order_detail = models.OrderDetails(
                order_id=db_order.order_id,
                product_id=item.product_id,
                quantity=quantity,
                unit_price=Decimal(str(item.unit_price)),
                total_price=Decimal(str(item.unit_price)) * quantity
            )
            product.stock_quantity -= quantity
            db.add(db_order_detail)

        await db.commit()
        await db.refresh(db_order)
        logger.info(f"Order {db_order.order_id} created for user {user.get('id')}")
        return {"order_id": db_order.order_id}
    except ValueError as e:
        await db.rollback()
        logger.error(f"Invalid data value: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid data value")
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Database error creating order: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        await db.rollback()
        logger.error(f"Unexpected error creating order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/orders/{order_id}", response_model=OrderResponse, status_code=status.HTTP_200_OK)
async def get_order(order_id: int, db: db_dependency, user: user_dependency):
    try:
        result = await db.execute(
            select(models.Orders)
            .filter(models.Orders.order_id == order_id, models.Orders.user_id == user.get("id"))
            .options(
                joinedload(models.Orders.order_details).joinedload(models.OrderDetails.product),
                joinedload(models.Orders.delivery_address),
                joinedload(models.Orders.billing_address)
            )
        )
        order = result.scalars().first()
        if not order:
            logger.info(f"Order not found: ID {order_id} for user {user.get('id')}")
            raise HTTPException(status_code=404, detail="Order not found")
        return order
    except SQLAlchemyError as e:
        logger.error(f"Error fetching order {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching order")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)