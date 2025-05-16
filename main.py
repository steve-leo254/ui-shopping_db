from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from pydantic_model import (
    ProductsBase, CartPayload, CartItem, UpdateProduct, CategoryBase, CategoryResponse,
    ProductResponse, OrderResponse, OrderDetailResponse, Role, PaginatedProductResponse, ImageResponse
)
from typing import Annotated, List
import models
from database import engine, db_dependency  # Use engine from database.py
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from sqlalchemy.exc import SQLAlchemyError
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
import asyncio

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

# Async schema creation
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

# Run init_db on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database schema...")
    await init_db()
    logger.info("Database schema initialized.")

user_dependency = Annotated[dict, Depends(get_active_user)]

def require_admin(user: user_dependency):
    if user.get("role") != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
        query = select(models.Products)
        if search:
            query = query.filter(models.Products.name.ilike(f"%{search}%"))
            logger.info(f"Product search query: {search}")
        total_result = await db.execute(select(func.count()).select_from(query))
        total = total_result.scalar()
        result = await db.execute(query.offset(skip).limit(limit))
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
        query = select(models.Products).filter(models.Products.user_id == user.get("id"))
        if search:
            query = query.filter(models.Products.name.ilike(f"%{search}%"))
            logger.info(f"Admin product search query: {search}")
        total_result = await db.execute(select(func.count()).select_from(query))
        total = total_result.scalar()
        result = await db.execute(query.offset(skip).limit(limit))
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
        product = await db.execute(
            select(models.Products).filter(
                models.Products.id == product_id,
                models.Products.user_id == user.get("id")
            )
        )
        product = product.scalar_one_or_none()
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
        order_details = await db.execute(
            select(models.OrderDetails).filter(models.OrderDetails.product_id == product_id)
        )
        if order_details.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Cannot delete product with existing orders")
        product = await db.execute(
            select(models.Products).filter(
                models.Products.id == product_id,
                models.Products.user_id == user.get("id")
            )
        )
        product = product.scalar_one_or_none()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        await db.execute(
            delete(models.Products).filter(
                models.Products.id == product_id,
                models.Products.user_id == user.get("id")
            )
        )
        await db.commit()
        return {"message": "Product deleted successfully"}
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error deleting product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_order", status_code=status.HTTP_201_CREATED)
async def create_order(db: db_dependency, user: user_dependency, order_payload: CartPayload):
    try:
        new_order = models.Orders(user_id=user.get("id"), total=0)
        db.add(new_order)
        await db.commit()
        await db.refresh(new_order)
        
        total_cost = Decimal('0')
        for item in order_payload.cart:
            product = await db.execute(
                select(models.Products).filter_by(id=item.id)
            )
            product = product.scalar_one_or_none()
            if not product:
                await db.rollback()
                raise HTTPException(status_code=404, detail=f"Product ID {item.id} not found")
            quantity = Decimal(str(item.quantity))
            if product.stock_quantity < quantity:
                await db.rollback()
                raise HTTPException(status_code=400, detail=f"Insufficient stock for product {product.name}")
            
            order_detail = models.OrderDetails(
                order_id=new_order.order_id,
                product_id=product.id,
                quantity=quantity,
                total_price=product.price * quantity,
            )
            total_cost += order_detail.total_price
            product.stock_quantity -= quantity
            db.add(order_detail)
        
        new_order.total = total_cost
        await db.commit()
        
        logger.info(f"Order {new_order.order_id} created for user {user.get('id')}")
        return {
            "message": "Order created successfully",
            "order_id": new_order.order_id,
        }
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error creating order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        await db.rollback()
        logger.error(f"Invalid quantity value: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid quantity value")

@app.get("/orders", response_model=List[OrderResponse], status_code=status.HTTP_200_OK)
async def fetch_orders(user: user_dependency, db: db_dependency, skip: int = 0, limit: int = 10):
    try:
        result = await db.execute(
            select(models.Orders)
            .filter(models.Orders.user_id == user.get("id"))
            .options(
                joinedload(models.Orders.order_details).joinedload(models.OrderDetails.product)
            )
            .offset(skip)
            .limit(limit)
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
        
        total_sales_result = await db.execute(
            select(func.sum(models.Orders.total)).filter(models.Orders.user_id == id)
        )
        total_sales = total_sales_result.scalar() or 0
        
        total_products_result = await db.execute(
            select(func.count(models.Products.id)).filter(models.Products.user_id == id)
        )
        total_products = total_products_result.scalar() or 0
        
        today_sale_result = await db.execute(
            select(func.sum(models.Orders.total)).filter(
                models.Orders.user_id == id,
                func.date(models.Orders.datetime) == today
            )
        )
        today_sale = today_sale_result.scalar() or 0
        
        return {
            "total_sales": float(total_sales),
            "total_products": total_products,
            "today_sale": float(today_sale),
        }
    except SQLAlchemyError as e:
        logger.error(f"Error fetching dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching dashboard data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)