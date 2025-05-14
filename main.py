from fastapi import FastAPI, HTTPException, Depends, status
from pydantic_model import (
    ProductsBase, CartPayload, CartItem, UpdateProduct, CategoryBase, CategoryResponse,
    ProductResponse, OrderResponse, OrderDetailResponse, Role, PaginatedProductResponse
)
from typing import Annotated, List
import models
from database import engine, db_dependency
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
import auth
from auth import get_active_user
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from datetime import datetime
import logging
from dotenv import load_dotenv
import os
from decimal import Decimal
from math import ceil
import asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(auth.router)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_dependency = Annotated[dict, Depends(get_active_user)]

def require_admin(user: user_dependency):
    if user.get("role") != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

@app.get("/public/products", response_model=PaginatedProductResponse, status_code=status.HTTP_200_OK)
async def browse_products(db: db_dependency, search: str = None, page: int = 1, limit: int = 10):
    try:
        skip = (page - 1) * limit
        async with db as session:
            query = select(models.Products)
            if search:
                query = query.filter(models.Products.name.ilike(f"%{search}%"))
                logger.info(f"Product search query: {search}")
            # Count query for total
            count_query = select(func.count()).select_from(query.subquery())
            count_result = await session.execute(count_query)
            total = count_result.scalar()
            # Paginated query
            query = query.offset(skip).limit(limit)
            result = await session.execute(query)
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
        async with db as session:
            result = await session.execute(select(models.Categories))
            categories = result.scalars().all()
            return categories
    except SQLAlchemyError as e:
        logger.error(f"Error fetching categories: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching categories")

@app.post("/categories", response_model=CategoryResponse, status_code=status.HTTP_201_CREATED)
async def create_category(user: user_dependency, db: db_dependency, category: CategoryBase):
    require_admin(user)
    try:
        async with db as session:
            db_category = models.Categories(**category.dict())
            session.add(db_category)
            await session.commit()
            await session.refresh(db_category)
            return db_category
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"Error creating category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/products", status_code=status.HTTP_201_CREATED)
async def add_product(user: user_dependency, db: db_dependency, create_product: ProductsBase):
    require_admin(user)
    try:
        async with db as session:
            add_product = models.Products(
                **create_product.dict(),
                user_id=user.get("id")
            )
            session.add(add_product)
            await session.commit()
            await session.refresh(add_product)
            return {"message": "Product added successfully"}
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"Error adding product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products", response_model=PaginatedProductResponse, status_code=status.HTTP_200_OK)
async def fetch_products(user: user_dependency, db: db_dependency, search: str = None, page: int = 1, limit: int = 10):
    require_admin(user)
    try:
        skip = (page - 1) * limit
        async with db as session:
            query = select(models.Products).filter(models.Products.user_id == user.get("id"))
            if search:
                query = query.filter(models.Products.name.ilike(f"%{search}%"))
                logger.info(f"Admin product search query: {search}")
            # Count query for total
            count_query = select(func.count()).select_from(query.subquery())
            count_result = await session.execute(count_query)
            total = count_result.scalar()
            # Paginated query
            query = query.offset(skip).limit(limit)
            result = await session.execute(query)
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
        async with db as session:
            result = await session.execute(
                select(models.Products).filter(
                    models.Products.id == product_id,
                    models.Products.user_id == user.get("id")
                )
            )
            product = result.scalars().first()
            if not product:
                raise HTTPException(status_code=404, detail="Product not found")
            update_dict = updated_data.dict(exclude_unset=True)
            for key, value in update_dict.items():
                setattr(product, key, value)
            await session.commit()
            await session.refresh(product)
            return {"message": "Product updated successfully"}
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"Error updating product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-product/{product_id}", status_code=status.HTTP_200_OK)
async def delete_product(product_id: int, db: db_dependency, user: user_dependency):
    require_admin(user)
    try:
        async with db as session:
            result = await session.execute(
                select(models.Products).filter(
                    models.Products.id == product_id,
                    models.Products.user_id == user.get("id")
                )
            )
            product = result.scalars().first()
            if not product:
                raise HTTPException(status_code=404, detail="Product not found")
            result = await session.execute(
                select(models.OrderDetails).filter(models.OrderDetails.product_id == product_id)
            )
            order_details = result.scalars().first()
            if order_details:
                raise HTTPException(
                    status_code=400, detail="Cannot delete product with existing orders")
            await session.delete(product)
            await session.commit()
            return {"message": "Product deleted successfully"}
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"Error deleting product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_order", status_code=status.HTTP_201_CREATED)
async def create_order(db: db_dependency, user: user_dependency, order_payload: CartPayload):
    try:
        async with db as session:
            new_order = models.Orders(user_id=user.get("id"), total=0)
            session.add(new_order)
            await session.commit()
            await session.refresh(new_order)

            total_cost = Decimal('0')
            for item in order_payload.cart:
                result = await session.execute(
                    select(models.Products).filter(models.Products.id == item.id)
                )
                product = result.scalars().first()
                if not product:
                    await session.rollback()
                    raise HTTPException(
                        status_code=404, detail=f"Product ID {item.id} not found")
                quantity = Decimal(str(item.quantity))
                if product.stock_quantity < quantity:
                    await session.rollback()
                    raise HTTPException(
                        status_code=400, detail=f"Insufficient stock for product {product.name}")

                order_detail = models.OrderDetails(
                    order_id=new_order.order_id,
                    product_id=product.id,
                    quantity=quantity,
                    total_price=product.price * quantity,
                )
                total_cost += order_detail.total_price
                product.stock_quantity -= quantity
                session.add(order_detail)

            new_order.total = total_cost
            await session.commit()

            logger.info(
                f"Order {new_order.order_id} created for user {user.get('id')}")
            return {
                "message": "Order created successfully",
                "order_id": new_order.order_id,
            }
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"Error creating order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        await session.rollback()
        logger.error(f"Invalid quantity value: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid quantity value")

@app.get("/orders", response_model=List[OrderResponse], status_code=status.HTTP_200_OK)
async def fetch_orders(user: user_dependency, db: db_dependency, skip: int = 0, limit: int = 10):
    try:
        async with db as session:
            result = await session.execute(
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
                        logger.warning(
                            f"Order {order.order_id} has order_detail {detail.order_detail_id} with missing product (product_id={detail.product_id})")
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
        async with db as session:
            total_sales_result = await session.execute(
                select(func.sum(models.Orders.total)).filter(models.Orders.user_id == id)
            )
            total_sales = total_sales_result.scalar() or 0

            total_products_result = await session.execute(
                select(func.count(models.Products.id)).filter(models.Products.user_id == id)
            )
            total_products = total_products_result.scalar() or 0

            today_sale_result = await session.execute(
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