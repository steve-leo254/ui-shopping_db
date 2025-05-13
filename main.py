from fastapi import FastAPI, HTTPException, Depends, status, Path, File, UploadFile
from pydantic_model import ProductsBase, CartPayload, CartItem, UpdateProduct
from typing import Annotated
import models
from database import engine, db_dependency
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError
import auth
from auth import get_active_user
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
from pathlib import Path
import shutil
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()
app.include_router(auth.router)

# Static files for image uploads
UPLOAD_FOLDER = Path("static/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_dependency = Annotated[dict, Depends(get_active_user)]

# Async table creation
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

# Run table creation at startup
@app.on_event("startup")
async def startup_event():
    await create_tables()

@app.get("/")
async def user(user: user_dependency, db: AsyncSession = Depends(db_dependency)):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication Failed")
    return {"user": user}

@app.post("/products")
async def add_product(
    user: user_dependency,
    db: AsyncSession = Depends(db_dependency),
    create_product: ProductsBase = Depends(),
    image: UploadFile = File(...),
):
    try:
        if not allowed_file(image.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        if (await db.execute(select(models.Products).filter_by(barcode=create_product.barcode))).scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Barcode already exists")
        
        category = await db.get(models.Categories, create_product.category_id)
        if not category:
            raise HTTPException(status_code=400, detail="Invalid category ID")

        filename = f"{user.get('id')}_{int(datetime.now(timezone.utc).timestamp())}_{image.filename}"
        file_path = UPLOAD_FOLDER / filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        image_url = f"/static/uploads/{filename}"

        add_product = models.Products(
            name=create_product.name,
            brand=create_product.brand,
            price=create_product.price,
            cost=create_product.cost,
            image_url=image_url,
            stock_quantity=create_product.stock_quantity,
            barcode=create_product.barcode,
            description=create_product.description,
            category_id=create_product.category_id,
            user_id=user.get("id"),
        )
        db.add(add_product)
        await db.commit()
        await db.refresh(add_product)

        stock_entry = models.Stock(
            product_id=add_product.id,
            quantity=add_product.stock_quantity,
            updated_at=datetime.now(timezone.utc)
        )
        db.add(stock_entry)
        await db.commit()

        logger.info(f"Product created: {add_product.name} by user {user.get('id')}")
        return {"message": "Product added successfully", "product_id": add_product.id}
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error creating product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products")
async def fetch_products(user: user_dependency, db: AsyncSession = Depends(db_dependency)):
    try:
        products = (await db.execute(
            select(models.Products).filter(models.Products.user_id == user.get("id"))
        )).scalars().all()
        return [
            {
                "id": p.id,
                "name": p.name,
                "brand": p.brand,
                "cost": float(p.cost),
                "price": float(p.price),
                "stock_quantity": p.stock_quantity,
                "barcode": p.barcode,
                "image_url": p.image_url,
                "description": p.description,
                "category_id": p.category_id,
            }
            for p in products
        ]
    except SQLAlchemyError as e:
        logger.error(f"Error fetching products: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching products: {str(e)}")

@app.post("/create_order")
async def create_order(
    user: user_dependency,
    order_payload: CartPayload,
    db: AsyncSession = Depends(db_dependency)
):
    cart_items = order_payload.cart
    try:
        new_order = models.Orders(
            user_id=user.get("id"),
            total=0,
            datetime=datetime.now(timezone.utc)
        )
        db.add(new_order)
        await db.commit()
        await db.refresh(new_order)

        total_cost = 0
        for item in cart_items:
            product = (await db.execute(
                select(models.Products).filter_by(id=item.id)
            )).scalar_one_or_none()
            if not product:
                raise HTTPException(status_code=404, detail=f"Product ID {item.id} not found")
            if product.stock_quantity < item.quantity:
                raise HTTPException(status_code=400, detail=f"Insufficient stock for product {item.id}")

            order_detail = models.OrderDetails(
                order_id=new_order.order_id,
                product_id=product.id,
                quantity=item.quantity,
                total_price=product.price * item.quantity,
            )
            total_cost += order_detail.total_price
            product.stock_quantity -= item.quantity
            db.add(order_detail)
            db.add(product)

            stock = (await db.execute(
                select(models.Stock).filter_by(product_id=product.id)
            )).scalar_one_or_none()
            if stock:
                stock.quantity = product.stock_quantity
                stock.updated_at = datetime.now(timezone.utc)
                db.add(stock)

        new_order.total = total_cost
        db.add(new_order)
        await db.commit()

        logger.info(f"Order created: {new_order.order_id} by user {user.get('id')}")
        return {"message": "Order created successfully", "order_id": new_order.order_id}
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error creating order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create order: {str(e)}")

@app.get("/orders")
async def fetch_orders(user: user_dependency, db: AsyncSession = Depends(db_dependency)):
    try:
        orders = (await db.execute(
            select(models.Orders).filter(models.Orders.user_id == user.get("id"))
        )).scalars().all()
        return [
            {
                "order_id": o.order_id,
                "user_id": o.user_id,
                "total": float(o.total),
                "datetime": o.datetime
            }
            for o in orders
        ]
    except SQLAlchemyError as e:
        logger.error(f"Error fetching orders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching orders: {str(e)}")

@app.get("/dashboard")
async def dashboard(user: user_dependency, db: AsyncSession = Depends(db_dependency)):
    user_id = user.get("id")
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    try:
        total_sales = (await db.execute(
            select(func.sum(models.Orders.total))
        )).scalar() or 0

        total_products = (await db.execute(
            select(func.count(models.Products.id))
        )).scalar() or 0

        sales_per_user = (await db.execute(
            select(models.Orders.user_id, func.sum(models.Orders.total).label("total_sales"))
            .filter(models.Orders.user_id == user_id)
            .group_by(models.Orders.user_id)
        )).all()
        user_sale = {uid: float(ts) for uid, ts in sales_per_user}.get(user_id, 0)

        sales_today = (await db.execute(
            select(
                models.Orders.user_id,
                func.sum(models.Orders.total).label("total_revenue"),
                func.count(models.Orders.order_id).label("total_sales")
            )
            .filter(func.date(models.Orders.datetime) == today)
            .group_by(models.Orders.user_id)
        )).all()

        today_sale_per_user = (await db.execute(
            select(func.sum(models.Orders.total))
            .filter(models.Orders.user_id == user_id, func.date(models.Orders.datetime) == today)
        )).scalar() or 0

        sales_yesterday = (await db.execute(
            select(models.Orders.user_id, func.sum(models.Orders.total).label("total_revenue"))
            .filter(func.date(models.Orders.datetime) == yesterday)
            .group_by(models.Orders.user_id)
        )).all()
        yesterday_sales_dict = {s.user_id: s.total_revenue for s in sales_yesterday}

        sales_data = []
        for sale in sales_today:
            previous_revenue = yesterday_sales_dict.get(sale.user_id, 0)
            percentage_change = (
                ((sale.total_revenue - previous_revenue) / previous_revenue * 100)
                if previous_revenue > 0 else (100 if sale.total_revenue > 0 else 0)
            )
            sales_data.append({
                "user_id": sale.user_id,
                "total_revenue": float(sale.total_revenue),
                "number_of_sales": sale.total_sales,
                "percentage_change": round(percentage_change, 2),
                "change_type": "increase" if percentage_change > 0 else "decrease" if percentage_change < 0 else "neutral"
            })

        top_salesmen = sorted(sales_data, key=lambda x: x["total_revenue"], reverse=True)[:5]

        logger.info(f"Dashboard data fetched for user {user_id}")
        return {
            "total_sales": float(total_sales),
            "total_products": total_products,
            "sales_per_user": user_sale,
            "top_salesmen": top_salesmen,
            "todaySalePerUser": float(today_sale_per_user),
        }
    except SQLAlchemyError as e:
        logger.error(f"Error fetching dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard data: {str(e)}")

@app.put("/update-product/{product_id}")
async def update_product(
    product_id: int,
    user: user_dependency,
    db: AsyncSession = Depends(db_dependency),
    updated_data: UpdateProduct = Depends(),
    image: UploadFile = File(None),
):
    try:
        product = (await db.execute(
            select(models.Products).filter(
                models.Products.id == product_id,
                models.Products.user_id == user.get("id")
            )
        )).scalar_one_or_none()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found or unauthorized")

        update_dict = updated_data.dict(exclude_unset=True)
        if "barcode" in update_dict:
            existing = (await db.execute(
                select(models.Products).filter(
                    models.Products.barcode == update_dict["barcode"],
                    models.Products.id != product_id
                )
            )).scalar_one_or_none()
            if existing:
                raise HTTPException(status_code=400, detail="Barcode already exists")

        if "category_id" in update_dict:
            category = await db.get(models.Categories, update_dict["category_id"])
            if not category:
                raise HTTPException(status_code=400, detail="Invalid category ID")

        if image and allowed_file(image.filename):
            filename = f"{user.get('id')}_{int(datetime.now(timezone.utc).timestamp())}_{image.filename}"
            file_path = UPLOAD_FOLDER / filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            update_dict["image_url"] = f"/static/uploads/{filename}"

        for key, value in update_dict.items():
            setattr(product, key, value)

        if "stock_quantity" in update_dict:
            stock = (await db.execute(
                select(models.Stock).filter_by(product_id=product.id)
            )).scalar_one_or_none()
            if stock:
                stock.quantity = update_dict["stock_quantity"]
                stock.updated_at = datetime.now(timezone.utc)
            else:
                stock = models.Stock(
                    product_id=product.id,
                    quantity=update_dict["stock_quantity"],
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(stock)

        await db.commit()
        await db.refresh(product)
        logger.info(f"Product updated: {product_id} by user {user.get('id')}")
        return {"message": "Product updated successfully"}
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error updating product: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating product: {str(e)}")

@app.delete("/delete-product/{product_id}")
async def delete_product(
    product_id: int,
    user: user_dependency,
    db: AsyncSession = Depends(db_dependency),
):
    try:
        product = (await db.execute(
            select(models.Products).filter(
                models.Products.id == product_id,
                models.Products.user_id == user.get("id")
            )
        )).scalar_one_or_none()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found or unauthorized")

        stock = (await db.execute(
            select(models.Stock).filter_by(product_id=product.id)
        )).scalar_one_or_none()
        if stock:
            await db.delete(stock)

        await db.delete(product)
        await db.commit()
        logger.info(f"Product deleted: {product_id} by user {user.get('id')}")
        return {"message": "Product deleted successfully"}
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error deleting product: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)