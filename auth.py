from fastapi import APIRouter, Depends, HTTPException, status
from pydantic_model import CreateUserRequest, LoginUserRequest, Token
from database import db_dependency
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models import Users
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Annotated
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/auth", tags=["auth"])
bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_bearer = OAuth2PasswordBearer(tokenUrl="auth/login")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secure-secret-key")
ALGORITHM = "HS256"

async def get_active_user(token: Annotated[str, Depends(oauth2_bearer)], db: db_dependency):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("id")
        if username is None or user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return {"username": username, "id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

@router.post("/register")
async def register_user(request: CreateUserRequest, db: db_dependency):
    user = await db.execute(select(Users).filter_by(email=request.email))
    if user.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = bcrypt_context.hash(request.password)
    db_user = Users(username=request.username, email=request.email, hashed_password=hashed_password)
    db.add(db_user)
    await db.commit()
    return {"message": "User created successfully"}

@router.post("/login", response_model=Token)
async def login_user(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: db_dependency):
    user = await db.execute(select(Users).filter_by(email=form_data.username))
    user = user.scalar_one_or_none()
    if not user or not bcrypt_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = jwt.encode(
        {"sub": user.username, "id": user.id, "exp": datetime.utcnow() + timedelta(minutes=30)},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return {"access_token": access_token, "token_type": "bearer"}