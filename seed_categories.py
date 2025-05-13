from sqlalchemy import select
from database import async_session
from models import Categories
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def seed_categories():
    async with async_session() as session:
        async with session.begin():
            categories = [
                {
                    "code": "TV",
                    "name": "TV/Monitors",
                    "description": "Televisions and monitors"
                },
                {
                    "code": "PC",
                    "name": "PC",
                    "description": "Personal computers and accessories"
                },
                {
                    "code": "GA",
                    "name": "Gaming/Console",
                    "description": "Gaming consoles and accessories"
                },
                {
                    "code": "PH",
                    "name": "Phones",
                    "description": "Mobile phones and accessories"
                },
            ]

            for cat in categories:
                # Check if category already exists by code
                existing = await session.execute(
                    select(Categories).filter_by(code=cat["code"])
                )
                if not existing.scalar_one_or_none():
                    new_category = Categories(
                        code=cat["code"],
                        name=cat["name"],
                        description=cat["description"]
                    )
                    session.add(new_category)
                    logger.info(f"Added category: {cat['name']} (code: {cat['code']})")
                else:
                    logger.info(f"Category already exists: {cat['name']} (code: {cat['code']})")

            await session.commit()
            logger.info("Categories added successfully.")

if __name__ == "__main__":
    asyncio.run(seed_categories())