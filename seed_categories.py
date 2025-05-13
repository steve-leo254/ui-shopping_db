# seed_categories.py
from main import app, db
from models import Categories

with app.app_context():
    categories = [
        {"code": "TV", "name": "TV/Monitors", "description": "Televisions and monitors"},
        {"code": "PC", "name": "PC", "description": "Personal computers and accessories"},
        {"code": "GA", "name": "Gaming/Console", "description": "Gaming consoles and accessories"},
        {"code": "PH", "name": "Phones", "description": "Mobile phones and accessories"},
    ]
    
    for cat in categories:
        if not Categories.query.filter_by(code=cat["code"]).first():
            new_category = Categories(
                code=cat["code"],
                name=cat["name"],
                description=cat["description"]
            )
            db.session.add(new_category)
    
    db.session.commit()
    print("Categories added successfully.")